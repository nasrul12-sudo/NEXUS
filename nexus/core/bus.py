"""ZeroMQ-based event bus.

Design:
  - PUB/SUB untuk event stream (fire-and-forget, drop OK).
  - Multipart message: [topic_frame, payload_frame].
    - topic_frame memungkinkan ZMQ SUB filter bekerja.
    - payload_frame berisi JSON envelope.
  - Slow joiner problem ditangani di level aplikasi (supervisor).
  - ZMQ socket TIDAK thread-safe. Publisher pakai lock untuk publish
    dari multiple thread (opsional). Subscriber single-thread.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import zmq

from nexus.core.config import get_config
from nexus.core.errors import BusError
from nexus.core.events import EventEnvelope
from nexus.core.logging import get_logger


log = get_logger(__name__)


# ---------- Publisher ----------

class Publisher:
    """Thread-safe PUB socket wrapper.

    Lock melindungi send_multipart dari race condition di multi-thread
    publish (ZMQ internal tidak atomic untuk multipart).
    """

    def __init__(self, endpoint: str, hwm: int = 1000):
        self._endpoint = endpoint
        self._lock = threading.Lock()
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.set_hwm(hwm)
        self._sock.setsockopt(zmq.LINGER, 0)
        try:
            self._sock.bind(endpoint)
        except zmq.ZMQError as e:
            raise BusError(f"Failed to bind publisher to {endpoint}: {e}") from e
        self._closed = False
        log.debug("bus.publisher.bind", endpoint=endpoint)

    def publish(self, event: EventEnvelope) -> None:
        """Publish event sebagai multipart [topic, payload]."""
        if self._closed:
            raise BusError("Publisher is closed")
        with self._lock:
            try:
                self._sock.send_multipart(
                    event.to_wire_multipart(), flags=zmq.DONTWAIT
                )
            except zmq.Again:
                # HWM reached — drop event (acceptable untuk real-time stream)
                log.warning("bus.publish.dropped", type=event.type)
            except zmq.ZMQError as e:
                raise BusError(f"Publish failed: {e}") from e

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._sock.close(linger=0)
            self._closed = True
        log.debug("bus.publisher.close", endpoint=self._endpoint)

    def __enter__(self) -> Publisher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------- Subscriber ----------

class Subscriber:
    """SUB socket wrapper. Not thread-safe — one per thread."""

    def __init__(
        self,
        endpoint: str,
        topics: list[str] | None = None,
        recv_timeout_ms: int = 100,
    ):
        self._endpoint = endpoint
        self._recv_timeout_ms = recv_timeout_ms
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        try:
            self._sock.connect(endpoint)
        except zmq.ZMQError as e:
            raise BusError(f"Failed to connect subscriber to {endpoint}: {e}") from e

        # Subscribe ke topic prefix.
        # Empty string = semua event.
        # ZMQ filter bekerja pada frame PERTAMA (topic_frame).
        for topic in (topics or [""]):
            self._sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        self._closed = False
        log.debug(
            "bus.subscriber.connect",
            endpoint=endpoint,
            topics=topics or ["*"],
        )

    @property
    def handshake_ok(self) -> bool:
        """True jika socket sudah connect dan subscribe.

        Catatan: ini TIDAK menjamin publisher sudah siap atau
        subscription sudah terpropagasi. Untuk sinkronisasi, gunakan
        recv_with_timeout() atau startup ordering di supervisor.
        """
        return not self._closed

    def recv(self) -> EventEnvelope | None:
        """Non-blocking receive dengan timeout dari config."""
        if self._closed:
            raise BusError("Subscriber is closed")
        try:
            frames = self._sock.recv_multipart()
        except zmq.Again:
            return None
        except zmq.ZMQError as e:
            raise BusError(f"Recv failed: {e}") from e
        try:
            return EventEnvelope.from_wire_multipart(frames)
        except Exception as e:
            log.error("bus.recv.parse_error", error=str(e), frame_count=len(frames))
            return None

    def recv_with_timeout(self, timeout: float = 1.0) -> EventEnvelope | None:
        """Retry recv sampai timeout.

        Args:
            timeout: total waktu tunggu (detik)

        Returns:
            Event atau None jika timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            evt = self.recv()
            if evt is not None:
                return evt
        return None

    def recv_nowait(self) -> EventEnvelope | None:
        """Fully non-blocking receive."""
        try:
            frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        except zmq.ZMQError as e:
            raise BusError(f"Recv failed: {e}") from e
        try:
            return EventEnvelope.from_wire_multipart(frames)
        except Exception as e:
            log.error("bus.recv.parse_error", error=str(e), frame_count=len(frames))
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._sock.close(linger=0)
        self._closed = True
        log.debug("bus.subscriber.close", endpoint=self._endpoint)

    def __enter__(self) -> Subscriber:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------- EventBus facade ----------

class EventBus:
    """High-level facade untuk publish/subscribe."""

    def __init__(self, endpoint: str | None = None, hwm: int | None = None):
        cfg = get_config()
        self._endpoint = endpoint or cfg.bus.publish_endpoint
        self._hwm = hwm or cfg.bus.hwm
        self._recv_timeout = cfg.bus.recv_timeout_ms

    @contextmanager
    def publisher(self) -> Iterator[Publisher]:
        pub = Publisher(self._endpoint, hwm=self._hwm)
        try:
            yield pub
        finally:
            pub.close()

    @contextmanager
    def subscriber(self, topics: list[str] | None = None) -> Iterator[Subscriber]:
        sub = Subscriber(
            self._endpoint, topics=topics, recv_timeout_ms=self._recv_timeout
        )
        try:
            yield sub
        finally:
            sub.close()

    def subscribe_loop(
        self,
        handler: Callable[[EventEnvelope], None],
        topics: list[str] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Blocking subscribe loop. Calls handler for each event."""
        stop_event = stop_event or threading.Event()
        with self.subscriber(topics=topics) as sub:
            log.info("bus.subscribe_loop.start", topics=topics or ["*"])
            while not stop_event.is_set():
                try:
                    evt = sub.recv()
                except BusError as e:
                    log.error("bus.subscribe_loop.error", error=str(e))
                    time.sleep(0.1)
                    continue
                if evt is None:
                    continue
                try:
                    handler(evt)
                except Exception:
                    log.exception("bus.handler.error", event_type=evt.type)
            log.info("bus.subscribe_loop.stop")