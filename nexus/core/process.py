"""Base class untuk semua NEXUS process.

Setiap process adalah subprocess Python terpisah dengan:
  - config loaded
  - logging configured
  - event bus publisher (untuk publish)
  - event bus subscriber (untuk consume, opsional)
  - state store access
  - resource manager access
  - graceful shutdown via SIGTERM/SIGINT + internal stop_event
  - publish PROCESS_READY saat startup
  - publish PROCESS_ERROR saat exception
  - publish PROCESS_SHUTDOWN saat exit
"""
from __future__ import annotations

import multiprocessing as mp
import signal
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from nexus.core.bus import EventBus, Publisher
from nexus.core.config import get_config
from nexus.core.events import EventEnvelope, EventType, make_event
from nexus.core.logging import get_logger, setup_logging


class NexusProcess(ABC):
    """Base class untuk semua NEXUS process."""

    #: Nama process (dipakai untuk logging & event source)
    name: str = "process"

    #: Source label untuk event (vision/voice/brain/action/system)
    source: str = "system"

    def __init__(self):
        self._stop_event = threading.Event()
        self._publisher: Publisher | None = None
        self._bus: EventBus | None = None
        self.log = get_logger(f"nexus.{self.name}")

    # ---------- Lifecycle hooks (override di subclass) ----------

    @abstractmethod
    def setup(self) -> None:
        """Setup resources. Dipanggil sekali sebelum run()."""

    @abstractmethod
    def run(self) -> None:
        """Main loop. Harus respect self._stop_event."""

    def teardown(self) -> None:
        """Cleanup resources. Override jika perlu."""

    # ---------- Public API ----------

    def stop(self) -> None:
        """Request graceful shutdown."""
        self.log.info("process.stop_requested")
        self._stop_event.set()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def publish(self, event: EventEnvelope) -> None:
        if self._publisher is None:
            raise RuntimeError("Publisher not initialized")
        self._publisher.publish(event)

    def emit(
        self,
        type: str | EventType,
        payload: dict[str, Any] | None = None,
        **meta: Any,
    ) -> None:
        """Convenience: build & publish event."""
        self.publish(make_event(
            type=type,
            source=self.source,  # type: ignore[arg-type]
            payload=payload,
            **meta,
        ))

    # ---------- Internal ----------

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.log.info("process.signal", signum=signum)
            self._stop_event.set()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _run(self) -> None:
        """Internal entry point (dipanggil di subprocess)."""
        setup_logging(process_name=self.name)
        self.log = get_logger(f"nexus.{self.name}")
        self._install_signal_handlers()

        self._bus = EventBus()

        try:
            self.log.info("process.starting", name=self.name)
            self.setup()

            # Publish ready AFTER setup
            with self._bus.publisher() as pub:
                self._publisher = pub
                pub.publish(make_event(
                    EventType.PROCESS_READY, self.source,  # type: ignore[arg-type]
                    {"name": self.name, "pid": __import__("os").getpid()},
                ))
                try:
                    self.run()
                except Exception as e:
                    self.log.exception("process.run_error")
                    pub.publish(make_event(
                        EventType.PROCESS_ERROR, self.source,  # type: ignore[arg-type]
                        {"name": self.name, "error": str(e), "type": type(e).__name__},
                    ))
                    raise
                finally:
                    pub.publish(make_event(
                        EventType.PROCESS_SHUTDOWN, self.source,  # type: ignore[arg-type]
                        {"name": self.name},
                    ))
        finally:
            try:
                self.teardown()
            except Exception:
                self.log.exception("process.teardown_error")
            self._publisher = None
            self.log.info("process.stopped", name=self.name)


def run_process(process: NexusProcess) -> None:
    """Run a NexusProcess in current process. Blocking.

    Digunakan untuk testing atau saat process sudah ada di current process.
    """
    process._run()


def _run_process_from_class(
    process_class: type[NexusProcess],
    args: tuple,
    kwargs: dict,
) -> None:
    """Entry point di child process.

    Construct NexusProcess dari class + args, lalu run.
    Ini menghindari pickle NexusProcess instance (yang punya threading.Event).
    """
    process = process_class(*args, **kwargs)
    process._run()


def spawn_process(
    process_class: type[NexusProcess],
    *args,
    **kwargs,
) -> mp.Process:
    """Spawn NexusProcess sebagai subprocess.

    Args:
        process_class: NexusProcess subclass (class, bukan instance)
        *args, **kwargs: constructor arguments. Harus pickle-able.

    Returns:
        mp.Process yang sudah start.

    Catatan:
        Menggunakan 'spawn' context untuk cross-platform consistency.
        Args/kwargs harus pickle-able. Jangan pass objek yang mengandung
        threading.Lock, socket, atau file handle.
    """
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_run_process_from_class,
        args=(process_class, args, kwargs),
        name=f"nexus-{process_class.name}",
        daemon=False,
    )
    p.start()
    return p