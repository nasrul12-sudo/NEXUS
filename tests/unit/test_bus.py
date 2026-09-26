"""Test untuk event bus.

Catatan: PUB/SUB ZeroMQ memiliki "slow joiner problem" — event yang
dipublish sebelum subscriber benar-benar terdaftar akan di-drop.
Solusi untuk test: gunakan `recv_with_timeout()` yang retry.

Untuk production, slow joiner ditangani oleh supervisor yang
memastikan subscriber sudah ready sebelum publisher mulai.
"""
import time

import pytest

from nexus.core.bus import Publisher, Subscriber
from nexus.core.events import make_event, EventType


# Waktu tunggu maksimum untuk event sampai (retry-based)
RECV_TIMEOUT = 2.0


def _publish_and_recv(pub, sub, event, timeout=RECV_TIMEOUT):
    """Helper: publish event dan tunggu sampai diterima.

    Karena slow joiner problem, kita perlu retry publish beberapa kali
    atau tunggu subscription propagation. Pendekatan paling andal:
    publish beberapa kali sampai subscriber menerima.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        pub.publish(event)
        received = sub.recv_with_timeout(timeout=0.1)
        if received is not None:
            return received
    return None


def test_pub_sub_roundtrip(tmp_path):
    endpoint = f"ipc://{tmp_path}/test.sock"
    with Publisher(endpoint) as pub:
        with Subscriber(endpoint, topics=["gesture."]) as sub:
            assert sub.handshake_ok
            evt = make_event(EventType.GESTURE_STABLE, "vision", {"gesture": "PINCH"})
            received = _publish_and_recv(pub, sub, evt)
            assert received is not None, "Event tidak diterima dalam timeout"
            assert received.type == "gesture.stable"
            assert received.payload["gesture"] == "PINCH"


def test_topic_filter(tmp_path):
    endpoint = f"ipc://{tmp_path}/test2.sock"
    with Publisher(endpoint) as pub:
        with Subscriber(endpoint, topics=["voice."]) as sub:
            assert sub.handshake_ok
            # Publish event yang TIDAK match topic
            pub.publish(make_event(EventType.GESTURE_STABLE, "vision", {}))
            # Publish event yang match — retry sampai diterima
            evt = make_event(EventType.VOICE_TRANSCRIPT, "voice", {"text": "hi"})
            received = _publish_and_recv(pub, sub, evt)
            assert received is not None
            assert received.type == "voice.transcript"


def test_multiple_subscribers(tmp_path):
    endpoint = f"ipc://{tmp_path}/test3.sock"
    with Publisher(endpoint) as pub:
        subs = [Subscriber(endpoint, topics=["test."]) for _ in range(3)]
        try:
            for sub in subs:
                assert sub.handshake_ok
            # Publish sampai SEMUA subscriber menerima
            evt = make_event("test.hello", "system", {"n": 1})
            deadline = time.time() + RECV_TIMEOUT
            received_by = [None] * 3
            while time.time() < deadline and any(r is None for r in received_by):
                pub.publish(evt)
                for i, sub in enumerate(subs):
                    if received_by[i] is None:
                        r = sub.recv_with_timeout(timeout=0.05)
                        if r is not None:
                            received_by[i] = r
            for i, r in enumerate(received_by):
                assert r is not None, f"Subscriber {i} tidak menerima event"
                assert r.type == "test.hello"
        finally:
            for sub in subs:
                sub.close()


def test_publish_without_subscriber_does_not_raise(tmp_path):
    endpoint = f"ipc://{tmp_path}/test4.sock"
    with Publisher(endpoint) as pub:
        # Publish tanpa subscriber tidak boleh raise
        pub.publish(make_event("test.hello", "system", {}))
        # Tidak ada cara untuk tahu subscriber count dengan PUB sederhana.
        # Test ini hanya memverifikasi tidak ada exception.
        time.sleep(0.1)