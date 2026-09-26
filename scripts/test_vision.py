"""Manual test: jalankan VisionProcess dan print events.

Usage:
    python scripts/test_vision.py

Tekan Ctrl+C untuk stop.

Yang diharapkan:
  - Vision process spawn dan mulai capture camera
  - Setiap frame dengan tangan terdeteksi → event 'hand.landmarks'
  - Tangan hilang dari frame → event 'hand.lost'
  - Setiap ~10 detik → event 'vision.fps' dengan FPS report
"""
from __future__ import annotations

import signal
import sys
import threading
import time

from nexus.core.bus import EventBus
from nexus.core.config import load_config, set_config
from nexus.core.events import EventEnvelope
from nexus.core.logging import setup_logging
from nexus.launcher.supervisor import ProcessSpec, Supervisor
from nexus.vision.process import VisionProcess


def main() -> int:
    setup_logging(level="INFO")
    set_config(load_config(env="development"))

    bus = EventBus()
    stop = threading.Event()

    # Statistics
    stats = {
        "landmarks": 0,
        "lost": 0,
        "fps_reports": 0,
        "start": time.time(),
    }

    def on_event(evt: EventEnvelope) -> None:
        if evt.type == "hand.landmarks":
            stats["landmarks"] += 1
            payload = evt.payload
            n = payload["hand_count"]
            hands = payload["hands"]
            desc = ", ".join(
                f"{h['handedness']}({h['confidence']:.2f})" for h in hands
            )
            print(f"[hand.landmarks #{stats['landmarks']:5d}] {n} hand(s): {desc}")

        elif evt.type == "hand.lost":
            stats["lost"] += 1
            print(f"[hand.lost      #{stats['lost']:5d}] tangan hilang dari frame")

        elif evt.type == "vision.fps":
            stats["fps_reports"] += 1
            print(
                f"[vision.fps     #{stats['fps_reports']:5d}] "
                f"fps={evt.payload['fps']:.1f} "
                f"processed={evt.payload['frames_processed']} "
                f"dropped={evt.payload['frames_dropped']}"
            )

        elif evt.type == "process.ready":
            if evt.payload.get("name") == "vision":
                print(f"[process.ready] vision process PID={evt.payload.get('pid')}")

        elif evt.type == "process.error":
            if evt.payload.get("name") == "vision":
                print(f"[process.error] vision ERROR: {evt.payload.get('error')}")

    def subscribe_loop() -> None:
        with bus.subscriber(topics=["hand.", "vision.", "process."]) as sub:
            while not stop.is_set():
                try:
                    evt = sub.recv_with_timeout(timeout=0.1)
                except Exception as e:
                    print(f"[bus.error] {e}")
                    time.sleep(0.1)
                    continue
                if evt is not None:
                    on_event(evt)

    # Start subscriber thread
    sub_thread = threading.Thread(target=subscribe_loop, daemon=True)
    sub_thread.start()

    # Start vision process via supervisor
    supervisor = Supervisor()
    supervisor.register(ProcessSpec(
        "vision",
        VisionProcess,
        restart_on_crash=False,
    ))
    supervisor.start_all()

    print()
    print("=" * 60)
    print("  NEXUS Vision Test — Ctrl+C untuk stop")
    print("=" * 60)
    print()

    # Handle Ctrl+C
    def on_sigint(signum, frame):
        print("\n[main] Stopping...")
        stop.set()
        supervisor.stop_all()
        elapsed = time.time() - stats["start"]
        print()
        print("=" * 60)
        print(f"  Stats setelah {elapsed:.1f}s:")
        print(f"    hand.landmarks events: {stats['landmarks']}")
        print(f"    hand.lost events:      {stats['lost']}")
        print(f"    fps reports:           {stats['fps_reports']}")
        if elapsed > 0:
            print(f"    effective FPS:         {stats['landmarks'] / elapsed:.1f}")
        print("=" * 60)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    # Main loop — just wait
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        on_sigint(signal.SIGINT, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())