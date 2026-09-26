"""Debug swipe detection — print hand movement & swipe candidates.

Jalankan, lalu lakukan gerakan swipe (kanan/kiri/atas/bawah).
Perhatikan output — apakah swipe terdeteksi?
"""
from __future__ import annotations

import signal
import sys
import time

import cv2
import numpy as np

from nexus.core.config import load_config, set_config
from nexus.core.logging import setup_logging
from nexus.vision.camera import OpenCVCameraProvider
from nexus.vision.debug.overlay import draw_hand_skeleton
from nexus.vision.gesture.features import hand_center
from nexus.vision.hand_tracking import MediaPipeHandTracker


def main() -> int:
    setup_logging(level="WARNING", process_name="debug_swipe")
    config = load_config(env="development")
    set_config(config)

    camera = OpenCVCameraProvider(
        device=config.vision.camera.device,
        width=config.vision.camera.width,
        height=config.vision.camera.height,
        fps=config.vision.camera.fps,
    )
    camera.open()

    tracker = MediaPipeHandTracker(
        model_path=config.vision.hand_tracking.model_path,
        max_hands=1,
        min_detection_confidence=0.7,
    )
    tracker.setup()

    print("Debug swipe. Lakukan gerakan swipe dengan tangan.")
    print("Tekan 'q' untuk quit.")
    print()

    # Custom swipe detector dengan debug
    from nexus.vision.gesture.rules import SwipeDetector
    gcfg = config.gesture
    detector = SwipeDetector(
        min_distance=gcfg.swipe_min_distance,
        max_duration_ms=gcfg.swipe_max_duration_ms,
        history_size=gcfg.swipe_history_size,
        min_velocity=getattr(gcfg, 'swipe_min_velocity', 0.3),
    )

    # Track last hand center untuk hitung raw velocity
    last_center = None
    last_ts = None

    stop = False
    def handler(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handler)

    frame_id = 0
    while not stop:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame_id += 1

        hands = tracker.detect(frame.data)
        canvas = frame.data.copy()

        if hands:
            hand = hands[0]
            draw_hand_skeleton(canvas, hand)

            now = frame.timestamp
            center = hand_center(hand.landmarks)[:2]

            # Raw velocity
            if last_center is not None and last_ts is not None:
                dt = now - last_ts
                if dt > 0:
                    velocity = (center - last_center) / dt
                    speed = float(np.linalg.norm(velocity))
                else:
                    velocity = np.zeros(2)
                    speed = 0.0
            else:
                velocity = np.zeros(2)
                speed = 0.0

            last_center = center
            last_ts = now

            # Update detector
            detector.update(now, center)

            # Try detect
            result = detector.detect(now)

            # Print setiap 5 frame
            if frame_id % 5 == 0:
                print(f"[{frame_id:4d}] center=({center[0]:.3f},{center[1]:.3f}) "
                      f"vel=({velocity[0]:+.2f},{velocity[1]:+.2f}) "
                      f"speed={speed:.2f} "
                      f"history={len(detector._history)}")
                if result is not None:
                    g, c = result
                    print(f"        >>> SWIPE DETECTED: {g.value} (conf={c:.2f})")

            # Draw info
            cv2.putText(canvas, f"center=({center[0]:.2f},{center[1]:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, f"speed={speed:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, f"history={len(detector._history)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if result is not None:
                g, c = result
                cv2.putText(canvas, f"SWIPE: {g.value} ({c:.2f})",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw trajectory
            for i, (ts, c) in enumerate(detector._history):
                h, w = canvas.shape[:2]
                px = int(c[0] * w)
                py = int(c[1] * h)
                cv2.circle(canvas, (px, py), 3, (255, 255, 0), -1)

        else:
            last_center = None
            last_ts = None

        cv2.imshow("Debug Swipe (q=quit)", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()
    tracker.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())