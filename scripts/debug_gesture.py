"""Debug gesture — print curvatures & features untuk tiap frame.

Jalankan dengan tangan di depan kamera. Tekan 'q' untuk quit.

Output:
  - Curvature per jari (0 = lurus, 1 = folded)
  - Finger spread
  - Pinch distance
  - Thumb orientation
  - Palm facing
  - Gesture yang terdeteksi + confidence
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
from nexus.vision.gesture import Gesture
from nexus.vision.gesture.features import (
    all_finger_curvatures,
    finger_spread,
    palm_facing,
    pinch_distance_normalized,
    thumb_orientation_vertical,
)
from nexus.vision.gesture.rules import classify_static
from nexus.vision.hand_tracking import MediaPipeHandTracker


def main() -> int:
    setup_logging(level="WARNING", process_name="debug_gesture")
    config = load_config(env="development")
    set_config(config)

    # Camera
    camera = OpenCVCameraProvider(
        device=config.vision.camera.device,
        width=config.vision.camera.width,
        height=config.vision.camera.height,
        fps=config.vision.camera.fps,
    )
    camera.open()

    # Hand tracker
    tracker = MediaPipeHandTracker(
        model_path=config.vision.hand_tracking.model_path,
        max_hands=1,   # 1 tangan saja untuk clarity
        min_detection_confidence=0.7,
        min_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    tracker.setup()

    print("Debug gesture. Tangan di depan kamera. Tekan 'q' untuk quit.")
    print()

    stop = False

    def handler(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handler)

    # Threshold tuning
    tuning = {
        "pinch_threshold": 0.3,
        "fist_curvature_min": 0.7,
        "open_palm_curvature_max": 0.3,
        "point_curvature_max": 0.4,
        "min_finger_spread_open": 0.8,
    }

    frame_count = 0
    last_print_ts = 0.0

    while not stop:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        hands = tracker.detect(frame.data)

        canvas = frame.data.copy()

        if hands:
            hand = hands[0]
            draw_hand_skeleton(canvas, hand)

            # Print features 10x per second (biar tidak spam)
            now = time.time()
            if now - last_print_ts > 0.1:
                last_print_ts = now
                curv = all_finger_curvatures(hand.landmarks)
                spread = finger_spread(hand.landmarks)
                pinch = pinch_distance_normalized(hand.landmarks)
                thumb_ori = thumb_orientation_vertical(hand.landmarks)
                facing = palm_facing(hand.landmarks)

                gesture, conf = classify_static(
                    hand,
                    pinch_threshold=tuning["pinch_threshold"],
                    fist_curvature_min=tuning["fist_curvature_min"],
                    open_palm_curvature_max=tuning["open_palm_curvature_max"],
                    point_curvature_max=tuning["point_curvature_max"],
                    min_finger_spread_open=tuning["min_finger_spread_open"],
                )

                print(f"\n=== Frame {frame_count} ===")
                print(f"  Curvatures:")
                for name, c in curv.items():
                    bar = "█" * int(c * 20)
                    print(f"    {name:6s}: {c:.3f} {bar}")
                print(f"  Spread:      {spread:.3f}")
                print(f"  Pinch dist:  {pinch:.3f}")
                print(f"  Thumb dir:   {thumb_ori}")
                print(f"  Palm facing: {facing}")
                print(f"  → Gesture:   {gesture.value} (conf={conf:.2f})")

                # Draw gesture on canvas
                cv2.putText(
                    canvas,
                    f"{gesture.value} ({conf:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Debug Gesture (q=quit)", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()
    tracker.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())