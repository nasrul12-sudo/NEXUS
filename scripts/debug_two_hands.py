"""Debug 2 tangan — print setiap frame gesture raw vs confirmed."""
import sys
import time
import signal

import cv2
import numpy as np

from nexus.core.config import load_config, set_config
from nexus.core.logging import setup_logging
from nexus.vision.camera import OpenCVCameraProvider
from nexus.vision.debug.overlay import draw_hand_skeleton, draw_hand_label
from nexus.vision.gesture import GestureEngine
from nexus.vision.gesture.features import hand_center
from nexus.vision.gesture.rules import classify_static
from nexus.vision.gesture.features import (
    all_finger_curvatures,
    finger_spread,
    pinch_distance_normalized,
)
from nexus.vision.hand_tracking import MediaPipeHandTracker


def main():
    setup_logging(level="WARNING")
    config = load_config(env="development")
    set_config(config)

    camera = OpenCVCameraProvider(
        device=config.vision.camera.device,
        width=640, height=480, fps=30,
    )
    camera.open()

    tracker = MediaPipeHandTracker(
        model_path=config.vision.hand_tracking.model_path,
        max_hands=2,
        min_detection_confidence=0.5,
    )
    tracker.setup()

    engine = GestureEngine()
    engine.setup()

    stop = False
    def handler(s, f):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handler)

    print("Debug 2 tangan (detail). Tunjukkan kedua tangan. Tekan 'q'.")
    print()
    print("Format: [frame] hands=[...] raw=[gesture(conf)] confirmed=[gesture]")
    print()

    frame_id = 0

    while not stop:
        frame = camera.read()
        if frame is None:
            continue
        frame_id += 1

        hands = tracker.detect(frame.data)
        canvas = frame.data.copy()

        # Kumpulkan info
        raw_gestures = {}
        confirmed_gestures = {}

        for hand in hands:
            draw_hand_skeleton(canvas, hand)
            draw_hand_label(canvas, hand)

            # === RAW classification (tanpa smoother) ===
            g, c = classify_static(
                hand,
                pinch_threshold=getattr(config.gesture, 'pinch_distance_threshold', 0.35),
                extended_curvature_max=getattr(config.gesture, 'extended_curvature_max', 0.25),
                folded_curvature_min=getattr(config.gesture, 'folded_curvature_min', 0.40),
                half_folded_curvature_min=getattr(config.gesture, 'half_folded_curvature_min', 0.25),
                min_finger_spread_open=getattr(config.gesture, 'min_finger_spread_open', 0.75),
            )
            raw_gestures[hand.handedness] = (g.value if g else "?", c)

            # === Engine (dengan smoother) ===
            smoothed_list = engine.process_hand(hand, frame.timestamp)
            for sm in smoothed_list:
                confirmed_gestures[hand.handedness] = sm.gesture.value

        # Print SETIAP FRAME
        n = len(hands)
        hand_info = ", ".join(f"{h.handedness}({h.confidence:.2f})" for h in hands)
        raw_info = ", ".join(f"{k}:{v[0]}({v[1]:.2f})" for k, v in raw_gestures.items())
        conf_info = ", ".join(f"{k}:{v}" for k, v in confirmed_gestures.items())

        print(f"[{frame_id:4d}] hands={n} [{hand_info}]")
        print(f"        raw=[{raw_info}]")
        if conf_info:
            print(f"        CONFIRMED=[{conf_info}]")

        cv2.imshow("Debug 2 Hands", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()