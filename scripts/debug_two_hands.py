"""Debug 2 tangan + mode manager — print setiap frame."""
import sys
import time
import signal

import cv2
import numpy as np

from nexus.brain import NexusMode, get_mode_manager
from nexus.core.config import load_config, set_config
from nexus.core.logging import setup_logging
from nexus.vision.camera import OpenCVCameraProvider
from nexus.vision.debug.overlay import draw_hand_skeleton, draw_hand_label
from nexus.vision.gesture import GestureEngine
from nexus.vision.gesture.features import hand_center
from nexus.vision.gesture.rules import classify_static
from nexus.vision.hand_tracking import MediaPipeHandTracker


MODE_COLORS = {
    NexusMode.IDLE: (128, 128, 128),
    NexusMode.LISTENING: (0, 255, 0),
    NexusMode.PROCESSING: (0, 165, 255),
    NexusMode.CONFIRMING: (0, 255, 255),
}


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

    mode_manager = get_mode_manager()
    mode_manager.start(fps=15.0)

    stop = False
    def handler(s, f):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handler)

    print("=" * 70)
    print("  DEBUG: 2 tangan + mode manager")
    print("=" * 70)
    print()
    print("  IDLE: tunjukkan 2 tangan OPEN_PALM (tahan 1 detik) → wake")
    print("  LISTENING: gesture diproses")
    print("  Diam 30 detik → kembali IDLE")
    print("  Tunjukkan 2 tangan OPEN_PALM lagi → exit ke IDLE")
    print()
    print("=" * 70)

    frame_id = 0
    start_time = time.time()

    while not stop:
        frame = camera.read()
        if frame is None:
            continue
        frame_id += 1

        hands = tracker.detect(frame.data)
        canvas = frame.data.copy()

        current_mode = mode_manager.mode

        # ============================================================
        # STEP 1: WAKE GESTURE CHECK (butuh semua tangan)
        # ============================================================
        wake_confirmed = engine.check_wake_gesture(hands, frame.timestamp)
        if wake_confirmed:
            print(f"        >>> WAKE GESTURE CONFIRMED — mode: {mode_manager.mode.value}")

        # ============================================================
        # STEP 2: Process individual hands
        # ============================================================
        raw_gestures = {}
        confirmed_gestures = {}

        for hand in hands:
            draw_hand_skeleton(canvas, hand)
            draw_hand_label(canvas, hand)

            # Raw classification (untuk debug)
            g, c = classify_static(
                hand,
                pinch_threshold=0.35,
                extended_curvature_max=0.25,
                folded_curvature_min=0.40,
                half_folded_curvature_min=0.25,
                min_finger_spread_open=0.75,
            )
            raw_gestures[hand.handedness] = (g.value, c)

            # Engine (dengan smoother + mode filter)
            smoothed_list = engine.process_hand(hand, frame.timestamp)
            for sm in smoothed_list:
                confirmed_gestures[hand.handedness] = sm.gesture.value

        # Print setiap 5 frame
        if frame_id % 5 == 0:
            n = len(hands)
            hand_info = ", ".join(f"{h.handedness}({h.confidence:.2f})" for h in hands)
            raw_info = ", ".join(f"{k}:{v[0]}" for k, v in raw_gestures.items())
            conf_info = ", ".join(f"{k}:{v}" for k, v in confirmed_gestures.items())
            wake_frames = mode_manager._wake_gesture_frames
            wake_needed = mode_manager._wake_gesture_needed

            elapsed = time.time() - start_time
            fps = frame_id / elapsed if elapsed > 0 else 0

            print(f"[{frame_id:4d}] mode={current_mode.value:11s} "
                  f"wake={wake_frames:2d}/{wake_needed:2d} "
                  f"fps={fps:.1f} "
                  f"hands={n} [{hand_info}]")
            if raw_info:
                print(f"          raw=[{raw_info}]")
            if conf_info:
                print(f"          CONFIRMED=[{conf_info}]")

        # Draw mode indicator
        color = MODE_COLORS.get(current_mode, (255, 255, 255))
        cv2.putText(
            canvas,
            f"MODE: {current_mode.value}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # Draw wake progress
        if current_mode == NexusMode.IDLE and mode_manager._wake_gesture_frames > 0:
            progress = mode_manager._wake_gesture_frames / max(mode_manager._wake_gesture_needed, 1)
            bar_width = int(300 * progress)
            cv2.rectangle(canvas, (10, 50), (310, 70), (50, 50, 50), -1)
            cv2.rectangle(canvas, (10, 50), (10 + bar_width, 70), (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f"Wake hold: {mode_manager._wake_gesture_frames}/{mode_manager._wake_gesture_needed}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Draw timeout info
        if current_mode == NexusMode.LISTENING:
            elapsed = time.time() - mode_manager._mode_entered_ts
            remaining = max(0, 30 - elapsed)
            cv2.putText(
                canvas,
                f"Timeout in: {remaining:.1f}s",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        cv2.imshow("NEXUS Mode Debug", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()
    tracker.close()
    mode_manager.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()