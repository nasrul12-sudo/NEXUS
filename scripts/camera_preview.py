#!/usr/bin/env python3
"""Camera Preview — debug tool untuk NEXUS vision pipeline.

Usage:
    python scripts/camera_preview.py                    # pakai config default
    python scripts/camera_preview.py --device 1         # pakai camera device 1
    python scripts/camera_preview.py --no-mirror        # disable mirror
    python scripts/camera_preview.py --width 1280 --height 720
    python scripts/camera_preview.py --no-mirror --show-landmark-names

Keyboard:
    q / ESC    quit
    s          snapshot (save PNG)
    r          toggle recording (save MP4)
    l          toggle hand labels
    k          toggle skeleton
    c          toggle crosshair
    m          toggle mirror
    h          toggle HUD

Output:
    Snapshot:   data/snapshots/snapshot_<timestamp>.png
    Recording:  data/recordings/recording_<timestamp>.mp4
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from nexus.core.config import get_config, load_config, set_config
from nexus.core.logging import setup_logging
from nexus.vision.camera import OpenCVCameraProvider
from nexus.vision.debug import CameraPreview, PreviewConfig
from nexus.vision.hand_tracking import MediaPipeHandTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEXUS Camera Preview — debug tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Camera
    parser.add_argument("--device", type=int, default=None,
                        help="Camera device index (default: dari config)")
    parser.add_argument("--width", type=int, default=None,
                        help="Camera width (default: dari config)")
    parser.add_argument("--height", type=int, default=None,
                        help="Camera height (default: dari config)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Camera FPS target (default: dari config)")

    # Preview
    parser.add_argument("--window-width", type=int, default=960,
                        help="Preview window width")
    parser.add_argument("--window-height", type=int, default=540,
                        help="Preview window height")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable mirror (selfie view)")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="Hide hand skeleton")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide handedness labels")
    parser.add_argument("--no-hud", action="store_true",
                        help="Hide HUD")
    parser.add_argument("--show-crosshair", action="store_true",
                        help="Show center crosshair")
    parser.add_argument("--show-landmark-names", action="store_true",
                        help="Show landmark names (WRIST, INDEX_TIP, etc)")

    # Hand tracking
    parser.add_argument("--max-hands", type=int, default=None,
                        help="Max hands to detect (1 or 2)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path ke hand_landmarker.task")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Config directory")
    parser.add_argument("--env", type=str, default="development",
                        help="Config environment (development/production/test)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    setup_logging(level=args.log_level, process_name="camera_preview")

    config = load_config(config_dir=args.config_dir, env=args.env)
    set_config(config)

    # Resolve vision config
    cam_cfg = config.vision.camera
    ht_cfg = config.vision.hand_tracking
    pose_cfg = getattr(config.vision, 'pose_tracking', None)

    device = args.device if args.device is not None else cam_cfg.device
    width = args.width if args.width is not None else cam_cfg.width
    height = args.height if args.height is not None else cam_cfg.height
    fps = args.fps if args.fps is not None else cam_cfg.fps
    max_hands = args.max_hands if args.max_hands is not None else ht_cfg.max_hands
    model_path = args.model_path if args.model_path is not None else ht_cfg.model_path

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"✗ Model tidak ditemukan: {model_file}", file=sys.stderr)
        print("  Jalankan: python scripts/download_models.py", file=sys.stderr)
        return 1

    # Camera
    print(f"Opening camera device={device} {width}x{height}@{fps}...")
    camera = OpenCVCameraProvider(
        device=device, width=width, height=height, fps=fps,
    )
    try:
        camera.open()
    except Exception as e:
        print(f"✗ Gagal buka camera: {e}", file=sys.stderr)
        return 1

    # Hand tracker
    print(f"Loading MediaPipe hand model: {model_file}")
    hand_tracker = MediaPipeHandTracker(
        model_path=model_path,
        max_hands=max_hands,
        min_detection_confidence=ht_cfg.min_detection_confidence,
        min_presence_confidence=ht_cfg.min_presence_confidence,
        min_tracking_confidence=ht_cfg.min_tracking_confidence,
    )
    try:
        hand_tracker.setup()
    except Exception as e:
        print(f"✗ Gagal setup hand tracker: {e}", file=sys.stderr)
        camera.close()
        return 1

    # Pose tracker (opsional)
    pose_tracker = None
    if pose_cfg is None or getattr(pose_cfg, 'enabled', True):
        pose_model = getattr(pose_cfg, 'model_path', 'data/models/pose_landmarker_lite.task') if pose_cfg else 'data/models/pose_landmarker_lite.task'
        if Path(pose_model).exists():
            print(f"Loading MediaPipe pose model: {pose_model}")
            from nexus.vision.pose import MediaPipePoseTracker
            pose_tracker = MediaPipePoseTracker(model_path=pose_model, num_poses=1)
            try:
                pose_tracker.setup()
            except Exception as e:
                print(f"⚠ Gagal setup pose tracker: {e}", file=sys.stderr)
                pose_tracker = None
        else:
            print(f"⚠ Pose model tidak ditemukan: {pose_model}", file=sys.stderr)
            print("  Pose features tidak akan tersedia. Jalankan: python scripts/download_models.py", file=sys.stderr)

    # Preview config
    preview_cfg = PreviewConfig(
        window_width=args.window_width,
        window_height=args.window_height,
        mirror=not args.no_mirror,
        show_skeleton=not args.no_skeleton,
        show_labels=not args.no_labels,
        show_hud=not args.no_hud,
        show_crosshair=args.show_crosshair,
        show_landmark_names=args.show_landmark_names,
    )

    preview = CameraPreview(
        camera,
        hand_tracker,
        preview_cfg,
        pose_tracker=pose_tracker,
    )

    # Signal handler
    def on_signal(signum, frame):
        print("\n[preview] Ctrl+C — stopping...", file=sys.stderr)
        preview.stop()

    signal.signal(signal.SIGINT, on_signal)

    exit_code = 0
    try:
        preview.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"✗ Preview error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        camera.close()
        hand_tracker.close()
        if pose_tracker is not None:
            pose_tracker.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())