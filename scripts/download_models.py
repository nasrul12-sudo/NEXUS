"""Download MediaPipe models untuk NEXUS."""
import urllib.request
from pathlib import Path

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def main():
    target_dir = Path("data/models")
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        target = target_dir / name
        if target.exists() and target.stat().st_size > 100_000:
            size_mb = target.stat().st_size / 1024 / 1024
            print(f"✓ {name} already exists ({size_mb:.1f} MB)")
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, target)
        size_mb = target.stat().st_size / 1024 / 1024
        if size_mb < 0.1:
            print(f"✗ {name} corrupted ({size_mb:.2f} MB)")
            target.unlink()
            return 1
        print(f"✓ {name} saved ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())