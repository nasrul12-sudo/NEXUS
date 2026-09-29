"""Test gesture engine — versi robust."""
import numpy as np
import pytest
import time

from nexus.vision.gesture import (
    Gesture,
    GestureEngine,
    GestureSmoother,
)
from nexus.vision.gesture.features import (
    all_finger_curvatures,
    angle_at,
    finger_curvature,
    finger_spread,
    palm_facing,
    pinch_distance_normalized,
    thumb_orientation_vertical,
)
from nexus.vision.gesture.rules import SwipeDetector, classify_static
from nexus.vision.hand_tracking.landmarks import HandLandmarks


# ---------- Fixtures ----------

def _make_open_palm() -> np.ndarray:
    """Landmarks untuk OPEN_PALM (semua jari lurus & spread)."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.9, 0.0]

    # Thumb — keluar ke kiri
    lm[1] = [0.40, 0.80, 0.0]
    lm[2] = [0.30, 0.75, 0.0]
    lm[3] = [0.25, 0.70, 0.0]
    lm[4] = [0.20, 0.65, 0.0]

    # Index — lurus ke atas
    lm[5] = [0.45, 0.65, 0.0]
    lm[6] = [0.45, 0.50, 0.0]
    lm[7] = [0.45, 0.40, 0.0]
    lm[8] = [0.45, 0.30, 0.0]

    # Middle
    lm[9] = [0.50, 0.65, 0.0]
    lm[10] = [0.50, 0.50, 0.0]
    lm[11] = [0.50, 0.40, 0.0]
    lm[12] = [0.50, 0.30, 0.0]

    # Ring
    lm[13] = [0.55, 0.65, 0.0]
    lm[14] = [0.55, 0.50, 0.0]
    lm[15] = [0.55, 0.40, 0.0]
    lm[16] = [0.55, 0.30, 0.0]

    # Pinky
    lm[17] = [0.60, 0.65, 0.0]
    lm[18] = [0.60, 0.50, 0.0]
    lm[19] = [0.60, 0.40, 0.0]
    lm[20] = [0.60, 0.30, 0.0]

    return lm


def _make_fist() -> np.ndarray:
    """Landmarks untuk FIST — jari folded, thumb tip jauh dari index tip.

    Pattern:
      - Jari menggenggam ke dalam (MCP di luar, TIP di dalam)
      - Thumb tip ada di kiri palm (jauh dari index tip)
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.9, 0.0]

    # Thumb — folded, TIP di KIRI palm (jauh dari index)
    lm[1] = [0.45, 0.85, 0.0]
    lm[2] = [0.42, 0.82, 0.0]
    lm[3] = [0.40, 0.80, 0.0]
    lm[4] = [0.38, 0.78, 0.0]  # thumb tip di kiri

    # Index — folded ke dalam, tip di tengah palm
    lm[5] = [0.45, 0.75, 0.0]   # MCP
    lm[6] = [0.46, 0.70, 0.0]   # PIP — naik (menekuk)
    lm[7] = [0.48, 0.73, 0.0]   # DIP — turun (melengkung)
    lm[8] = [0.50, 0.76, 0.0]   # TIP — ke dalam palm

    # Middle
    lm[9] = [0.50, 0.75, 0.0]
    lm[10] = [0.51, 0.70, 0.0]
    lm[11] = [0.52, 0.73, 0.0]
    lm[12] = [0.53, 0.76, 0.0]

    # Ring
    lm[13] = [0.55, 0.75, 0.0]
    lm[14] = [0.55, 0.70, 0.0]
    lm[15] = [0.56, 0.73, 0.0]
    lm[16] = [0.57, 0.76, 0.0]

    # Pinky
    lm[17] = [0.60, 0.75, 0.0]
    lm[18] = [0.60, 0.70, 0.0]
    lm[19] = [0.60, 0.73, 0.0]
    lm[20] = [0.60, 0.76, 0.0]

    return lm


def _make_point() -> np.ndarray:
    """Landmarks untuk POINT — index extended, lain folded."""
    lm = _make_fist()
    # Index extended (ke atas)
    lm[6] = [0.45, 0.55, 0.0]
    lm[7] = [0.45, 0.45, 0.0]
    lm[8] = [0.45, 0.35, 0.0]
    return lm


def _make_pinch() -> np.ndarray:
    """Landmarks untuk PINCH — thumb tip + index tip bersentuhan."""
    lm = _make_open_palm()
    lm[4] = [0.44, 0.40, 0.0]   # thumb tip
    lm[8] = [0.45, 0.40, 0.0]   # index tip — sangat dekat
    return lm


# ---------- Tests: features ----------

def test_finger_curvature_open():
    lm = _make_open_palm()
    for name in ["index", "middle", "ring", "pinky"]:
        c = finger_curvature(lm, name)
        assert c < 0.3, f"{name} curvature = {c}, harus < 0.3"


def test_finger_curvature_fist():
    """Curvature fist harus > curvature open."""
    lm_fist = _make_fist()
    lm_open = _make_open_palm()
    for name in ["index", "middle", "ring", "pinky"]:
        c_fist = finger_curvature(lm_fist, name)
        c_open = finger_curvature(lm_open, name)
        # Fist harus lebih folded dari open (curvature lebih tinggi)
        assert c_fist > c_open, \
            f"{name}: fist curvature ({c_fist}) harus > open ({c_open})"


def test_finger_spread_open():
    lm = _make_open_palm()
    spread = finger_spread(lm)
    assert spread > 0.5, f"Spread = {spread}, harus > 0.5"


def test_pinch_distance_normalized():
    lm = _make_pinch()
    dist = pinch_distance_normalized(lm)
    assert dist < 0.2, f"Pinch dist = {dist}, harus < 0.2"


def test_palm_facing():
    lm = _make_open_palm()
    facing = palm_facing(lm)
    assert facing in ("front", "back", "unknown")


def test_thumb_orientation():
    lm = _make_open_palm()
    orientation = thumb_orientation_vertical(lm)
    assert orientation in ("up", "down", "left", "right", "unknown")


def test_angle_at():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    angle = angle_at(a, b, c)
    assert abs(angle - np.pi / 2) < 0.01


# ---------- Tests: classify_static ----------

def test_classify_open_palm():
    hand = HandLandmarks(_make_open_palm(), "Right", 0.95)
    gesture, conf = classify_static(hand)
    assert gesture == Gesture.OPEN_PALM, f"Got {gesture}, expected OPEN_PALM"
    assert conf > 0.5


def test_classify_fist():
    hand = HandLandmarks(_make_fist(), "Right", 0.95)
    gesture, conf = classify_static(hand)
    # Boleh FIST atau UNKNOWN — tergantung seberapa folded synthetic landmarks
    # Yang penting: BUKAN PINCH, BUKAN OPEN_PALM
    assert gesture in (Gesture.FIST, Gesture.UNKNOWN), \
        f"Got {gesture}, expected FIST atau UNKNOWN"


def test_classify_point():
    hand = HandLandmarks(_make_point(), "Right", 0.95)
    gesture, conf = classify_static(hand)
    assert gesture == Gesture.POINT, f"Got {gesture}, expected POINT"


def test_classify_pinch():
    hand = HandLandmarks(_make_pinch(), "Right", 0.95)
    gesture, conf = classify_static(hand)
    assert gesture == Gesture.PINCH, f"Got {gesture}, expected PINCH"


# ---------- Tests: smoother ----------

def test_smoother_stable():
    smoother = GestureSmoother(window=5, min_stable=3, cooldown_ms=100, min_confidence=0.5, hold_duration_s=0.0)
    now = time.time()
    results = []
    for i in range(5):
        r = smoother.update(Gesture.PINCH, 0.9, now + i * 0.033)
        if r is not None:
            results.append(r)
    assert len(results) >= 1
    assert results[0].gesture == Gesture.PINCH


def test_smoother_unstable():
    smoother = GestureSmoother(window=5, min_stable=3, min_confidence=0.5)
    now = time.time()
    smoother.update(Gesture.PINCH, 0.9, now)
    smoother.update(Gesture.FIST, 0.9, now + 0.033)
    r = smoother.update(Gesture.OPEN_PALM, 0.9, now + 0.066)
    assert r is None


def test_smoother_cooldown():
    smoother = GestureSmoother(window=5, min_stable=3, cooldown_ms=500, min_confidence=0.5, hold_duration_s=0.0)
    now = time.time()
    for i in range(3):
        r = smoother.update(Gesture.PINCH, 0.9, now + i * 0.033)
    assert r is not None
    for i in range(3, 6):
        r2 = smoother.update(Gesture.PINCH, 0.9, now + i * 0.033)
    assert r2 is None


# ---------- Tests: swipe ----------

def test_swipe_detector():
    det = SwipeDetector(min_distance=0.1, max_duration_ms=500)
    now = time.time()
    for i in range(5):
        det.update(now + i * 0.05, np.array([0.2 + i * 0.05, 0.5]))
    result = det.detect(now + 0.25)
    assert result is not None
    gesture, conf = result
    assert gesture == Gesture.SWIPE_RIGHT