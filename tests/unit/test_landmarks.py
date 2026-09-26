import numpy as np
import pytest

from nexus.vision.hand_tracking.landmarks import (
    FINGER_MCPS,
    FINGER_TIPS,
    NUM_LANDMARKS,
    HandLandmarks,
    Landmark,
    distance,
    hand_center,
    hand_scale,
    normalize_landmarks,
)


def _make_landmarks() -> np.ndarray:
    """Create valid dummy landmarks (21, 3)."""
    rng = np.random.default_rng(42)
    return rng.random((NUM_LANDMARKS, 3), dtype=np.float32)


def test_landmark_enum():
    assert int(Landmark.WRIST) == 0
    assert int(Landmark.INDEX_TIP) == 8
    assert int(Landmark.PINKY_TIP) == 20
    assert len(FINGER_TIPS) == 5
    assert len(FINGER_MCPS) == 5


def test_hand_landmarks_valid():
    lm = _make_landmarks()
    hand = HandLandmarks(landmarks=lm, handedness="Right", confidence=0.95)
    assert hand.landmarks.shape == (21, 3)
    assert hand.handedness == "Right"
    assert hand.confidence == 0.95


def test_hand_landmarks_invalid_shape():
    with pytest.raises(ValueError, match="Expected landmarks shape"):
        HandLandmarks(
            landmarks=np.zeros((20, 3), dtype=np.float32),
            handedness="Right",
            confidence=0.9,
        )


def test_hand_landmarks_invalid_handedness():
    with pytest.raises(ValueError, match="Invalid handedness"):
        HandLandmarks(
            landmarks=_make_landmarks(),
            handedness="Unknown",
            confidence=0.9,
        )


def test_hand_landmarks_invalid_confidence():
    with pytest.raises(ValueError, match="Invalid confidence"):
        HandLandmarks(
            landmarks=_make_landmarks(),
            handedness="Right",
            confidence=1.5,
        )


def test_hand_landmarks_accessors():
    lm = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    lm[Landmark.WRIST] = [0.1, 0.2, 0.3]
    lm[Landmark.INDEX_TIP] = [0.5, 0.6, 0.7]
    hand = HandLandmarks(landmarks=lm, handedness="Left", confidence=1.0)

    assert hand.x(Landmark.WRIST) == pytest.approx(0.1)
    assert hand.y(Landmark.WRIST) == pytest.approx(0.2)
    assert hand.z(Landmark.WRIST) == pytest.approx(0.3)
    np.testing.assert_allclose(
        hand.get(Landmark.INDEX_TIP), [0.5, 0.6, 0.7]
    )


def test_hand_landmarks_roundtrip():
    lm = _make_landmarks()
    hand = HandLandmarks(landmarks=lm, handedness="Right", confidence=0.9)
    d = hand.to_dict()
    restored = HandLandmarks.from_dict(d)
    np.testing.assert_allclose(restored.landmarks, lm)
    assert restored.handedness == "Right"
    assert restored.confidence == pytest.approx(0.9)


def test_distance_2d():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 100.0])  # z huge, tapi di-ignore
    assert distance(a, b) == pytest.approx(5.0)
    assert distance(a, b, use_z=True) > 100.0


def test_hand_center():
    lm = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    for i, mcp in enumerate(FINGER_MCPS):
        lm[mcp] = [i * 0.1, i * 0.2, 0.0]
    center = hand_center(lm)
    assert center.shape == (3,)
    # Mean of (0,0.1,0.2,0.3,0.4) = 0.2
    assert center[0] == pytest.approx(0.2)


def test_hand_scale():
    lm = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    lm[Landmark.WRIST] = [0.0, 0.0, 0.0]
    lm[Landmark.MIDDLE_MCP] = [0.3, 0.4, 0.0]  # distance = 0.5
    assert hand_scale(lm) == pytest.approx(0.5)


def test_normalize_landmarks():
    lm = _make_landmarks()
    normalized = normalize_landmarks(lm)
    assert normalized.shape == (21, 3)
    # Setelah normalisasi, center harus dekat 0
    center = hand_center(normalized)
    assert abs(center[0]) < 1.0
    assert abs(center[1]) < 1.0