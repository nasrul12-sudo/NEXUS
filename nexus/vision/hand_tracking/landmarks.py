"""21-landmark schema + utilities.

MediaPipe Hand Landmarker mendeteksi 21 titik per tangan:

    0: WRIST
    1: THUMB_CMC,      2: THUMB_MCP,      3: THUMB_IP,       4: THUMB_TIP
    5: INDEX_MCP,      6: INDEX_PIP,      7: INDEX_DIP,      8: INDEX_TIP
    9: MIDDLE_MCP,    10: MIDDLE_PIP,    11: MIDDLE_DIP,    12: MIDDLE_TIP
    13: RING_MCP,     14: RING_PIP,      15: RING_DIP,      16: RING_TIP
    17: PINKY_MCP,    18: PINKY_PIP,     19: PINKY_DIP,     20: PINKY_TIP

Koordinat:
  - x, y: normalized [0, 1] relative to image width/height
  - z: relative depth (smaller = closer to camera)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np


class Landmark(IntEnum):
    """Indices untuk 21 landmarks MediaPipe."""

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Convenience groupings
FINGER_TIPS = [
    Landmark.THUMB_TIP,
    Landmark.INDEX_TIP,
    Landmark.MIDDLE_TIP,
    Landmark.RING_TIP,
    Landmark.PINKY_TIP,
]

FINGER_PIPS = [
    Landmark.THUMB_IP,
    Landmark.INDEX_PIP,
    Landmark.MIDDLE_PIP,
    Landmark.RING_PIP,
    Landmark.PINKY_PIP,
]

FINGER_MCPS = [
    Landmark.THUMB_MCP,
    Landmark.INDEX_MCP,
    Landmark.MIDDLE_MCP,
    Landmark.RING_MCP,
    Landmark.PINKY_MCP,
]

NUM_LANDMARKS = 21


@dataclass
class HandLandmarks:
    """21 landmarks + handedness + confidence.

    Attributes:
        landmarks: np.ndarray shape (21, 3), float32.
                   Koordinat normalized [0,1] untuk x,y; z relative depth.
        handedness: "Left" atau "Right".
        confidence: detection confidence [0, 1].
        world_landmarks: optional np.ndarray shape (21, 3), metric 3D
                         (jika MediaPipe menghasilkan world landmarks).
    """

    landmarks: np.ndarray
    handedness: str
    confidence: float
    world_landmarks: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.landmarks.shape != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"Expected landmarks shape ({NUM_LANDMARKS}, 3), "
                f"got {self.landmarks.shape}"
            )
        if self.handedness not in ("Left", "Right"):
            raise ValueError(f"Invalid handedness: {self.handedness}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if self.world_landmarks is not None:
            if self.world_landmarks.shape != (NUM_LANDMARKS, 3):
                raise ValueError(
                    f"Expected world_landmarks shape ({NUM_LANDMARKS}, 3), "
                    f"got {self.world_landmarks.shape}"
                )

    def get(self, idx: Landmark | int) -> np.ndarray:
        """Get landmark coordinates (x, y, z)."""
        return self.landmarks[int(idx)]

    def x(self, idx: Landmark | int) -> float:
        return float(self.landmarks[int(idx), 0])

    def y(self, idx: Landmark | int) -> float:
        return float(self.landmarks[int(idx), 1])

    def z(self, idx: Landmark | int) -> float:
        return float(self.landmarks[int(idx), 2])

    def to_dict(self) -> dict:
        """Serialize untuk event payload."""
        return {
            "landmarks": self.landmarks.tolist(),
            "handedness": self.handedness,
            "confidence": self.confidence,
            "world_landmarks": (
                self.world_landmarks.tolist()
                if self.world_landmarks is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> HandLandmarks:
        return cls(
            landmarks=np.asarray(data["landmarks"], dtype=np.float32),
            handedness=data["handedness"],
            confidence=float(data["confidence"]),
            world_landmarks=(
                np.asarray(data["world_landmarks"], dtype=np.float32)
                if data.get("world_landmarks") is not None
                else None
            ),
        )


# ---------- Geometric helpers (dipakai gesture engine nanti) ----------

def distance(
    a: np.ndarray, b: np.ndarray, use_z: bool = False
) -> float:
    """Euclidean distance antara 2 landmark.

    Args:
        a, b: shape (3,) landmark coordinates
        use_z: jika True, include z. Default False (2D only).
    """
    if use_z:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a[:2] - b[:2]))


def hand_center(landmarks: np.ndarray) -> np.ndarray:
    """Compute center of palm (mean of MCP joints)."""
    mcp_idx = [int(m) for m in FINGER_MCPS]
    return landmarks[mcp_idx].mean(axis=0)


def hand_scale(landmarks: np.ndarray) -> float:
    """Compute hand scale (distance wrist → middle MCP).

    Digunakan untuk normalisasi gesture (scale-invariant).
    """
    return distance(
        landmarks[int(Landmark.WRIST)],
        landmarks[int(Landmark.MIDDLE_MCP)],
    )


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks: translate ke hand center, scale ke unit.

    Returns:
        np.ndarray shape (21, 3), normalized.
    """
    center = hand_center(landmarks)
    scale = hand_scale(landmarks)
    if scale < 1e-6:
        return landmarks.copy()
    normalized = landmarks.copy()
    normalized[:, :2] = (normalized[:, :2] - center[:2]) / scale
    return normalized