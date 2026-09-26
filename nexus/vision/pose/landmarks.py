"""33-landmark schema untuk MediaPipe Pose.

Reference:
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

Landmark index:
    0: NOSE
    1: LEFT_EYE_INNER      2: LEFT_EYE      3: LEFT_EYE_OUTER
    4: RIGHT_EYE_INNER     5: RIGHT_EYE     6: RIGHT_EYE_OUTER
    7: LEFT_EAR            8: RIGHT_EAR
    9: MOUTH_LEFT         10: MOUTH_RIGHT
    11: LEFT_SHOULDER     12: RIGHT_SHOULDER
    13: LEFT_ELBOW        14: RIGHT_ELBOW
    15: LEFT_WRIST        16: RIGHT_WRIST
    17: LEFT_PINKY        18: RIGHT_PINKY
    19: LEFT_INDEX        20: RIGHT_INDEX
    21: LEFT_THUMB        22: RIGHT_THUMB
    23: LEFT_HIP          24: RIGHT_HIP
    25: LEFT_KNEE         26: RIGHT_KNEE
    27: LEFT_ANKLE        28: RIGHT_ANKLE
    29: LEFT_HEEL         30: RIGHT_HEEL
    31: LEFT_FOOT_INDEX   32: RIGHT_FOOT_INDEX
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class PoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


NUM_POSE_LANDMARKS = 33

# Connections untuk drawing skeleton
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left hand
    (15, 17), (15, 19), (15, 21), (17, 19),
    # Right hand
    (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


@dataclass
class PoseLandmarks:
    """33 pose landmarks + visibility.

    Attributes:
        landmarks: np.ndarray shape (33, 3), normalized [0,1] untuk x,y; z relative.
        visibility: np.ndarray shape (33,), 0..1 — seberapa yakin MediaPipe
                    bahwa landmark terlihat (tidak terhalang).
        world_landmarks: optional np.ndarray (33, 3), metric 3D.
    """

    landmarks: np.ndarray
    visibility: np.ndarray
    world_landmarks: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.landmarks.shape != (NUM_POSE_LANDMARKS, 3):
            raise ValueError(
                f"Expected landmarks shape ({NUM_POSE_LANDMARKS}, 3), "
                f"got {self.landmarks.shape}"
            )
        if self.visibility.shape != (NUM_POSE_LANDMARKS,):
            raise ValueError(
                f"Expected visibility shape ({NUM_POSE_LANDMARKS},), "
                f"got {self.visibility.shape}"
            )

    def get(self, idx: PoseLandmark | int) -> np.ndarray:
        return self.landmarks[int(idx)]

    def is_visible(self, idx: PoseLandmark | int, threshold: float = 0.5) -> bool:
        """True jika landmark cukup visible."""
        return bool(self.visibility[int(idx)] >= threshold)

    def to_dict(self) -> dict:
        return {
            "landmarks": self.landmarks.tolist(),
            "visibility": self.visibility.tolist(),
            "world_landmarks": (
                self.world_landmarks.tolist()
                if self.world_landmarks is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PoseLandmarks:
        return cls(
            landmarks=np.asarray(data["landmarks"], dtype=np.float32),
            visibility=np.asarray(data["visibility"], dtype=np.float32),
            world_landmarks=(
                np.asarray(data["world_landmarks"], dtype=np.float32)
                if data.get("world_landmarks") is not None else None
            ),
        )