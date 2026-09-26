"""PoseTracker abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nexus.vision.pose.landmarks import PoseLandmarks


class PoseTracker(ABC):
    """Abstract pose tracker."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize model."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> PoseLandmarks | None:
        """Detect pose in image. Returns None if no pose detected."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""