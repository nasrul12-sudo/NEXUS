"""HandTracker abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nexus.vision.hand_tracking.landmarks import HandLandmarks


class HandTracker(ABC):
    """Abstract hand tracker.

    Lifecycle:
        tracker = MediaPipeHandTracker()
        tracker.setup()
        try:
            while True:
                frame = camera.read()
                results = tracker.detect(frame.data)
                for hand in results:
                    process(hand)
        finally:
            tracker.close()
    """

    @abstractmethod
    def setup(self) -> None:
        """Initialize model. Raise HandTrackerError jika gagal."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[HandLandmarks]:
        """Detect hands di image.

        Args:
            image: BGR image, shape (H, W, 3), uint8.

        Returns:
            List of HandLandmarks (0, 1, atau 2 tangan).
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources. Idempotent."""

    @property
    @abstractmethod
    def max_hands(self) -> int:
        """Maximum number of hands yang dideteksi."""