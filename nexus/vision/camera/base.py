"""CameraProvider abstraction.

Semua camera implementation harus mengikuti interface ini.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from nexus.core.errors import NexusError


class CameraError(NexusError):
    """Error saat camera operation gagal.

    Didefinisikan di base.py (bukan di provider) karena ini bagian
    dari kontrak abstraction. Setiap provider bisa raise error ini
    tanpa import dari provider lain.
    """
    code = "CAMERA_ERROR"


@dataclass
class Frame:
    """Satu frame dari camera."""

    data: np.ndarray          # HxWx3, BGR (OpenCV convention)
    timestamp: float          # unix timestamp saat frame di-capture
    frame_index: int          # increment per frame

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]


class CameraProvider(ABC):
    """Abstract camera provider.

    Lifecycle:
        cam = OpenCVCameraProvider(device=0)
        cam.open()
        try:
            while True:
                frame = cam.read()
                if frame is None:
                    continue
                process(frame)
        finally:
            cam.close()
    """

    @abstractmethod
    def open(self) -> None:
        """Open camera. Raise CameraError jika gagal."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Read next frame.

        Returns:
            Frame atau None jika tidak ada frame (timeout/EOF).
        """

    @abstractmethod
    def close(self) -> None:
        """Close camera. Idempotent."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """True jika camera siap dibaca."""

    @property
    @abstractmethod
    def fps(self) -> int:
        """Configured FPS target."""

    def __enter__(self) -> CameraProvider:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()