"""OpenCV-based camera provider."""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2

from nexus.core.logging import get_logger
from nexus.vision.camera.base import CameraError, CameraProvider, Frame


log = get_logger(__name__)


class OpenCVCameraProvider(CameraProvider):
    """OpenCV VideoCapture wrapper."""

    def __init__(
        self,
        device: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: int | None = None,
        buffer_size: int = 1,
    ):
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._buffer_size = buffer_size

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.RLock()
        self._frame_index = 0
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0

    def open(self) -> None:
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return

            kwargs: dict[str, Any] = {}
            if self._backend is not None:
                kwargs["apiPreference"] = self._backend

            cap = cv2.VideoCapture(self._device, **kwargs)
            if not cap.isOpened():
                raise CameraError(
                    f"Failed to open camera device={self._device}",
                    device=self._device,
                )

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

            self._cap = cap
            self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._frame_index = 0

            log.info(
                "camera.opened",
                device=self._device,
                width=self._actual_width,
                height=self._actual_height,
                fps=self._actual_fps,
            )

    def read(self) -> Frame | None:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ok, data = self._cap.read()
            if not ok or data is None:
                return None
            self._frame_index += 1
            return Frame(
                data=data,
                timestamp=time.time(),
                frame_index=self._frame_index,
            )

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                log.info("camera.closed", device=self._device)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def actual_resolution(self) -> tuple[int, int]:
        return (self._actual_width, self._actual_height)