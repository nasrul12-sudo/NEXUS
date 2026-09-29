"""Mode Manager — state machine untuk NEXUS activation.

State:
    IDLE          — hanya wake gesture yang diproses
    LISTENING     — gesture aktif diproses, timeout 30s
    PROCESSING    — sedang eksekusi command (singkat)
    CONFIRMING    — menunggu konfirmasi user (thumbs up/down)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from nexus.core.logging import get_logger


log = get_logger(__name__)


class NexusMode(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    CONFIRMING = "CONFIRMING"


@dataclass
class ModeConfig:
    """Konfigurasi mode manager."""
    listening_timeout_s: float = 30.0
    processing_timeout_s: float = 5.0
    confirming_timeout_s: float = 15.0
    wake_gesture_hold_s: float = 1.0
    action_gesture_hold_s: float = 0.5


class ModeManager:
    """State machine untuk activation mode."""

    def __init__(self, config: ModeConfig | None = None):
        self._config = config or ModeConfig()
        self._mode: NexusMode = NexusMode.IDLE
        self._mode_entered_ts: float = time.time()
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[NexusMode, NexusMode], None]] = []

        # Wake gesture state
        self._wake_gesture_frames: int = 0
        self._wake_gesture_needed: int = 0

        # Action gesture state
        self._action_gesture_state: dict = {}

        # Statistics
        self._stats: dict = {
            "activations": 0,
            "deactivations": 0,
            "gesture_rejections_idle": 0,
        }

    # ---------- Lifecycle ----------

    def start(self, fps: float = 15.0) -> None:
        self._wake_gesture_needed = max(3, int(self._config.wake_gesture_hold_s * fps))
        log.info(
            "mode_manager.start",
            listening_timeout_s=self._config.listening_timeout_s,
            wake_gesture_frames_needed=self._wake_gesture_needed,
        )

    def stop(self) -> None:
        log.info("mode_manager.stop", stats=self._stats)

    # ---------- Mode access ----------

    @property
    def mode(self) -> NexusMode:
        with self._lock:
            return self._mode

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._mode != NexusMode.IDLE

    def on_mode_change(self, callback: Callable[[NexusMode, NexusMode], None]) -> None:
        self._callbacks.append(callback)

    # ---------- Mode transitions ----------

    def _set_mode(self, new_mode: NexusMode) -> None:
        with self._lock:
            old_mode = self._mode
            if old_mode == new_mode:
                return
            self._mode = new_mode
            self._mode_entered_ts = time.time()

            if new_mode == NexusMode.LISTENING:
                self._stats["activations"] += 1
            elif new_mode == NexusMode.IDLE and old_mode != NexusMode.IDLE:
                self._stats["deactivations"] += 1

            log.info(
                "mode_manager.transition",
                from_mode=old_mode.value,
                to_mode=new_mode.value,
            )

        for cb in self._callbacks:
            try:
                cb(old_mode, new_mode)
            except Exception:
                log.exception("mode_manager.callback_error")

    def activate(self, source: str = "unknown") -> None:
        with self._lock:
            if self._mode == NexusMode.IDLE:
                log.info("mode_manager.activate", source=source)
                self._set_mode(NexusMode.LISTENING)

    def deactivate(self, source: str = "unknown") -> None:
        log.info("mode_manager.deactivate", source=source)
        self._set_mode(NexusMode.IDLE)

    def confirm_mode(self) -> None:
        self._set_mode(NexusMode.CONFIRMING)

    def processing_mode(self) -> None:
        self._set_mode(NexusMode.PROCESSING)

    # ---------- Gesture processing ----------

    def should_process_gesture(
        self,
        gesture_name: str,
        hand_count: int = 1,
        timestamp: float | None = None,
    ) -> bool:
        """Cek apakah gesture boleh diproses di mode saat ini."""
        if timestamp is None:
            timestamp = time.time()

        self._check_timeout(timestamp)
        mode = self.mode

        # IDLE: hanya wake gesture
        if mode == NexusMode.IDLE:
            if self._is_wake_gesture(gesture_name, hand_count):
                self._wake_gesture_frames += 1
                if self._wake_gesture_frames >= self._wake_gesture_needed:
                    log.info(
                        "mode_manager.wake_gesture_confirmed",
                        gesture=gesture_name,
                        hold_frames=self._wake_gesture_frames,
                    )
                    self.activate(source="wake_gesture")
                    self._wake_gesture_frames = 0
                    return False
            else:
                self._wake_gesture_frames = 0

            self._stats["gesture_rejections_idle"] += 1
            return False

        # LISTENING: semua gesture (kecuali wake)
        if mode == NexusMode.LISTENING:
            if self._is_wake_gesture(gesture_name, hand_count):
                self._wake_gesture_frames += 1
                if self._wake_gesture_frames >= self._wake_gesture_needed:
                    log.info("mode_manager.exit_by_wake_gesture")
                    self.deactivate(source="wake_gesture")
                    self._wake_gesture_frames = 0
                return False
            else:
                self._wake_gesture_frames = 0

            self._mode_entered_ts = timestamp
            return True

        # PROCESSING: ignore
        if mode == NexusMode.PROCESSING:
            return False

        # CONFIRMING: hanya thumbs up/down
        if mode == NexusMode.CONFIRMING:
            return gesture_name in ("THUMBS_UP", "THUMBS_DOWN")

        return False

    def _is_wake_gesture(self, gesture_name: str, hand_count: int) -> bool:
        return gesture_name == "OPEN_PALM" and hand_count >= 2

    def _check_timeout(self, timestamp: float) -> None:
        elapsed = timestamp - self._mode_entered_ts
        mode = self.mode

        if mode == NexusMode.LISTENING:
            if elapsed > self._config.listening_timeout_s:
                log.info(
                    "mode_manager.timeout",
                    mode="LISTENING",
                    elapsed_s=round(elapsed, 1),
                )
                self.deactivate(source="timeout")

        elif mode == NexusMode.PROCESSING:
            if elapsed > self._config.processing_timeout_s:
                log.warning("mode_manager.timeout", mode="PROCESSING")
                self._set_mode(NexusMode.LISTENING)

        elif mode == NexusMode.CONFIRMING:
            if elapsed > self._config.confirming_timeout_s:
                log.info("mode_manager.timeout", mode="CONFIRMING")
                self._set_mode(NexusMode.LISTENING)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "mode": self._mode.value,
                "elapsed_s": time.time() - self._mode_entered_ts,
                "wake_gesture_frames": self._wake_gesture_frames,
                "wake_gesture_needed": self._wake_gesture_needed,
                "stats": dict(self._stats),
            }


# ---------- Singleton ----------

_manager: ModeManager | None = None


def get_mode_manager() -> ModeManager:
    global _manager
    if _manager is None:
        _manager = ModeManager()
    return _manager


def reset_mode_manager() -> None:
    global _manager
    _manager = None
