"""Temporal smoothing untuk gesture recognition.

Konsep:
  - Setiap frame, engine menghasilkan (gesture, confidence)
  - Smoother menyimpan history N frame terakhir
  - Gesture dianggap "stable" jika ≥ min_stable_frames frame
    terakhir memiliki gesture yang sama
  - UNKNOWN diabaikan — tidak mengotori history
  - Setelah stable, gesture di-publish dengan confidence rata-rata
  - Cooldown mencegah gesture yang sama trigger berulang
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from nexus.vision.gesture.gestures import Gesture


@dataclass
class SmoothedGesture:
    gesture: Gesture
    confidence: float
    duration_ms: float
    frame_count: int


class GestureSmoother:
    """Temporal smoothing dengan sliding window."""

    def __init__(
        self,
        *,
        window: int = 4,
        min_stable: int = 2,
        cooldown_ms: int = 300,
        min_confidence: float = 0.5,
    ):
        if min_stable > window:
            raise ValueError("min_stable tidak boleh lebih dari window")

        self._window = window
        self._min_stable = min_stable
        self._cooldown_s = cooldown_ms / 1000.0
        self._min_confidence = min_confidence

        self._history: deque[tuple[Gesture, float, float]] = deque(maxlen=window)
        self._last_published: Gesture = Gesture.UNKNOWN
        self._last_publish_ts: float = 0.0
        self._current_streak_start: float | None = None
        self._consecutive_unknown: int = 0  # hitung UNKNOWN berturut-turut

    def update(
        self,
        gesture: Gesture,
        confidence: float,
        timestamp: float | None = None,
    ) -> SmoothedGesture | None:
        """Update smoother dengan gesture frame ini.

        UNKNOWN diabaikan — tidak dimasukkan ke history.

        Returns:
            SmoothedGesture jika gesture stable & confirmed, None otherwise.
        """
        if timestamp is None:
            timestamp = time.time()

        # ---- Skip UNKNOWN ----
        if gesture == Gesture.UNKNOWN or confidence < self._min_confidence:
            self._consecutive_unknown += 1
            # Kalau terlalu banyak UNKNOWN berturut-turut, reset history
            # (tangan hilang / pose berubah signifikan)
            if self._consecutive_unknown >= self._window:
                self.reset()
                self._current_streak_start = None
            return None

        # Reset UNKNOWN counter saat ada gesture valid
        self._consecutive_unknown = 0

        # Tambahkan ke history
        self._history.append((gesture, confidence, timestamp))
        return self._try_confirm(timestamp)

    def _try_confirm(self, timestamp: float) -> SmoothedGesture | None:
        if len(self._history) < self._min_stable:
            return None

        # Ambil min_stable frame terakhir
        recent = list(self._history)[-self._min_stable:]

        # Cek konsistensi — semua frame harus gesture yang sama
        gestures = [g for g, _, _ in recent]
        if len(set(gestures)) != 1:
            self._current_streak_start = None
            return None

        confirmed_gesture = gestures[0]

        # Cek confidence rata-rata
        avg_confidence = sum(c for _, c, _ in recent) / len(recent)
        if avg_confidence < self._min_confidence:
            self._current_streak_start = None
            return None

        # Track streak duration
        if self._current_streak_start is None:
            self._current_streak_start = recent[0][2]
        duration_ms = (timestamp - self._current_streak_start) * 1000

        # Cooldown check
        if confirmed_gesture == self._last_published:
            time_since_publish = timestamp - self._last_publish_ts
            if time_since_publish < self._cooldown_s:
                return None

        # Confirmed!
        self._last_published = confirmed_gesture
        self._last_publish_ts = timestamp

        return SmoothedGesture(
            gesture=confirmed_gesture,
            confidence=avg_confidence,
            duration_ms=duration_ms,
            frame_count=len(recent),
        )

    def reset(self) -> None:
        """Reset smoother state. Dipanggil saat tangan hilang."""
        self._history.clear()
        self._current_streak_start = None
        self._consecutive_unknown = 0

    def reset_last_published(self) -> None:
        """Reset hanya last published."""
        self._last_published = Gesture.UNKNOWN
        self._last_publish_ts = 0.0