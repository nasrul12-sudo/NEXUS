"""Rule-based gesture classification — rotation & scale invariant.

Setiap rule menggunakan features yang sudah dinormalisasi:
  - Curvature (angle-based) — tidak peduli rotasi
  - Distance normalized by hand scale — tidak peduli skala
  - Palm orientation — deteksi via cross product
  - Thumb direction — relative ke palm axis
"""
from __future__ import annotations

import numpy as np

from nexus.vision.gesture.features import (
    FINGER_NAMES,
    all_finger_curvatures,
    finger_spread,
    palm_facing,
    pinch_distance_normalized,
    thumb_orientation_vertical,
    thumb_to_palm_angle,
)
from nexus.vision.gesture.gestures import Gesture
from nexus.vision.hand_tracking.landmarks import HandLandmarks


"""Rule-based gesture classification — tuned dengan data empiris."""

def classify_static(
    hand: HandLandmarks,
    *,
    pinch_threshold: float = 0.35,
    extended_curvature_max: float = 0.25,
    folded_curvature_min: float = 0.40,
    half_folded_curvature_min: float = 0.25,
    min_finger_spread_open: float = 0.75,
) -> tuple[Gesture, float]:
    landmarks = hand.landmarks
    curvatures = all_finger_curvatures(landmarks)

    thumb_c = curvatures["thumb"]
    index_c = curvatures["index"]
    middle_c = curvatures["middle"]
    ring_c = curvatures["ring"]
    pinky_c = curvatures["pinky"]
    non_thumb = np.array([index_c, middle_c, ring_c, pinky_c])

    # ---- PINCH ----
    pinch_dist = pinch_distance_normalized(landmarks)
    if pinch_dist < pinch_threshold:
        conf = 1.0 - (pinch_dist / pinch_threshold) * 0.7
        # Hanya PINCH jika jari lain tidak extended (kalau tidak, mungkin OK sign)
        if not all(c < extended_curvature_max for c in non_thumb):
            return Gesture.PINCH, float(np.clip(conf, 0.4, 1.0))

    # ---- FIST ----
    # Thumb tidak butuh fully folded (memang tidak pernah)
    thumb_ok = thumb_c > extended_curvature_max
    non_thumb_folded = all(c > folded_curvature_min for c in non_thumb)
    if thumb_ok and non_thumb_folded:
        avg_c = float(np.mean(non_thumb))
        conf = 0.6 + 0.4 * min(1.0, (avg_c - folded_curvature_min) / 0.2)
        return Gesture.FIST, float(np.clip(conf, 0.6, 1.0))

    # ---- OPEN_PALM ----
    # Ketat: semua jari < extended_curvature_max DAN spread > min
    all_ext_low = all(c < extended_curvature_max for c in curvatures.values())
    if all_ext_low:
        spread = finger_spread(landmarks)
        if spread > min_finger_spread_open:
            ext_score = 1.0 - float(np.mean(list(curvatures.values()))) / extended_curvature_max
            spread_score = min(1.0, (spread - min_finger_spread_open) / 0.4)
            conf = 0.5 + 0.5 * (0.5 * ext_score + 0.5 * spread_score)
            return Gesture.OPEN_PALM, float(np.clip(conf, 0.5, 1.0))

    # ---- POINT: index extended + 2 dari 3 jari lain half-folded ----
    if index_c < extended_curvature_max:
        others_half = sum([
            middle_c > half_folded_curvature_min,
            ring_c > half_folded_curvature_min,
            pinky_c > half_folded_curvature_min,
        ])
        # Middle & ring minimal ada yang folded (penting!)
        if others_half >= 2 and middle_c > half_folded_curvature_min:
            conf = 0.6 + 0.4 * (1.0 - index_c / extended_curvature_max)
            return Gesture.POINT, float(np.clip(conf, 0.6, 1.0))

    # ---- TWO_FINGER: index + middle extended, min 1 dari ring/pinky half-folded ----
    if index_c < extended_curvature_max and middle_c < extended_curvature_max:
        others_half = sum([
            ring_c > half_folded_curvature_min,
            pinky_c > half_folded_curvature_min,
        ])
        if others_half >= 1:
            conf = 0.6 + 0.4 * (1.0 - max(index_c, middle_c) / extended_curvature_max)
            return Gesture.TWO_FINGER, float(np.clip(conf, 0.6, 1.0))

    # ---- THREE_FINGER ----
    if (
        index_c < extended_curvature_max
        and middle_c < extended_curvature_max
        and ring_c < extended_curvature_max
        and pinky_c > half_folded_curvature_min
    ):
        conf = 0.6 + 0.4 * (1.0 - max(index_c, middle_c, ring_c) / extended_curvature_max)
        return Gesture.THREE_FINGER, float(np.clip(conf, 0.6, 1.0))

    # ---- THUMBS_UP / THUMBS_DOWN ----
    if (
        thumb_c < half_folded_curvature_min
        and all(c > folded_curvature_min for c in non_thumb)
    ):
        orientation = thumb_orientation_vertical(landmarks)
        if orientation == "up":
            return Gesture.THUMBS_UP, 0.80
        elif orientation == "down":
            return Gesture.THUMBS_DOWN, 0.80
        elif orientation == "left":
            return Gesture.THUMBS_LEFT, 0.70
        elif orientation == "right":
            return Gesture.THUMBS_RIGHT, 0.70

    # ---- ROCK ----
    if (
        index_c < extended_curvature_max
        and pinky_c < extended_curvature_max
        and middle_c > folded_curvature_min
        and ring_c > folded_curvature_min
    ):
        conf = 0.6 + 0.4 * (1.0 - max(index_c, pinky_c) / extended_curvature_max)
        return Gesture.ROCK, float(np.clip(conf, 0.6, 1.0))

    return Gesture.UNKNOWN, 0.0


# ---------- Swipe detection (velocity-based) ----------

class SwipeDetector:
    """Detect swipe dengan velocity + direction.

    Menggunakan history posisi + timestamp untuk hitung velocity.
    Robust ke gerakan lambat vs cepat.
    """

    def __init__(
        self,
        *,
        min_distance: float = 0.15,
        max_duration_ms: int = 600,
        history_size: int = 15,
        min_velocity: float = 0.3,  # unit normalized / detik
    ):
        self._min_distance = min_distance
        self._max_duration_ms = max_duration_ms
        self._history_size = history_size
        self._min_velocity = min_velocity
        self._history: list[tuple[float, np.ndarray]] = []

    def update(self, timestamp: float, center: np.ndarray) -> None:
        self._history.append((timestamp, center))
        if len(self._history) > self._history_size:
            self._history.pop(0)

    def clear(self) -> None:
        self._history.clear()

    def detect(self, current_ts: float) -> tuple[Gesture, float] | None:
        if len(self._history) < 3:
            return None

        cutoff = current_ts - (self._max_duration_ms / 1000.0)
        recent = [(ts, c) for ts, c in self._history if ts >= cutoff]
        if len(recent) < 3:
            return None

        start_ts, start_c = recent[0]
        end_ts, end_c = recent[-1]
        dt = end_ts - start_ts
        if dt <= 0:
            return None

        delta = end_c - start_c
        distance_moved = float(np.linalg.norm(delta))

        if distance_moved < self._min_distance:
            return None

        velocity = distance_moved / dt
        if velocity < self._min_velocity:
            return None

        dx, dy = delta[0], delta[1]

        # Direction: dominant axis
        if abs(dx) > abs(dy):
            gesture = Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT
            dominance = abs(dx) / (abs(dx) + abs(dy) + 1e-6)
        else:
            gesture = Gesture.SWIPE_DOWN if dy > 0 else Gesture.SWIPE_UP
            dominance = abs(dy) / (abs(dx) + abs(dy) + 1e-6)

        # Confidence: distance + dominance + velocity
        dist_conf = min(distance_moved / self._min_distance, 1.5) / 1.5
        vel_conf = min(velocity / self._min_velocity, 2.0) / 2.0
        confidence = float(0.4 * dist_conf + 0.3 * dominance + 0.3 * vel_conf)
        return gesture, confidence

# ---------- Body-aware gesture classification ----------

def classify_thumbs_with_body(
    hand: HandLandmarks,
    body_frame,  # BodyFrame | None
    *,
    thumb_curvature_max: float = 0.3,
    others_folded_min: float = 0.4,
) -> tuple[Gesture, float]:
    """Klasifikasi THUMBS_UP/DOWN/LEFT/RIGHT dengan body frame.

    Tanpa body frame, kita fallback ke palm axis.
    Dengan body frame, kita pakai arah thumb relatif ke tubuh.

    Args:
        hand: HandLandmarks
        body_frame: BodyFrame atau None
        thumb_curvature_max: thumb harus < ini untuk dianggap extended
        others_folded_min: jari lain harus > ini untuk dianggap folded

    Returns:
        (gesture, confidence)
    """
    from nexus.vision.gesture.features import (
        all_finger_curvatures,
        thumb_direction_in_body_frame,
    )

    landmarks = hand.landmarks
    curvatures = all_finger_curvatures(landmarks)

    thumb_c = curvatures["thumb"]
    non_thumb = [
        curvatures["index"],
        curvatures["middle"],
        curvatures["ring"],
        curvatures["pinky"],
    ]

    # Thumb harus extended
    if thumb_c > thumb_curvature_max:
        return Gesture.UNKNOWN, 0.0

    # Jari lain harus folded
    if not all(c > others_folded_min for c in non_thumb):
        return Gesture.UNKNOWN, 0.0

    # Tentukan arah thumb
    if body_frame is not None:
        direction = thumb_direction_in_body_frame(landmarks, body_frame)
        conf = 0.85
    else:
        # Fallback: palm axis
        from nexus.vision.gesture.features import thumb_orientation_vertical
        direction = thumb_orientation_vertical(landmarks)
        conf = 0.7

    if direction == "up":
        return Gesture.THUMBS_UP, conf
    elif direction == "down":
        return Gesture.THUMBS_DOWN, conf
    elif direction == "left":
        return Gesture.THUMBS_LEFT, conf * 0.9
    elif direction == "right":
        return Gesture.THUMBS_RIGHT, conf * 0.9

    return Gesture.UNKNOWN, 0.0


def classify_swipe_with_body(
    velocity_body: np.ndarray,
    *,
    min_velocity: float = 0.3,
    min_duration_ms: int = 50,
    max_duration_ms: int = 600,
) -> tuple[Gesture, float]:
    """Klasifikasi swipe dengan velocity dalam body frame.

    Args:
        velocity_body: velocity (forward, right, up) dalam body frame
        min_velocity: minimum magnitude velocity
        min_duration_ms, max_duration_ms: durasi swipe

    Returns:
        (gesture, confidence)
    """
    from nexus.vision.gesture.features import classify_swipe_in_body_frame

    direction, confidence = classify_swipe_in_body_frame(
        velocity_body, min_velocity=min_velocity
    )

    if direction == "up":
        return Gesture.SWIPE_UP, confidence
    elif direction == "down":
        return Gesture.SWIPE_DOWN, confidence
    elif direction == "right":
        return Gesture.SWIPE_RIGHT, confidence
    elif direction == "left":
        return Gesture.SWIPE_LEFT, confidence

    return Gesture.UNKNOWN, 0.0

# ---------- Swipe detection (velocity-based) ----------

class SwipeDetector:
    """Detect swipe dengan velocity + direction.

    Menggunakan history posisi + timestamp untuk hitung velocity.
    Robust ke gerakan lambat vs cepat.
    """

    def __init__(
        self,
        *,
        min_distance: float = 0.15,
        max_duration_ms: int = 600,
        history_size: int = 15,
        min_velocity: float = 0.3,
    ):
        self._min_distance = min_distance
        self._max_duration_ms = max_duration_ms
        self._history_size = history_size
        self._min_velocity = min_velocity
        # (timestamp, center_xy)
        self._history: list[tuple[float, np.ndarray]] = []

    def update(self, timestamp: float, center: np.ndarray) -> None:
        """Tambah posisi hand center ke history."""
        self._history.append((timestamp, np.asarray(center, dtype=np.float32)))
        if len(self._history) > self._history_size:
            self._history.pop(0)

    def clear(self) -> None:
        """Reset history."""
        self._history.clear()

    def detect(self, current_ts: float) -> tuple[Gesture, float] | None:
        """Detect swipe dari history. Returns (Gesture, confidence) atau None."""
        if len(self._history) < 3:
            return None

        # Filter history dalam max_duration
        cutoff = current_ts - (self._max_duration_ms / 1000.0)
        recent = [(ts, c) for ts, c in self._history if ts >= cutoff]
        if len(recent) < 3:
            return None

        start_ts, start_c = recent[0]
        end_ts, end_c = recent[-1]
        dt = end_ts - start_ts
        if dt <= 0:
            return None

        delta = end_c - start_c
        distance_moved = float(np.linalg.norm(delta))

        if distance_moved < self._min_distance:
            return None

        velocity = distance_moved / dt
        if velocity < self._min_velocity:
            return None

        dx, dy = float(delta[0]), float(delta[1])

        # Dominant axis
        if abs(dx) > abs(dy):
            gesture = Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT
            dominance = abs(dx) / (abs(dx) + abs(dy) + 1e-6)
        else:
            gesture = Gesture.SWIPE_DOWN if dy > 0 else Gesture.SWIPE_UP
            dominance = abs(dy) / (abs(dx) + abs(dy) + 1e-6)

        # Confidence
        dist_conf = min(distance_moved / self._min_distance, 1.5) / 1.5
        vel_conf = min(velocity / self._min_velocity, 2.0) / 2.0
        confidence = float(0.4 * dist_conf + 0.3 * dominance + 0.3 * vel_conf)
        return gesture, confidence