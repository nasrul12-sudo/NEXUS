"""Feature extraction dari hand landmarks — rotation & scale invariant.

Prinsip:
  1. Semua fitur dinormalisasi terhadap hand scale (jarak wrist→middle_mcp)
  2. Semua sudut dihitung dari vector antar landmark (bukan posisi absolut)
  3. Finger extension diukur dari ANGLE antar joint, bukan distance
  4. Thumb detection menggunakan sudut relative ke palm plane
  5. Palm orientation dideteksi dari cross product (2D → sign)
"""
from __future__ import annotations

import numpy as np

from nexus.vision.hand_tracking.landmarks import (
    FINGER_MCPS,
    FINGER_PIPS,
    FINGER_TIPS,
    Landmark,
    distance,
    hand_center,
    hand_scale,
)


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Joint chain per jari (MCP → PIP → DIP → TIP)
FINGER_CHAINS: dict[str, list[int]] = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


# ---------- Basic geometry ----------

def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Sudut (radian) di titik b, antara vector b→a dan b→c.

    Rotation-invariant: tidak peduli orientasi global.
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1[:2])
    n2 = np.linalg.norm(v2[:2])
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.dot(v1[:2], v2[:2]) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Sudut (radian) antara dua vector 2D."""
    n1 = np.linalg.norm(a[:2])
    n2 = np.linalg.norm(b[:2])
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.dot(a[:2], b[:2]) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


# ---------- Finger extension (angle-based, rotation-invariant) ----------

def finger_curvature(landmarks: np.ndarray, finger: str) -> float:
    """Ukur curvature jari (0 = lurus, 1 = tertekuk penuh).

    Rotation-invariant: pakai sudut antar joint, bukan posisi absolut.
    """
    chain = FINGER_CHAINS[finger]
    if len(chain) < 3:
        return 0.0

    angles = []
    for i in range(1, len(chain) - 1):
        a = landmarks[chain[i - 1]]
        b = landmarks[chain[i]]
        c = landmarks[chain[i + 1]]
        angles.append(angle_at(a, b, c))

    if not angles:
        return 0.0

    avg_angle = float(np.mean(angles))
    curvature = 1.0 - (avg_angle / np.pi)
    return float(np.clip(curvature, 0.0, 1.0))


def all_finger_curvatures(landmarks: np.ndarray) -> dict[str, float]:
    """Curvature semua jari."""
    return {
        name: finger_curvature(landmarks, name)
        for name in FINGER_NAMES
    }


def finger_extensions(landmarks: np.ndarray) -> np.ndarray:
    """Extension ratio per jari (0..1, 1 = extended).

    Kombinasi dari:
      - Curvature (angle-based) — rotation invariant
      - Distance wrist→tip normalized — scale invariant
    """
    curvatures = all_finger_curvatures(landmarks)
    ext = np.zeros(5, dtype=np.float32)
    for i, name in enumerate(FINGER_NAMES):
        ext[i] = 1.0 - curvatures[name]
    return ext


# ---------- Thumb-specific detection ----------

def thumb_direction(landmarks: np.ndarray) -> np.ndarray:
    """Vector arah thumb (dari MCP ke TIP) normalized."""
    mcp = landmarks[int(Landmark.THUMB_MCP)]
    tip = landmarks[int(Landmark.THUMB_TIP)]
    v = tip - mcp
    n = np.linalg.norm(v[:2])
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([v[0] / n, v[1] / n, v[2] / n], dtype=np.float32)


def thumb_to_palm_angle(landmarks: np.ndarray) -> float:
    """Sudut antara thumb dan palm axis."""
    wrist = landmarks[int(Landmark.WRIST)]
    middle_mcp = landmarks[int(Landmark.MIDDLE_MCP)]
    thumb_mcp = landmarks[int(Landmark.THUMB_MCP)]
    thumb_tip = landmarks[int(Landmark.THUMB_TIP)]

    palm_axis = middle_mcp - wrist
    thumb_vec = thumb_tip - thumb_mcp
    return angle_between(palm_axis, thumb_vec)


def is_thumb_extended(landmarks: np.ndarray) -> bool:
    """Thumb extended jika curvature rendah DAN sudut ke palm besar."""
    curvature = finger_curvature(landmarks, "thumb")
    palm_angle = thumb_to_palm_angle(landmarks)
    return curvature < 0.5 and palm_angle > np.radians(30)


def thumb_orientation_vertical(landmarks: np.ndarray) -> str:
    """Arah thumb: "up", "down", "left", "right", atau "unknown".

    Rotation-aware: pakai palm axis sebagai referensi.
    """
    wrist = landmarks[int(Landmark.WRIST)]
    middle_mcp = landmarks[int(Landmark.MIDDLE_MCP)]
    thumb_tip = landmarks[int(Landmark.THUMB_TIP)]

    palm_axis = middle_mcp[:2] - wrist[:2]
    palm_axis_norm = np.linalg.norm(palm_axis)
    if palm_axis_norm < 1e-6:
        return "unknown"

    thumb_vec = thumb_tip[:2] - wrist[:2]
    thumb_vec_norm = np.linalg.norm(thumb_vec)
    if thumb_vec_norm < 1e-6:
        return "unknown"

    cos_a = np.dot(palm_axis, thumb_vec) / (palm_axis_norm * thumb_vec_norm)
    cos_a = np.clip(cos_a, -1, 1)
    angle = np.arccos(cos_a)

    cross_z = palm_axis[0] * thumb_vec[1] - palm_axis[1] * thumb_vec[0]

    if angle < np.radians(45):
        return "up"
    elif angle > np.radians(135):
        return "down"
    elif cross_z > 0:
        return "right"
    else:
        return "left"


# ---------- Palm orientation (3D aware) ----------

def palm_facing(landmarks: np.ndarray) -> str:
    """Deteksi apakah palm menghadap kamera atau membelakangi."""
    wrist = landmarks[int(Landmark.WRIST)]
    index_mcp = landmarks[int(Landmark.INDEX_MCP)]
    pinky_mcp = landmarks[int(Landmark.PINKY_MCP)]

    v1 = index_mcp[:2] - wrist[:2]
    v2 = pinky_mcp[:2] - wrist[:2]

    cross_z = v1[0] * v2[1] - v1[1] * v2[0]

    if abs(cross_z) < 1e-6:
        return "unknown"
    return "front" if cross_z > 0 else "back"


# ---------- Pinch & spread (scale invariant) ----------

def pinch_distance_normalized(landmarks: np.ndarray) -> float:
    """Distance thumb tip → index tip, normalized by hand scale."""
    thumb_tip = landmarks[int(Landmark.THUMB_TIP)]
    index_tip = landmarks[int(Landmark.INDEX_TIP)]
    raw = distance(thumb_tip, index_tip)
    scale = hand_scale(landmarks)
    if scale < 1e-6:
        return 0.0
    return raw / scale


def finger_spread(landmarks: np.ndarray) -> float:
    """Rata-rata jarak antar fingertip, normalized by hand scale."""
    tips = [landmarks[int(t)] for t in FINGER_TIPS]
    distances = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            distances.append(distance(tips[i], tips[j]))
    if not distances:
        return 0.0
    avg_dist = float(np.mean(distances))
    scale = hand_scale(landmarks)
    if scale < 1e-6:
        return 0.0
    return avg_dist / scale


# ---------- Velocity & motion ----------

def hand_velocity(
    prev_landmarks: np.ndarray,
    curr_landmarks: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Velocity hand center (2D), dalam unit normalized per detik."""
    if dt <= 0:
        return np.zeros(2, dtype=np.float32)
    prev_center = hand_center(prev_landmarks)[:2]
    curr_center = hand_center(curr_landmarks)[:2]
    return ((curr_center - prev_center) / dt).astype(np.float32)


def hand_rotation(landmarks: np.ndarray) -> float:
    """Sudut rotasi tangan (radian) — sudut palm axis dari vertikal."""
    wrist = landmarks[int(Landmark.WRIST)]
    middle_mcp = landmarks[int(Landmark.MIDDLE_MCP)]
    axis = middle_mcp[:2] - wrist[:2]
    if np.linalg.norm(axis) < 1e-6:
        return 0.0
    angle = np.arctan2(axis[0], -axis[1])
    return float(angle)

# ---------- Body-aware features (butuh BodyFrame) ----------

def thumb_direction_in_body_frame(
    landmarks: np.ndarray,
    frame,  # BodyFrame
) -> str:
    """Arah thumb relatif ke tubuh.

    Returns:
        "up", "down", "left", "right", "forward", "backward", "unknown"
    """
    from nexus.vision.context.body_frame import direction_in_body_frame
    from nexus.vision.hand_tracking.landmarks import Landmark

    thumb_mcp = landmarks[int(Landmark.THUMB_MCP)]
    thumb_tip = landmarks[int(Landmark.THUMB_TIP)]
    thumb_vec = thumb_tip - thumb_mcp
    return direction_in_body_frame(thumb_vec, frame)


def index_direction_in_body_frame(
    landmarks: np.ndarray,
    frame,
) -> str:
    """Arah index finger relatif ke tubuh."""
    from nexus.vision.context.body_frame import direction_in_body_frame
    from nexus.vision.hand_tracking.landmarks import Landmark

    index_mcp = landmarks[int(Landmark.INDEX_MCP)]
    index_tip = landmarks[int(Landmark.INDEX_TIP)]
    index_vec = index_tip - index_mcp
    return direction_in_body_frame(index_vec, frame)


def hand_position_in_body_frame(
    landmarks: np.ndarray,
    frame,
) -> np.ndarray:
    """Posisi hand center relatif ke body frame.

    Returns:
        vektor 3D (forward, right, up) relatif ke origin body frame.
        Berguna untuk deteksi "tangan di atas kepala", "tangan di kiri", dll.
    """
    from nexus.vision.context.body_frame import world_to_body_frame
    center = hand_center(landmarks)
    relative = center - frame.origin
    return world_to_body_frame(relative, frame)


def hand_velocity_in_body_frame(
    prev_landmarks: np.ndarray,
    curr_landmarks: np.ndarray,
    dt: float,
    frame,
) -> np.ndarray:
    """Velocity hand center dalam body frame.

    Berguna untuk swipe detection yang konsisten terlepas dari rotasi tubuh.
    """
    from nexus.vision.context.body_frame import world_to_body_frame
    if dt <= 0:
        return np.zeros(3, dtype=np.float32)
    prev_center = hand_center(prev_landmarks)[:2]
    curr_center = hand_center(curr_landmarks)[:2]
    velocity_world = (curr_center - prev_center) / dt
    return world_to_body_frame(velocity_world, frame)


def classify_swipe_in_body_frame(
    velocity_body: np.ndarray,
    *,
    min_velocity: float = 0.3,
) -> tuple[str, float]:
    """Klasifikasi swipe dari velocity dalam body frame.

    Returns:
        (direction, confidence): direction = "up"/"down"/"left"/"right"/"unknown"
    """
    magnitude = float(np.linalg.norm(velocity_body))
    if magnitude < min_velocity:
        return "unknown", 0.0

    # Cari komponen dominan (forward tidak dihitung untuk swipe)
    f, r, u = velocity_body[0], velocity_body[1], velocity_body[2]
    magnitudes = {
        "up": max(0, u),
        "down": max(0, -u),
        "right": max(0, r),
        "left": max(0, -r),
    }
    best = max(magnitudes, key=magnitudes.get)
    confidence = min(1.0, magnitudes[best] / magnitude)

    return best, float(confidence)