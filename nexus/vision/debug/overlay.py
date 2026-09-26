"""Drawing primitives untuk debug visualization.

Semua fungsi menerima BGR image (OpenCV convention) dan menggambar
overlay. Tidak ada state — fungsi murni.

Referensi:
  - MediaPipe hand connections:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from nexus.vision.hand_tracking.landmarks import (
    FINGER_MCPS,
    FINGER_PIPS,
    FINGER_TIPS,
    HandLandmarks,
    Landmark,
)


# ---------- Hand skeleton connections ----------

# Connections untuk 21 landmark MediaPipe
# Format: (start_idx, end_idx, color_group)
# color_group: "thumb", "index", "middle", "ring", "pinky", "palm"
HAND_CONNECTIONS: list[tuple[int, int, str]] = [
    # Thumb
    (0, 1, "thumb"), (1, 2, "thumb"), (2, 3, "thumb"), (3, 4, "thumb"),
    # Index
    (0, 5, "index"), (5, 6, "index"), (6, 7, "index"), (7, 8, "index"),
    # Middle
    (5, 9, "palm"), (9, 10, "middle"), (10, 11, "middle"), (11, 12, "middle"),
    # Ring
    (9, 13, "palm"), (13, 14, "ring"), (14, 15, "ring"), (15, 16, "ring"),
    # Pinky
    (13, 17, "palm"), (17, 18, "pinky"), (18, 19, "pinky"), (19, 20, "pinky"),
    # Palm base
    (0, 17, "palm"),
]


# BGR colors (OpenCV)
COLOR_MAP = {
    "palm":    (200, 200, 200),    # light gray
    "thumb":   (0, 165, 255),      # orange
    "index":   (0, 255, 0),        # green
    "middle":  (255, 255, 0),      # cyan
    "ring":    (255, 0, 255),      # magenta
    "pinky":   (0, 128, 255),      # light blue
    "tip":     (0, 0, 255),        # red (landmark dots)
    "wrist":   (255, 255, 255),    # white
}


# ---------- Drawing functions ----------

def draw_hand_skeleton(
    image: np.ndarray,
    hand: HandLandmarks,
    *,
    line_thickness: int = 2,
    dot_radius: int = 4,
    highlight_tips: bool = True,
    show_labels: bool = False,
) -> None:
    """Draw hand skeleton pada image (in-place).

    Args:
        image: BGR image (H, W, 3), akan dimodifikasi in-place
        hand: HandLandmarks dari MediaPipe
        line_thickness: ketebalan garis koneksi
        dot_radius: radius titik landmark
        highlight_tips: jika True, fingertip digambar lebih besar
        show_labels: jika True, tampilkan label landmark (WRIST, INDEX_TIP, dll)
    """
    h, w = image.shape[:2]
    landmarks_px = _normalized_to_pixel(hand.landmarks, w, h)

    # Draw connections
    for start_idx, end_idx, group in HAND_CONNECTIONS:
        color = COLOR_MAP.get(group, (255, 255, 255))
        cv2.line(
            image,
            landmarks_px[start_idx],
            landmarks_px[end_idx],
            color,
            line_thickness,
            cv2.LINE_AA,
        )

    # Draw landmarks
    for i, pt in enumerate(landmarks_px):
        is_tip = i in {int(t) for t in FINGER_TIPS}
        is_wrist = i == int(Landmark.WRIST)

        if is_wrist:
            color = COLOR_MAP["wrist"]
            radius = dot_radius + 2
        elif is_tip and highlight_tips:
            color = COLOR_MAP["tip"]
            radius = dot_radius + 1
        else:
            color = (0, 255, 255)  # yellow
            radius = dot_radius

        cv2.circle(image, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, pt, radius, (0, 0, 0), 1, cv2.LINE_AA)  # outline

    # Optional labels
    if show_labels:
        for i, pt in enumerate(landmarks_px):
            try:
                name = Landmark(i).name
            except ValueError:
                name = str(i)
            # Skip labels untuk non-tips supaya tidak terlalu ramai
            if i not in {int(t) for t in FINGER_TIPS} and i != int(Landmark.WRIST):
                continue
            cv2.putText(
                image,
                name,
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def draw_hand_label(
    image: np.ndarray,
    hand: HandLandmarks,
    *,
    position: tuple[int, int] | None = None,
    font_scale: float = 0.6,
) -> None:
    """Draw handedness + confidence label dekat wrist.

    Args:
        image: BGR image
        hand: HandLandmarks
        position: optional (x, y) pixel. Kalau None, pakai wrist position.
        font_scale: skala font
    """
    h, w = image.shape[:2]

    if position is None:
        wrist = hand.landmarks[int(Landmark.WRIST)]
        px = int(wrist[0] * w)
        py = int(wrist[1] * h)
        position = (px, py - 20)

    text = f"{hand.handedness} {hand.confidence:.2f}"

    # Background untuk readability
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    x, y = position
    cv2.rectangle(
        image,
        (x - 4, y - text_h - 6),
        (x + text_w + 4, y + baseline - 2),
        (0, 0, 0),
        -1,
    )

    # Color berdasarkan handedness
    color = (0, 255, 0) if hand.handedness == "Right" else (255, 150, 0)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_hud_text(
    image: np.ndarray,
    lines: Iterable[str],
    *,
    position: tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: tuple[int, int, int] = (0, 255, 0),
    line_height: int = 25,
    background: bool = True,
) -> None:
    """Draw HUD text block di pojok.

    Args:
        image: BGR image
        lines: iterable of string, satu per baris
        position: (x, y) pixel untuk baris pertama
        font_scale: skala font
        color: BGR color
        line_height: jarak antar baris
        background: jika True, gambar background semi-transparan
    """
    lines = list(lines)
    if not lines:
        return

    if background:
        # Hitung bounding box
        max_w = 0
        for line in lines:
            (w, _), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            max_w = max(max_w, w)
        total_h = line_height * len(lines) + 10
        x, y = position
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x - 8, y - 22),
            (x + max_w + 8, y - 22 + total_h),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    x, y = position
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_fps_indicator(
    image: np.ndarray,
    fps: float,
    *,
    position: tuple[int, int] | None = None,
    font_scale: float = 0.7,
) -> None:
    """Draw FPS indicator di pojok kanan atas."""
    h, w = image.shape[:2]
    if position is None:
        position = (w - 120, 30)

    text = f"FPS: {fps:.1f}"

    # Color: hijau jika >= 25, kuning jika >= 15, merah jika < 15
    if fps >= 25:
        color = (0, 255, 0)
    elif fps >= 15:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_center_crosshair(
    image: np.ndarray,
    *,
    size: int = 20,
    color: tuple[int, int, int] = (100, 100, 100),
    thickness: int = 1,
) -> None:
    """Draw crosshair di tengah image (referensi visual)."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)
    

def draw_feature_debug(
    image: np.ndarray,
    hand: HandLandmarks,
    *,
    position: tuple[int, int] = (10, 10),
) -> None:
    """Tampilkan curvature & spread untuk debugging."""
    from nexus.vision.gesture.features import (
        all_finger_curvatures,
        finger_spread,
        palm_facing,
    )

    curvatures = all_finger_curvatures(hand.landmarks)
    spread = finger_spread(hand.landmarks)
    facing = palm_facing(hand.landmarks)

    lines = [f"Curvatures ({hand.handedness}):"]
    for name, c in curvatures.items():
        bar = "█" * int(c * 10)
        lines.append(f"  {name:6s}: {c:.2f} {bar}")
    lines.append(f"  spread: {spread:.2f}")
    lines.append(f"  facing: {facing}")

    draw_hud_text(
        image, lines,
        position=position,
        font_scale=0.45,
        color=(200, 200, 100),
        line_height=16,
    )


# ---------- Helpers ----------

def _normalized_to_pixel(
    landmarks: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Convert normalized landmarks [0,1] ke pixel coordinates.

    Args:
        landmarks: shape (21, 3), nilai normalized
        width: image width
        height: image height

    Returns:
        list of (x, y) tuples, shape (21, 2)
    """
    pixels: list[tuple[int, int]] = []
    for lm in landmarks:
        x = int(np.clip(lm[0] * width, 0, width - 1))
        y = int(np.clip(lm[1] * height, 0, height - 1))
        pixels.append((x, y))
    return pixels

def draw_pose_skeleton(
    image: np.ndarray,
    pose,  # PoseLandmarks
    *,
    line_thickness: int = 2,
    dot_radius: int = 3,
    min_visibility: float = 0.5,
) -> None:
    """Draw pose skeleton pada image."""
    from nexus.vision.pose.landmarks import POSE_CONNECTIONS

    h, w = image.shape[:2]
    lms = pose.landmarks
    vis = pose.visibility

    def to_px(idx):
        lm = lms[idx]
        x = int(np.clip(lm[0] * w, 0, w - 1))
        y = int(np.clip(lm[1] * h, 0, h - 1))
        return (x, y)

    # Draw connections
    for start, end in POSE_CONNECTIONS:
        if vis[start] < min_visibility or vis[end] < min_visibility:
            continue
        cv2.line(image, to_px(start), to_px(end), (200, 100, 200), line_thickness, cv2.LINE_AA)

    # Draw landmarks
    for i in range(len(lms)):
        if vis[i] < min_visibility:
            continue
        cv2.circle(image, to_px(i), dot_radius, (255, 100, 200), -1, cv2.LINE_AA)


def draw_body_frame(
    image: np.ndarray,
    body_frame,  # BodyFrame
    *,
    axis_length: float = 0.15,
    thickness: int = 3,
) -> None:
    """Draw body frame axes (forward, right, up) dari origin."""
    h, w = image.shape[:2]

    origin = body_frame.origin
    ox = int(origin[0] * w)
    oy = int(origin[1] * h)

    # Forward: biru, Right: hijau, Up: merah
    for axis, color in [
        (body_frame.forward, (255, 0, 0)),   # BGR blue
        (body_frame.right, (0, 255, 0)),     # BGR green
        (body_frame.up, (0, 0, 255)),        # BGR red
    ]:
        ex = int((origin[0] + axis[0] * axis_length) * w)
        ey = int((origin[1] + axis[1] * axis_length) * h)
        cv2.arrowedLine(
            image, (ox, oy), (ex, ey),
            color, thickness, cv2.LINE_AA, tipLength=0.3,
        )