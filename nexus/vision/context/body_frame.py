"""Body frame extraction dari pose landmarks.

Body frame adalah sistem koordinat lokal user:
    - forward: vektor dari bahu ke kamera (atau dari hip ke bahu)
    - up: vektor dari hip ke bahu (vertikal tubuh)
    - right: cross product forward × up

Dengan body frame, gesture bisa dipahami secara SEMANTIK:
    - "thumbs up" = thumb sejajar dengan body.up
    - "point forward" = index sejajar dengan body.forward
    - "swipe right" = velocity sejajar dengan body.right

Tanpa body frame, gesture hanya relatif ke frame kamera.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus.vision.pose.landmarks import PoseLandmark, PoseLandmarks


@dataclass
class BodyFrame:
    """Sistem koordinat lokal user.

    Attributes:
        origin: titik origin (biasanya mid-shoulder), shape (3,)
        forward: unit vector (arah depan tubuh), shape (3,)
        up: unit vector (arah atas tubuh), shape (3,)
        right: unit vector (arah kanan tubuh), shape (3,)
        shoulder_width: lebar bahu (untuk normalisasi scale), scalar
        visibility: rata-rata visibility landmark yang dipakai
    """

    origin: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    right: np.ndarray
    shoulder_width: float
    visibility: float

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "forward": self.forward.tolist(),
            "up": self.up.tolist(),
            "right": self.right.tolist(),
            "shoulder_width": float(self.shoulder_width),
            "visibility": float(self.visibility),
        }


class BodyFrameExtractor:
    """Extract body frame dari pose landmarks.

    Membutuhkan minimal: LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP.
    Kalau visibility rendah, return None.
    """

    MIN_VISIBILITY = 0.5

    def extract(self, pose: PoseLandmarks) -> BodyFrame | None:
        """Extract body frame. Returns None jika pose tidak cukup visible."""
        ls = pose.get(PoseLandmark.LEFT_SHOULDER)
        rs = pose.get(PoseLandmark.RIGHT_SHOULDER)
        lh = pose.get(PoseLandmark.LEFT_HIP)
        rh = pose.get(PoseLandmark.RIGHT_HIP)

        # Cek visibility
        vis_indices = [
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP,
        ]
        vis_values = [pose.visibility[int(i)] for i in vis_indices]
        avg_vis = float(np.mean(vis_values))

        if avg_vis < self.MIN_VISIBILITY:
            return None

        # Origin: mid-shoulder (bisa juga mid-hip, tapi mid-shoulder lebih stabil)
        origin = ((ls + rs) / 2).astype(np.float32)

        # Up: dari mid-hip ke mid-shoulder
        mid_hip = (lh + rh) / 2
        up_raw = origin - mid_hip

        # Forward: dari origin ke kamera (z negatif)
        # Karena kita tidak tahu orientasi kamera, kita asumsikan:
        # - forward = arah dari tubuh ke kamera
        # - Estimasi: cross product dari shoulder vector dan up vector
        shoulder_vec = rs - ls  # kiri ke kanan
        # Forward = cross(shoulder, up) → arah tegak lurus
        forward_raw = np.cross(shoulder_vec, up_raw)

        # Normalize
        up = self._normalize(up_raw)
        forward = self._normalize(forward_raw)

        if up is None or forward is None:
            return None

        # Right: cross(forward, up) — orthogonalize
        right = self._normalize(np.cross(forward, up))
        if right is None:
            return None

        # Re-orthogonalize up (Gram-Schmidt) agar benar-benar orthogonal
        up = self._normalize(np.cross(right, forward))
        if up is None:
            return None

        # Shoulder width: normalisasi scale
        shoulder_width = float(np.linalg.norm((rs - ls)[:2]))
        if shoulder_width < 1e-6:
            return None

        return BodyFrame(
            origin=origin,
            forward=forward,
            up=up,
            right=right,
            shoulder_width=shoulder_width,
            visibility=avg_vis,
        )

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray | None:
        n = np.linalg.norm(v[:2])  # hanya 2D (x,y)
        if n < 1e-6:
            return None
        result = np.array([v[0] / n, v[1] / n, 0.0], dtype=np.float32)
        return result


# ---------- Helper functions untuk gesture ----------

def world_to_body_frame(vec: np.ndarray, frame: BodyFrame) -> np.ndarray:
    """Transform vector dari world (image) coords ke body frame.

    Args:
        vec: vektor 3D dalam image coords
        frame: BodyFrame

    Returns:
        vektor dalam body frame: (forward, right, up) components
    """
    # Project ke masing-masing axis
    f = float(np.dot(vec[:2], frame.forward[:2]))
    r = float(np.dot(vec[:2], frame.right[:2]))
    u = float(np.dot(vec[:2], frame.up[:2]))
    return np.array([f, r, u], dtype=np.float32)


def direction_in_body_frame(
    vec: np.ndarray,
    frame: BodyFrame,
) -> str:
    """Tentukan arah dominan vector dalam body frame.

    Returns:
        "up", "down", "left", "right", "forward", "backward", "unknown"
    """
    components = world_to_body_frame(vec, frame)
    f, r, u = components[0], components[1], components[2]

    # Cari komponen dominan (absolute value)
    magnitudes = {
        "up": max(0, u),
        "down": max(0, -u),
        "right": max(0, r),
        "left": max(0, -r),
        "forward": max(0, f),
        "backward": max(0, -f),
    }

    best_dir = max(magnitudes, key=magnitudes.get)
    if magnitudes[best_dir] < 0.3:  # threshold dominance
        return "unknown"
    return best_dir


def angle_in_body_frame(
    vec: np.ndarray,
    frame: BodyFrame,
) -> float:
    """Sudut vektor dalam body frame (radian) dari sumbu forward.

    Returns:
        Sudut 0..π (0 = forward, π/2 = right/up, π = backward)
    """
    components = world_to_body_frame(vec, frame)
    f, r, u = components
    # Sudut dari forward axis
    horiz = np.sqrt(r * r + u * u)
    return float(np.arctan2(horiz, f))