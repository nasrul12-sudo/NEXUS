"""Gesture Engine — orchestrator dengan per-hand state."""
from __future__ import annotations

import time

from nexus.brain.mode_manager import NexusMode, get_mode_manager
from nexus.core.config import get_config
from nexus.core.events import EventEnvelope, EventType
from nexus.core.logging import get_logger
from nexus.vision.context.body_frame import BodyFrame
from nexus.vision.gesture.features import (
    hand_center,
    all_finger_curvatures,
    finger_spread
)
from nexus.vision.gesture.gestures import Gesture
from nexus.vision.gesture.rules import (
    SwipeDetector,
    classify_static,
    classify_thumbs_with_body,
)
from nexus.vision.gesture.smoother import GestureSmoother, SmoothedGesture
from nexus.vision.hand_tracking.landmarks import HandLandmarks

log = get_logger(__name__)

WAKE_GESTURE = {Gesture.OPEN_PALM}

SAFE_GESTURE ={
    Gesture.OPEN_PALM,   # wake / exit
    Gesture.FIST,        # alternative exit 
} 

CONFIRMATION_GESTURES = {
    Gesture.THUMBS_DOWN,   # close app
    Gesture.FIST,          # minimize
}


class GestureEngine:
    """Gesture recognition engine dengan per-hand state.

    Setiap tangan (Left/Right) punya SwipeDetector & GestureSmoother sendiri
    untuk menghindari state pollution saat 2 tangan digunakan bersamaan.
    """

    def __init__(self):
        cfg = get_config()
        gcfg = cfg.gesture

        self._enabled = gcfg.enabled

        # Per-hand state (dibuat on-demand)
        self._swipe_detectors: dict[str, SwipeDetector] = {}
        self._smoothers: dict[str, GestureSmoother] = {}

        # Config untuk membuat state per-hand
        self._swipe_cfg = {
            "min_distance": gcfg.swipe_min_distance,
            "max_duration_ms": gcfg.swipe_max_duration_ms,
            "history_size": gcfg.swipe_history_size,
            "min_velocity": getattr(gcfg, "swipe_min_velocity", 0.15),
        }
        self._smoother_cfg = {
            "window": gcfg.smoothing_window,
            "min_stable": gcfg.min_stable_frames,
            "cooldown_ms": gcfg.cooldown_ms,
            "min_confidence": gcfg.min_confidence,
            "hold_duration_s": getattr(gcfg, "hold_duration_s", 0.5),
            "lock_frames": getattr(gcfg, "lock_frames", 5)
        }

        # Thresholds
        self._pinch_threshold = getattr(gcfg, "pinch_distance_threshold", 0.35)
        self._extended_curvature_max = getattr(gcfg, "extended_curvature_max", 0.25)
        self._folded_curvature_min = getattr(gcfg, "folded_curvature_min", 0.40)
        self._half_folded_curvature_min = getattr(gcfg, "half_folded_curvature_min", 0.25)
        self._min_finger_spread_open = getattr(gcfg, "min_finger_spread_open", 0.75)

        # Body-frame thresholds
        self._thumb_curvature_max = getattr(gcfg, "thumb_curvature_max", 0.3)
        self._thumb_others_folded_min = getattr(gcfg, "thumb_others_folded_min", 0.4)

        self._mode_manager = get_mode_manager()

        self._last_timestamp: float | None = None
        self._last_body_frame: BodyFrame | None = None

    def setup(self) -> None:
        log.info(
            "gesture.engine.setup",
            enabled=self._enabled,
            body_frame_support=True,
            per_hand_state=True,
            mode_manager=True
        )

    def set_body_frame(self, body_frame: BodyFrame | None) -> None:
        """Update body frame (shared antar tangan)."""
        self._last_body_frame = body_frame

    def _get_swipe_detector(self, key: str) -> SwipeDetector:
        """Get or create swipe detector untuk tangan tertentu."""
        if key not in self._swipe_detectors:
            self._swipe_detectors[key] = SwipeDetector(**self._swipe_cfg)
            log.debug("gesture.engine.new_swipe_detector", hand=key)
        return self._swipe_detectors[key]

    def _get_smoother(self, key: str) -> GestureSmoother:
        """Get or create smoother untuk tangan tertentu."""
        if key not in self._smoothers:
            self._smoothers[key] = GestureSmoother(**self._smoother_cfg)
            log.debug("gesture.engine.new_smoother", hand=key)
        return self._smoothers[key]

    def check_wake_gesture(self, hands: list[HandLandmarks], timestamp: float) -> bool: 
        if self._mode_manager.mode != NexusMode.IDLE:
            self._mode_manager._wake_gesture_frames = 0
            self._mode_manager._wake_gesture_frames = 0
            return False

        open_palm_count = sum(1 for h in hands if self._is_open_palm_raw(h))

        if open_palm_count < 2:
            missing = getattr(self._mode_manager, "_wake_missing_frame", 0) + 1
            self._mode_manager._wake_gesture_frames = missing

            if missing >= 3:
                self._mode_manager._wake_gesture_frames = 0
                self._mode_manager._wake_gesture_frames = 0
            return False

        self._mode_manager._wake_gesture_frames = 0

        was_idle = self._mode_manager.mode == NexusMode.IDLE
        self._mode_manager.should_process_gesture(
            gesture_name="OPEN_PALM",
            hand_count=2,
            timestamp=timestamp,
        )

        if was_idle and self._mode_manager.mode == NexusMode.LISTENING:
            log.info(
                "gesture.engine.wake_gesture_confirmed",
                hand_count=open_palm_count
            )
            return True
        return False

    def _is_open_palm_raw(self, hand: HandLandmarks) -> bool:
        try:
            curv = all_finger_curvatures(hand.landmarks)
            spread = finger_spread(hand.landmarks)

            all_extended = all(c < 0.4 for c in curv.values())
            enough_spread = spread > 0.6

            return all_extended and enough_spread
        except Exception:
            return False

    def process_hand(
        self,
        hand: HandLandmarks,
        timestamp: float,
    ) -> list[SmoothedGesture]:
        """Process single hand, returns list of smoothed gestures."""
        if not self._enabled:
            return []

        key = hand.handedness  # "Left" atau "Right"
        detector = self._get_swipe_detector(key)
        smoother = self._get_smoother(key)

        results: list[SmoothedGesture] = []
        mode = self._mode_manager.mode

        # Update swipe detector dengan hand center
        center = hand_center(hand.landmarks)[:2]
        detector.update(timestamp, center)

        # ===== 1. SWIPE (prioritas tertinggi, bypass smoother) =====
        if mode == NexusMode.LISTENING:
            swipe_result = detector.detect(timestamp)
            if swipe_result is not None:
                gesture, confidence = swipe_result
                if self._mode_manager.should_process_gesture(
                    gesture_name=gesture.value,
                    hand_count=1,
                    timestamp=timestamp,
                ):
                    smoothed = SmoothedGesture(
                        gesture=gesture,
                        confidence=confidence,
                        duration_ms=0.0,
                        frame_count=1,
                    )
                    results.append(smoothed)
                    detector.clear()

                    log.debug(
                        "gesture.engine.swipe_published",
                        gesture=gesture.value,
                        confidence=round(confidence, 2),
                        hand=key
                    )

                self._last_timestamp = timestamp
                return results

        # ===== 2. THUMBS =====
        gesture_thumbs, conf_thumbs = classify_thumbs_with_body(
            hand,
            self._last_body_frame,
            thumb_curvature_max=self._thumb_curvature_max,
            others_folded_min=self._thumb_others_folded_min,
        )
        if gesture_thumbs != Gesture.UNKNOWN:
            smoothed = smoother.update(gesture_thumbs, conf_thumbs, timestamp)
            if smoothed is not None:
                if self._mode_manager.should_process_gesture(
                    gesture_name=smoothed.gesture.value,
                    hand_count=1,
                    timestamp=timestamp,
                ):
                    results.append(smoothed)
                    log.debug(
                        "gesture.engine.thumbs_published",
                        gesture=smoothed.gesture.value,
                        confidence=round(smoothed.confidence, 2),
                        hand=key,
                        mode=mode.value,
                    )

                self._last_timestamp = timestamp
                return results

        # ===== 3. STATIC =====
        gesture, confidence = classify_static(
            hand,
            pinch_threshold=self._pinch_threshold,
            extended_curvature_max=self._extended_curvature_max,
            folded_curvature_min=self._folded_curvature_min,
            half_folded_curvature_min=self._half_folded_curvature_min,
            min_finger_spread_open=self._min_finger_spread_open,
        )

        smoothed = smoother.update(gesture, confidence, timestamp)
        if smoothed is not None:
            if self._mode_manager.should_process_gesture(
                gesture_name=smoothed.gesture.value,
                hand_count=1,
                timestamp=timestamp,
            ):
                results.append(smoothed)
                log.debug(
                    "gesture.engine.static_published",
                    gesture=smoothed.gesture.value,
                    confidence=round(smoothed.confidence, 2),
                    hend=key,
                    mode=mode.value
                )

        self._last_timestamp = timestamp
        return results

    def process_hand_lost(self, timestamp: float, handedness: str | None = None) -> None:
        """Dipanggil saat tangan hilang.

        Args:
            timestamp: waktu
            handedness: "Left"/"Right" — kalau None, reset semua
        """
        if handedness is None:
            # Reset semua
            for smoother in self._smoothers.values():
                smoother.reset()
                smoother.reset_last_published()
            for detector in self._swipe_detectors.values():
                detector.clear()
            log.debug("gesture.engine.reset_all_hands")
        else:
            if handedness in self._smoothers:
                self._smoothers[handedness].reset()
                self._smoothers[handedness].reset_last_published()
            if handedness in self._swipe_detectors:
                self._swipe_detectors[handedness].clear()
            log.debug("gesture.engine.reset_hand", hand=handedness)

        self._last_timestamp = None

    def get_status(self):
        return {
            "mode": self._mode_manager.mode.value,
            "handedness_tracked": list(self._smoothers.keys()),
            "body_frame_available": self._last_body_frame is not None,
            "last_timestamp": self._last_timestamp,
        }


def make_gesture_event(
    smoothed: SmoothedGesture,
    hand: HandLandmarks,
    body_frame: BodyFrame | None = None,
    mode: str | None = None,
) -> EventEnvelope:
    """Build event payload."""
    center = hand_center(hand.landmarks)
    payload = {
        "gesture": smoothed.gesture.value,
        "confidence": smoothed.confidence,
        "duration_ms": smoothed.duration_ms,
        "frame_count": smoothed.frame_count,
        "handedness": hand.handedness,
        "hand_confidence": hand.confidence,
        "position": [float(center[0]), float(center[1])],
        "source": "rule_based",
        "body_frame_available": body_frame is not None,
        "mode": mode or "unknown",
    }
    if body_frame is not None:
        payload["body_frame"] = body_frame.to_dict()
    return EventEnvelope(
        type=EventType.GESTURE_STABLE.value,
        source="vision",
        payload=payload,
    )