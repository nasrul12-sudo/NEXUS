"""Gesture definitions."""
from __future__ import annotations

from enum import Enum


class Gesture(str, Enum):
    # Static single-hand
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    POINT = "POINT"
    PINCH = "PINCH"
    TWO_FINGER = "TWO_FINGER"
    THREE_FINGER = "THREE_FINGER"     # NEW
    ROCK = "ROCK"                     # NEW (index + pinky)
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"
    THUMBS_LEFT = "THUMBS_LEFT"       # NEW
    THUMBS_RIGHT = "THUMBS_RIGHT"     # NEW

    # Dynamic
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    SWIPE_UP = "SWIPE_UP"
    SWIPE_DOWN = "SWIPE_DOWN"

    # Two-hand
    TWO_HAND_PINCH = "TWO_HAND_PINCH"
    TWO_HAND_ZOOM = "TWO_HAND_ZOOM"
    TWO_HAND_ROTATE = "TWO_HAND_ROTATE"

    # Special
    UNKNOWN = "UNKNOWN"
    NO_HAND = "NO_HAND"


STATIC_GESTURES = {
    Gesture.OPEN_PALM,
    Gesture.FIST,
    Gesture.POINT,
    Gesture.PINCH,
    Gesture.TWO_FINGER,
    Gesture.THREE_FINGER,
    Gesture.ROCK,
    Gesture.THUMBS_UP,
    Gesture.THUMBS_DOWN,
    Gesture.THUMBS_LEFT,
    Gesture.THUMBS_RIGHT,
}

DYNAMIC_GESTURES = {
    Gesture.SWIPE_LEFT,
    Gesture.SWIPE_RIGHT,
    Gesture.SWIPE_UP,
    Gesture.SWIPE_DOWN,
}

TWO_HAND_GESTURES = {
    Gesture.TWO_HAND_PINCH,
    Gesture.TWO_HAND_ZOOM,
    Gesture.TWO_HAND_ROTATE,
}