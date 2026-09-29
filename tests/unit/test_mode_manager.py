import time 
import pytest

from nexus.brain.mode_manager import ModeManager, NexusMode

def test_initial_mode_idle():
    mm = ModeManager()
    mm.start(fps=15.0)
    assert mm.mode == NexusMode.IDLE
    assert not mm.is_active

def test_activate_deactivate():
    mm = ModeManager()
    mm.start(fps=15.0)
    mm.activate(source="test")
    assert mm.mode == NexusMode.LISTENING
    assert mm.is_active
    mm.deactivate(source="test")
    assert mm.mode == NexusMode.IDLE

def test_gesture_rejected_when_idle():
    mm = ModeManager()
    mm.start(fps=15.0)
    assert not mm.should_process_gesture("POINT", hand_count=1)

def test_wake_gesture_needed_hold():
    mm = ModeManager()
    mm.start(fps=15.0)
    assert not mm.should_process_gesture("OPEN_PALM", hand_count=2)
    assert mm.mode == NexusMode.IDLE

    for _ in range(20):
        mm.should_process_gesture("OPEN_PALM", hand_count=2)
    assert mm.mode == NexusMode.LISTENING

def test_timeout_returns_idle():
    mm = ModeManager()
    mm._config.listening_timeout_s = 0.1
    mm.start(fps=15.0)
    mm.activate(source="test")
    assert mm.mode == NexusMode.LISTENING
    time.sleep(0.15)

    mm.should_process_gesture("POINT", hand_count=1)
    assert mm.mode == NexusMode.IDLE
