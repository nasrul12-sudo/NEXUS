from nexus.core.events import EventEnvelope, EventType, make_event, EventMeta


def test_make_event():
    evt = make_event(EventType.GESTURE_STABLE, "vision", {"gesture": "PINCH"})
    assert evt.type == "gesture.stable"
    assert evt.source == "vision"
    assert evt.payload["gesture"] == "PINCH"


def test_roundtrip():
    evt = make_event(EventType.VOICE_TRANSCRIPT, "voice", {"text": "open chrome"})
    wire = evt.to_wire()
    restored = EventEnvelope.from_wire(wire)
    assert restored.id == evt.id
    assert restored.type == evt.type
    assert restored.payload == evt.payload


def test_invalid_type():
    import pytest
    with pytest.raises(Exception):
        EventEnvelope(type="invalid", source="vision")