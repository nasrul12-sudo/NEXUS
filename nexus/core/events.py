"""Event schema & envelope untuk NEXUS.

Semua komunikasi antar-proses menggunakan EventEnvelope.

Serialisasi:
  - ZMQ multipart: [topic_frame, payload_frame]
  - topic_frame: byte prefix untuk SUB filter (misal b"gesture.stable")
  - payload_frame: JSON envelope

Catatan penting:
  ZMQ SUB filter match pada byte prefix frame pertama. Kita kirim topic
  sebagai frame terpisah agar filter bekerja. Jika kita kirim JSON
  single-frame (dimulai '{'), filter tidak akan pernah match.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from nexus.core.errors import EventSchemaError


# ---------- Source & Type ----------

Source = Literal["vision", "voice", "brain", "action", "system", "api", "launcher"]


class EventType(str, Enum):
    """Canonical event types."""

    # Vision
    HAND_LANDMARKS = "hand.landmarks"
    HAND_LOST = "hand.lost"
    GESTURE_RAW = "gesture.raw"
    GESTURE_STABLE = "gesture.stable"
    SCREEN_CONTEXT = "screen.context"

    # Voice
    VOICE_WAKEWORD = "voice.wakeword"
    VOICE_SPEECH_START = "voice.speech_start"
    VOICE_SPEECH_END = "voice.speech_end"
    VOICE_TRANSCRIPT = "voice.transcript"
    VOICE_STATE = "voice.state"

    # Brain
    INTENT_RESOLVED = "intent.resolved"
    INTENT_PLAN = "intent.plan"
    INTENT_AMBIGUOUS = "intent.ambiguous"
    RESPONSE_TEXT = "response.text"

    # Action
    COMMAND_REQUEST = "command.request"
    COMMAND_RESULT = "command.result"
    COMMAND_CONFIRM_REQUIRED = "command.confirm_required"
    COMMAND_CONFIRMED = "command.confirmed"

    # System
    SYSTEM_STATUS = "system.status"
    SYSTEM_STATE = "system.state"
    HUD_STATE = "hud.state"

    # output (untuk channel)
    OUTPUT_TEXT = "output.text" 
    OUTPUT_SPEAK = "output.speak"
    OUTPUT_NOTIFY = "output.notify"
    OUTPUT_SOUND = "output.sound"

    # Lifecycle
    PROCESS_READY = "process.ready"
    PROCESS_ERROR = "process.error"
    PROCESS_SHUTDOWN = "process.shutdown"


# ---------- Envelope ----------

class EventMeta(BaseModel):
    latency_ms: float | None = None
    confidence: float | None = None
    trace_id: str | None = None
    vram_used_mb: int | None = None
    model: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EventEnvelope(BaseModel):
    """Canonical event envelope."""

    id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    type: str
    ts: float = Field(default_factory=time.time)
    source: Source
    payload: dict[str, Any] = Field(default_factory=dict)
    meta: EventMeta = Field(default_factory=EventMeta)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if "." not in v:
            raise ValueError(f"Event type must use dot notation: {v}")
        return v

    @field_validator("ts")
    @classmethod
    def _validate_ts(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    # ---------- Serialization ----------

    @property
    def topic(self) -> str:
        """ZMQ topic untuk filtering.

        Format: type (misal "gesture.stable", "voice.transcript").
        Subscriber bisa filter dengan prefix seperti "gesture." atau "voice.".
        """
        return self.type

    def to_wire_multipart(self) -> list[bytes]:
        """Serialize untuk ZMQ multipart: [topic_frame, payload_frame].

        Frame 1 (topic): byte prefix untuk SUB filter.
        Frame 2 (payload): JSON envelope lengkap.
        """
        return [
            self.topic.encode("utf-8"),
            self.model_dump_json().encode("utf-8"),
        ]

    def to_wire(self) -> bytes:
        """Legacy single-frame serialization (JSON only).

        Deprecated: gunakan to_wire_multipart() untuk ZMQ.
        Dipertahankan untuk kompatibilitas & debugging.
        """
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_wire(cls, data: bytes) -> EventEnvelope:
        """Legacy single-frame deserialization."""
        try:
            return cls.model_validate_json(data)
        except Exception as e:
            raise EventSchemaError(f"Invalid event payload: {e}") from e

    @classmethod
    def from_wire_multipart(cls, frames: list[bytes]) -> EventEnvelope:
        """Deserialize dari ZMQ multipart frames.

        Args:
            frames: [topic, payload] — topic diabaikan (sudah difilter ZMQ),
                    payload di-parse sebagai JSON envelope.
        """
        if len(frames) < 2:
            raise EventSchemaError(
                f"Expected 2 frames (topic, payload), got {len(frames)}"
            )
        payload = frames[-1]  # frame terakhir = JSON envelope
        try:
            return cls.model_validate_json(payload)
        except Exception as e:
            raise EventSchemaError(f"Invalid event payload: {e}") from e


# ---------- Helpers ----------

def make_event(
    type: str | EventType,
    source: Source,
    payload: dict[str, Any] | None = None,
    *,
    trace_id: str | None = None,
    confidence: float | None = None,
    latency_ms: float | None = None,
    **meta_extra: Any,
) -> EventEnvelope:
    """Convenience factory untuk membuat event."""
    return EventEnvelope(
        type=type.value if isinstance(type, EventType) else type,
        source=source,
        payload=payload or {},
        meta=EventMeta(
            trace_id=trace_id,
            confidence=confidence,
            latency_ms=latency_ms,
            extra=meta_extra,
        ),
    )