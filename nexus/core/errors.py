"""Error taxonomy untuk NEXUS."""
from __future__ import annotations


class NexusError(Exception):
    """Base exception untuk semua error NEXUS."""

    code: str = "NEXUS_ERROR"
    recoverable: bool = False

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context,
        }


class ConfigError(NexusError):
    code = "CONFIG_ERROR"


class EventSchemaError(NexusError):
    code = "EVENT_SCHEMA_ERROR"


class BusError(NexusError):
    code = "BUS_ERROR"
    recoverable = True


class StateError(NexusError):
    code = "STATE_ERROR"
    recoverable = True


class ResourceError(NexusError):
    code = "RESOURCE_ERROR"
    recoverable = True


class ResourceTimeout(ResourceError):
    code = "RESOURCE_TIMEOUT"
    recoverable = True


class ProcessError(NexusError):
    code = "PROCESS_ERROR"


class SafetyViolation(NexusError):
    code = "SAFETY_VIOLATION"


class CommandError(NexusError):
    code = "COMMAND_ERROR"
    recoverable = True