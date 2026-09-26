"""OutputChannel abstraction.

Setiap channel menerima OutputMessage dan menampilkannya dengan caranya sendiri.
Channel bisa di-enable/disable via config, dan punya filter berdasarkan
channel type.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputChannelType(str, Enum):
    """Tipe channel — untuk filtering."""

    TEXT = "text"           # untuk terminal
    SPEAK = "speak"         # untuk TTS
    NOTIFY = "notify"       # untuk desktop notification
    SOUND = "sound"         # untuk sound effects


class OutputPriority(str, Enum):
    """Priority level untuk filter."""

    DEBUG = "debug"
    INFO = "info"
    NORMAL = "normal"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class OutputMessage:
    """Pesan output yang dikirim ke semua channel.

    Channel akan memilih apa yang mau ditampilkan berdasarkan
    channel_type, priority, dan source.
    """

    text: str
    channel_type: OutputChannelType
    priority: OutputPriority = OutputPriority.NORMAL
    source: str = "system"                   # vision/voice/brain/action/system
    event_type: str | None = None            # event type yang trigger
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def should_display(self, min_priority: OutputPriority) -> bool:
        """Cek apakah message memenuhi minimum priority."""
        order = [
            OutputPriority.DEBUG,
            OutputPriority.INFO,
            OutputPriority.NORMAL,
            OutputPriority.WARNING,
            OutputPriority.ERROR,
            OutputPriority.CRITICAL,
        ]
        return order.index(self.priority) >= order.index(min_priority)


class OutputChannel(ABC):
    """Base class untuk semua output channel."""

    #: Nama channel (untuk logging & config)
    name: str = "channel"

    @abstractmethod
    def start(self) -> None:
        """Initialize channel. Dipanggil sekali."""

    @abstractmethod
    def output(self, message: OutputMessage) -> None:
        """Process message. Channel memutuskan mau display atau tidak."""

    @abstractmethod
    def stop(self) -> None:
        """Cleanup channel."""

    @property
    def is_available(self) -> bool:
        """True jika channel siap menerima message."""
        return True