"""Sound effect channel — stub untuk sekarang."""
from __future__ import annotations

from nexus.core.logging import get_logger
from nexus.output.channels.base import (
    OutputChannel,
    OutputChannelType,
    OutputMessage,
)


log = get_logger(__name__)


class SoundOutputChannel(OutputChannel):
    """Sound effect channel.

    Implementasi konkret akan ditambahkan kemudian dengan pygame
    atau simpleaudio. Untuk sekarang hanya log.
    """

    name = "sound"

    def __init__(self, *, on: list[str] | None = None):
        self._on = set(on or ["gesture_confirmed", "wake_word"])
        self._enabled = True

    def start(self) -> None:
        log.info("output.sound.start", on=list(self._on))

    def output(self, message: OutputMessage) -> None:
        if message.channel_type != OutputChannelType.SOUND:
            return
        # Future: play sound
        log.debug("output.sound.play", event=message.event_type)

    def stop(self) -> None:
        pass