"""TTS output channel — stub untuk sekarang, implementasi penuh di Phase 5."""
from __future__ import annotations

from nexus.core.logging import get_logger
from nexus.output.channels.base import (
    OutputChannel,
    OutputChannelType,
    OutputMessage,
)


log = get_logger(__name__)


class TTSOutputChannel(OutputChannel):
    """Text-to-speech channel.

    Provider abstraction — implementasi konkret (Piper, Coqui) akan
    ditambahkan di Phase 5. Untuk sekarang, channel ini hanya log
    apa yang akan diucapkan.
    """

    name = "tts"

    def __init__(
        self,
        *,
        provider: str = "piper",
        voice: str = "en_US-lessac-medium",
        speak_on: list[str] | None = None,
    ):
        self._provider = provider
        self._voice = voice
        self._speak_on = set(speak_on or ["voice_response", "error", "confirmation"])
        self._tts_impl: Any = None
        self._enabled = True

    def start(self) -> None:
        log.info(
            "output.tts.start",
            provider=self._provider,
            voice=self._voice,
            speak_on=list(self._speak_on),
        )
        # Phase 5: load Piper model di sini
        # self._tts_impl = PiperProvider(voice=self._voice)
        # self._tts_impl.setup()

    def output(self, message: OutputMessage) -> None:
        if message.channel_type != OutputChannelType.SPEAK:
            return
        # Phase 5: self._tts_impl.speak(message.text)
        log.info("output.tts.speak", text=message.text[:100])

    def stop(self) -> None:
        if self._tts_impl is not None:
            try:
                self._tts_impl.close()
            except Exception:
                log.exception("output.tts.close_error")


# Type hint untuk import Any
from typing import Any  # noqa: E402