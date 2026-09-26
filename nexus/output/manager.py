"""OutputManager — fan-out message ke semua channel."""
from __future__ import annotations

import threading

from nexus.core.config import get_config
from nexus.core.logging import get_logger
from nexus.output.channels import (
    NotificationOutputChannel,
    OutputChannel,
    OutputChannelType,
    OutputMessage,
    OutputPriority,
    SoundOutputChannel,
    TerminalOutputChannel,
    TTSOutputChannel,
)


log = get_logger(__name__)


class OutputManager:
    """Fan-out output messages ke semua channel.

    Usage:
        mgr = OutputManager()
        mgr.start()
        mgr.emit("Opening Chrome", channel_type=OutputChannelType.TEXT, source="action")
        mgr.stop()
    """

    def __init__(self):
        self._channels: list[OutputChannel] = []
        self._lock = threading.Lock()
        self._started = False

    def register(self, channel: OutputChannel) -> None:
        with self._lock:
            self._channels.append(channel)
            if self._started:
                channel.start()
        log.info("output.channel.registered", name=channel.name)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            for ch in self._channels:
                try:
                    ch.start()
                except Exception:
                    log.exception("output.channel.start_error", name=ch.name)
            self._started = True
        log.info("output.manager.started", channels=[c.name for c in self._channels])

    def stop(self) -> None:
        with self._lock:
            for ch in self._channels:
                try:
                    ch.stop()
                except Exception:
                    log.exception("output.channel.stop_error", name=ch.name)
            self._started = False
        log.info("output.manager.stopped")

    def emit(
        self,
        text: str,
        *,
        channel_type: OutputChannelType,
        priority: OutputPriority = OutputPriority.NORMAL,
        source: str = "system",
        event_type: str | None = None,
        payload: dict | None = None,
    ) -> None:
        """Emit message ke semua channel."""
        message = OutputMessage(
            text=text,
            channel_type=channel_type,
            priority=priority,
            source=source,
            event_type=event_type,
            payload=payload or {},
        )
        self.emit_message(message)

    def emit_message(self, message: OutputMessage) -> None:
        with self._lock:
            channels = list(self._channels)
        for ch in channels:
            try:
                ch.output(message)
            except Exception:
                log.exception("output.channel.error", name=ch.name)


# ---------- Factory ----------

_manager: OutputManager | None = None
_manager_lock = threading.Lock()


def get_output_manager() -> OutputManager:
    """Get or create global OutputManager dengan channels dari config."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            return _manager

        cfg = get_config()
        mgr = OutputManager()

        # Terminal
        if cfg.output.terminal.enabled:
            mgr.register(TerminalOutputChannel(
                color=cfg.output.terminal.color,
                verbosity=cfg.output.terminal.verbosity,
                show_gesture=cfg.output.terminal.show_gesture,
                show_voice=cfg.output.terminal.show_voice,
                show_intent=cfg.output.terminal.show_intent,
                show_command=cfg.output.terminal.show_command,
            ))

        # TTS
        if cfg.output.tts.enabled:
            mgr.register(TTSOutputChannel(
                provider=cfg.output.tts.provider,
                voice=cfg.output.tts.voice,
                speak_on=cfg.output.tts.speak_on,
            ))

        # Notification
        if cfg.output.notification.enabled:
            mgr.register(NotificationOutputChannel(
                provider=cfg.output.notification.provider,
                on=cfg.output.notification.on,
            ))

        # Sound
        if cfg.output.sound.enabled:
            mgr.register(SoundOutputChannel(on=cfg.output.sound.on))

        _manager = mgr
        return mgr


def reset_output_manager() -> None:
    """Reset global output manager (untuk testing)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.stop()
        _manager = None