"""Terminal output channel dengan Rich formatting."""
from __future__ import annotations

from typing import Any

from nexus.core.logging import get_logger
from nexus.output.channels.base import (
    OutputChannel,
    OutputChannelType,
    OutputMessage,
    OutputPriority,
)


log = get_logger(__name__)


# ANSI color codes (fallback jika Rich tidak tersedia)
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"

COLORS = {
    "vision": "\033[36m",       # cyan
    "voice": "\033[35m",        # magenta
    "brain": "\033[33m",        # yellow
    "action": "\033[32m",       # green
    "system": "\033[37m",       # white
    "error": "\033[31m",        # red
    "warning": "\033[33m",      # yellow
    "success": "\033[32m",      # green
    "debug": "\033[90m",        # gray
}


class TerminalOutputChannel(OutputChannel):
    """Print output ke terminal dengan color coding."""

    name = "terminal"

    def __init__(
        self,
        *,
        color: bool = True,
        verbosity: str = "normal",
        show_gesture: bool = True,
        show_voice: bool = True,
        show_intent: bool = True,
        show_command: bool = True,
    ):
        self._color = color
        self._verbosity = verbosity
        self._show_gesture = show_gesture
        self._show_voice = show_voice
        self._show_intent = show_intent
        self._show_command = show_command
        self._enabled = True

    def start(self) -> None:
        log.debug("output.terminal.start", color=self._color, verbosity=self._verbosity)

    def output(self, message: OutputMessage) -> None:
        # Filter by verbosity
        min_priority = self._min_priority()
        if not message.should_display(min_priority):
            return

        # Filter by event type
        if not self._should_show_event(message.event_type):
            return

        # Format
        text = self._format(message)
        print(text, flush=True)

    def stop(self) -> None:
        pass

    def _min_priority(self) -> OutputPriority:
        if self._verbosity == "quiet":
            return OutputPriority.WARNING
        if self._verbosity == "verbose":
            return OutputPriority.DEBUG
        return OutputPriority.INFO

    def _should_show_event(self, event_type: str | None) -> bool:
        if event_type is None:
            return True
        if event_type.startswith("gesture."):
            return self._show_gesture
        if event_type.startswith("voice."):
            return self._show_voice
        if event_type.startswith("intent."):
            return self._show_intent
        if event_type.startswith("command."):
            return self._show_command
        return True

    def _format(self, message: OutputMessage) -> str:
        source = message.source
        prefix = self._colorize(f"[{source:6s}]", source)
        text = message.text

        if message.priority == OutputPriority.ERROR:
            text = self._colorize(text, "error")
        elif message.priority == OutputPriority.WARNING:
            text = self._colorize(text, "warning")

        return f"{prefix} {text}"

    def _colorize(self, text: str, color_key: str) -> str:
        if not self._color:
            return text
        color = COLORS.get(color_key, "")
        if not color:
            return text
        return f"{color}{text}{ANSI_RESET}"