"""Desktop notification channel — Linux notify-send."""
from __future__ import annotations

import shutil
import subprocess

from nexus.core.logging import get_logger
from nexus.output.channels.base import (
    OutputChannel,
    OutputChannelType,
    OutputMessage,
    OutputPriority,
)


log = get_logger(__name__)


class NotificationOutputChannel(OutputChannel):
    """Desktop notification via notify-send (Linux)."""

    name = "notification"

    def __init__(
        self,
        *,
        provider: str = "notify-send",
        on: list[str] | None = None,
    ):
        self._provider = provider
        self._on = set(on or ["error", "warning"])
        self._available = shutil.which("notify-send") is not None
        self._enabled = True

    def start(self) -> None:
        log.info(
            "output.notification.start",
            provider=self._provider,
            available=self._available,
            on=list(self._on),
        )

    def output(self, message: OutputMessage) -> None:
        if not self._available:
            return
        if message.channel_type != OutputChannelType.NOTIFY:
            return

        # Only notify for configured priorities
        priority_key = message.priority.value
        if priority_key not in self._on:
            return

        urgency = "normal"
        if message.priority == OutputPriority.ERROR:
            urgency = "critical"
        elif message.priority == OutputPriority.WARNING:
            urgency = "normal"

        title = f"NEXUS — {message.source}"
        try:
            subprocess.run(
                [
                    "notify-send",
                    "-u", urgency,
                    "-a", "NEXUS",
                    title,
                    message.text,
                ],
                check=False,
                timeout=2,
                capture_output=True,
            )
        except Exception:
            log.exception("output.notification.failed")

    def stop(self) -> None:
        pass


from typing import Any  # noqa: E402