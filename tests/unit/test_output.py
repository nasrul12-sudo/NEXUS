"""Test output channels."""
import pytest

from nexus.output.channels.base import (
    OutputChannelType,
    OutputMessage,
    OutputPriority,
)
from nexus.output.channels.terminal import TerminalOutputChannel


def test_output_message_priority():
    msg = OutputMessage(
        text="test",
        channel_type=OutputChannelType.TEXT,
        priority=OutputPriority.WARNING,
    )
    assert msg.should_display(OutputPriority.INFO)
    assert msg.should_display(OutputPriority.WARNING)
    assert not msg.should_display(OutputPriority.ERROR)


def test_terminal_channel(capsys):
    ch = TerminalOutputChannel(color=False)
    ch.start()
    msg = OutputMessage(
        text="hello",
        channel_type=OutputChannelType.TEXT,
        source="system",
        priority=OutputPriority.INFO,
    )
    ch.output(msg)
    captured = capsys.readouterr()
    assert "hello" in captured.out
    ch.stop()