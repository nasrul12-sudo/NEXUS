"""Structured logging setup dengan structlog."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from nexus.core.config import get_config


_configured = False


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
    process_name: str | None = None,
) -> None:
    """Setup structlog + stdlib logging. Idempotent."""
    global _configured
    if _configured:
        return

    cfg = get_config()
    level = level or cfg.nexus.log_level
    fmt = fmt or cfg.nexus.log_format

    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if process_name:
        shared_processors.insert(
            0, lambda _, __, event_dict: {**event_dict, "process": process_name}
        )

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # stdlib logging → structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Silence noisy libs
    for noisy in ("asyncio", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a bound logger with initial context."""
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger