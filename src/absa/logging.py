"""Structured logging setup (structlog over the stdlib).

Call :func:`configure_logging` once at process start (API, CLI, demo). Use
:func:`get_logger` everywhere else. Console renderer by default; JSON when
``ABSA_LOG_JSON=true`` (recommended for the serving process).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from absa.config import get_settings


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Configure structlog + stdlib logging.

    Args:
        level: Log level name; defaults to ``ABSA_LOG_LEVEL``.
        json_logs: Force JSON (True) or console (False) output; defaults to
            ``ABSA_LOG_JSON``.
    """
    settings = get_settings()
    level_name = (level or settings.log_level).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)
    use_json = settings.log_json if json_logs is None else json_logs

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
