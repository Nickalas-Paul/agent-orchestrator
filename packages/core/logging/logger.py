"""Structured JSON logging setup using structlog."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


_CONFIGURED = False


def _resolve_log_level(level_name: str) -> int:
    """Map a level name to a standard logging level integer."""
    return getattr(logging, level_name.upper(), logging.INFO)


def configure_logging(
    *,
    log_level: str | None = None,
    log_format: str | None = None,
    force: bool = False,
) -> None:
    """Configure structlog and stdlib logging once for the process.

    Args:
        log_level: Override for ``LOG_LEVEL`` (default INFO).
        log_format: ``json`` or ``console``. Defaults to ``LOG_FORMAT``.
        force: Reconfigure even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    format_name = (log_format or os.getenv("LOG_FORMAT", "json")).lower()
    level = _resolve_log_level(level_name)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if format_name == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for ``name``.

    Example::

        logger = get_logger("orchestrator")
        logger.info("task_decomposed", task_count=3, session_id="abc")
    """
    configure_logging()
    return structlog.get_logger(name)
