"""Structured JSON logging setup using structlog."""

from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from typing import Any

import structlog


_CONFIGURED = False
_SENSITIVE_LOGGER_NAME = "agent_orchestrator.sensitive"


class SensitivityLevel(str, Enum):
    """Data sensitivity classification for log routing."""

    OPERATIONAL = "operational"  # Agent invocations, latency, errors, counts
    SENSITIVE = "sensitive"  # Document content, PII detections, audit fallbacks


def filter_by_sensitivity(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Tag and optionally dual-route sensitive log events.

    When ``sensitivity == "sensitive"``:
    - Prefixes the event name with ``[SENSITIVE]`` for the default stream
    - Mirrors the event to the dedicated sensitive logger when handlers exist

    Absent or ``operational`` sensitivity passes through unchanged.
    """
    del logger, method_name  # unused; required by structlog processor signature
    sensitivity = event_dict.get("sensitivity")
    is_sensitive = sensitivity in {
        SensitivityLevel.SENSITIVE,
        SensitivityLevel.SENSITIVE.value,
        "sensitive",
    }
    if not is_sensitive:
        return event_dict

    event = event_dict.get("event", "")
    if isinstance(event, str) and not event.startswith("[SENSITIVE]"):
        event_dict["event"] = f"[SENSITIVE] {event}"

    sensitive_logger = logging.getLogger(_SENSITIVE_LOGGER_NAME)
    if sensitive_logger.handlers:
        sensitive_logger.log(
            getattr(logging, str(event_dict.get("level", "info")).upper(), logging.INFO),
            "%s",
            event_dict.get("event"),
            extra={"sensitivity": SensitivityLevel.SENSITIVE.value},
        )
    return event_dict


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
        filter_by_sensitivity,
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

    # Optional dedicated sensitive sink (stderr by default for separation).
    sensitive_logger = logging.getLogger(_SENSITIVE_LOGGER_NAME)
    sensitive_logger.handlers.clear()
    sensitive_logger.propagate = False
    sensitive_handler = logging.StreamHandler(sys.stderr)
    sensitive_handler.setFormatter(formatter)
    sensitive_logger.addHandler(sensitive_handler)
    sensitive_logger.setLevel(level)

    sensitive_log_file = os.getenv("SENSITIVE_LOG_FILE")
    if sensitive_log_file:
        file_handler = logging.FileHandler(sensitive_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        sensitive_logger.addHandler(file_handler)

    _CONFIGURED = True


def _resolve_log_level(level_name: str) -> int:
    """Map a level name to a standard logging level integer."""
    return getattr(logging, level_name.upper(), logging.INFO)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for ``name``.

    Example::

        logger = get_logger("orchestrator")
        logger.info("task_decomposed", task_count=3, session_id="abc")
    """
    configure_logging()
    return structlog.get_logger(name)


def get_sensitive_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger pre-bound with ``sensitivity="sensitive"``."""
    return get_logger(name).bind(sensitivity=SensitivityLevel.SENSITIVE.value)
