"""Tests for sensitivity-aware logging separation."""

from __future__ import annotations

import logging

import pytest
import structlog

from packages.core.logging import logger as logger_module
from packages.core.logging.logger import (
    SensitivityLevel,
    configure_logging,
    filter_by_sensitivity,
    get_logger,
    get_sensitive_logger,
)


@pytest.fixture(autouse=True)
def reset_logging_state() -> None:
    """Ensure each test starts with a clean logging configuration."""
    logger_module._CONFIGURED = False
    root = logging.getLogger()
    root.handlers.clear()
    yield
    logger_module._CONFIGURED = False
    root.handlers.clear()


def test_get_sensitive_logger_binds_sensitivity() -> None:
    configure_logging(log_level="INFO", log_format="json", force=True)
    log = get_sensitive_logger("test.sensitive")
    assert log._context.get("sensitivity") == SensitivityLevel.SENSITIVE.value


def test_sensitivity_level_enum_values() -> None:
    assert SensitivityLevel.OPERATIONAL.value == "operational"
    assert SensitivityLevel.SENSITIVE.value == "sensitive"


def test_configure_logging_includes_sensitivity_processor() -> None:
    configure_logging(log_level="INFO", log_format="json", force=True)
    processors = structlog.get_config()["processors"]
    assert filter_by_sensitivity in processors


def test_default_logger_has_no_sensitivity_tag() -> None:
    configure_logging(log_level="INFO", log_format="json", force=True)
    log = get_logger("test.operational")
    assert "sensitivity" not in getattr(log, "_context", {})
