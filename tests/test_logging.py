"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging

import pytest

from packages.core.logging import logger as logger_module
from packages.core.logging.logger import configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_logging_state() -> None:
    """Ensure each test starts with a clean logging configuration."""
    logger_module._CONFIGURED = False
    root = logging.getLogger()
    root.handlers.clear()
    yield
    logger_module._CONFIGURED = False
    root.handlers.clear()


def test_get_logger_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(log_level="INFO", log_format="json", force=True)
    log = get_logger("test.logging")
    log.info("hello_event", task_count=3)

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "Expected JSON log output on stdout"
    payload = json.loads(captured[-1])
    assert payload["event"] == "hello_event"
    assert payload["task_count"] == 3
    assert payload["level"] == "info"
    assert payload["logger"] == "test.logging"
    assert "timestamp" in payload


def test_bound_context_appears_in_log_output(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(log_level="INFO", log_format="json", force=True)
    log = get_logger("test.bound").bind(session_id="abc-123", agent="orchestrator")
    log.info("task_decomposed", task_count=2)

    captured = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(captured[-1])
    assert payload["session_id"] == "abc-123"
    assert payload["agent"] == "orchestrator"
    assert payload["task_count"] == 2
    assert payload["event"] == "task_decomposed"
