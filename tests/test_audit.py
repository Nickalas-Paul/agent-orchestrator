"""Tests for audit models and the insert-only audit logger."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry, HITLStatus
from packages.core.database.connection import DatabasePool


def test_audit_entry_model_validation() -> None:
    entry = AuditEntry(
        session_id="sess-1",
        agent_name="evaluator",
        action_type=ActionType.EVALUATION,
        input_payload={"ok": True},
        output_payload={"score": 0.9},
        confidence_score=0.9,
        hitl_status=HITLStatus.PENDING_REVIEW,
        token_count=10,
        cost_usd=0.01,
    )
    assert entry.action_type == ActionType.EVALUATION
    assert entry.hitl_status == HITLStatus.PENDING_REVIEW

    with pytest.raises(ValidationError):
        AuditEntry(
            session_id="sess-1",
            agent_name="evaluator",
            action_type="not_a_real_type",  # type: ignore[arg-type]
        )


def test_audit_logger_fallback_on_db_failure(capsys: pytest.CaptureFixture[str]) -> None:
    from packages.core.logging.logger import configure_logging

    configure_logging(log_level="ERROR", log_format="json", force=True)
    audit_logger = AuditLogger()
    entry = AuditEntry(
        session_id="sess-fail",
        agent_name="gap_analyzer",
        action_type=ActionType.INVOKE,
    )
    with patch.object(DatabasePool, "get_connection", side_effect=RuntimeError("db down")):
        audit_logger.log(entry)

    captured = capsys.readouterr().out
    assert "audit_write_failed" in captured
    assert "sess-fail" in captured
    assert "fallback_payload" in captured


def test_audit_query_session() -> None:
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = [("session_id",), ("agent_name",), ("action_type",)]
    mock_cursor.fetchall.return_value = [
        ("sess-1", "evaluator", "evaluation"),
        ("sess-1", "hitl_gate", "hitl_decision"),
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch.object(DatabasePool, "get_connection") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_conn
        rows = AuditLogger().query_session("sess-1")

    assert len(rows) == 2
    assert rows[0]["agent_name"] == "evaluator"
    assert rows[1]["action_type"] == "hitl_decision"
    mock_cursor.execute.assert_called_once()
    assert mock_cursor.execute.call_args[0][1] == ("sess-1",)
