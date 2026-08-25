"""Insert-only audit log writer backed by Postgres with structlog fallback."""

from __future__ import annotations

import json
from typing import Any

from packages.core.audit.models import AuditEntry
from packages.core.database.connection import DatabasePool
from packages.core.logging import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """Insert-only audit log writer. Writes to agent_audit_logs table in Postgres.

    Falls back to structlog JSON logging if database is unavailable,
    so audit data is never silently lost.
    """

    def log(self, entry: AuditEntry) -> None:
        """Write a single audit entry to Postgres. Never raises — logs errors instead."""
        try:
            with DatabasePool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO agent_audit_logs
                            (session_id, agent_name, action_type, input_payload, output_payload,
                             confidence_score, hitl_status, hitl_reviewer, token_count, cost_usd, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            entry.session_id,
                            entry.agent_name,
                            entry.action_type.value,
                            json.dumps(entry.input_payload, default=str),
                            json.dumps(entry.output_payload, default=str),
                            entry.confidence_score,
                            entry.hitl_status.value if entry.hitl_status else None,
                            entry.hitl_reviewer,
                            entry.token_count,
                            entry.cost_usd,
                            entry.created_at,
                        ),
                    )
            logger.info(
                "audit_entry_written",
                session_id=entry.session_id,
                agent_name=entry.agent_name,
                action_type=entry.action_type.value,
            )
        except Exception as exc:  # noqa: BLE001 - never lose audit data
            logger.error(
                "audit_write_failed",
                error=str(exc),
                session_id=entry.session_id,
                agent_name=entry.agent_name,
                action_type=entry.action_type.value,
                fallback_payload=entry.model_dump(mode="json"),
            )

    def query_session(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all audit entries for a session. For HITL review and debugging."""
        with DatabasePool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_audit_logs WHERE session_id = %s ORDER BY created_at ASC",
                    (session_id,),
                )
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
