"""Pydantic models for the insert-only audit log."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Kinds of auditable actions written to ``agent_audit_logs``."""

    INVOKE = "invoke"
    TOOL_CALL = "tool_call"
    EVALUATION = "evaluation"
    HITL_DECISION = "hitl_decision"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"


class HITLStatus(str, Enum):
    """Human-review state recorded on an audit entry."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class AuditEntry(BaseModel):
    """Single audit log entry. Insert-only — never updated or deleted."""

    session_id: str
    agent_name: str
    action_type: ActionType
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float | None = None
    hitl_status: HITLStatus | None = None
    hitl_reviewer: str | None = None
    token_count: int = 0
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
