"""Shared Pydantic schemas for agent I/O and orchestration messages."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a new UUID string for model identifiers."""
    return str(uuid4())


class TaskStatus(str, Enum):
    """Lifecycle status for a decomposed task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TokenUsage(BaseModel):
    """Token tracking for a single model invocation."""

    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""
    estimated_cost_usd: float = 0.0


class AgentMessage(BaseModel):
    """Universal message format exchanged between agents."""

    session_id: str
    source_agent: str
    target_agent: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


class TaskDefinition(BaseModel):
    """A single decomposed task within an execution plan."""

    task_id: str = Field(default_factory=_new_id)
    description: str
    agent_type: str
    dependencies: list[str] = Field(default_factory=list)
    priority: int = 0
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING


class ExecutionPlan(BaseModel):
    """Full DAG of tasks derived from an original user request."""

    plan_id: str = Field(default_factory=_new_id)
    session_id: str
    original_request: str
    tasks: list[TaskDefinition] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)


class AgentResponse(BaseModel):
    """Result returned by an agent after processing a task."""

    task_id: str
    agent_name: str
    output: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    processing_time_ms: int = 0
    created_at: datetime = Field(default_factory=_utc_now)


class BusinessMetrics(BaseModel):
    """Business alignment metrics for a pipeline run."""

    cost_per_interaction_usd: float = 0.0
    task_completion_status: str = "completed"  # "completed" | "pending_review" | "failed"
    processing_time_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    agent_call_count: int = 0
