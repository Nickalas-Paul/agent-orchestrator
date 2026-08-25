"""Pydantic models for HITL pipeline job state and review decisions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle status for a persisted pipeline job."""

    RUNNING = "running"
    COMPLETED = "completed"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class ReviewDecision(str, Enum):
    """Human decision submitted through the HITL gate."""

    APPROVE = "approve"
    REJECT = "reject"


class PipelineJob(BaseModel):
    """Tracks the state of a pipeline execution across HITL pause/resume."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str
    pipeline_name: str = "rfp_analysis"
    status: JobStatus = JobStatus.RUNNING
    pipeline_inputs: dict[str, Any] = Field(default_factory=dict)
    pipeline_outputs: dict[str, Any] = Field(default_factory=dict)
    evaluator_reasoning: dict[str, Any] = Field(default_factory=dict)
    evaluator_confidence: float | None = None
    reviewer: str | None = None
    review_decision: ReviewDecision | None = None
    review_notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HITLReview(BaseModel):
    """A human review decision submitted to the HITL gate."""

    job_id: str
    reviewer: str
    decision: ReviewDecision
    notes: str | None = None
