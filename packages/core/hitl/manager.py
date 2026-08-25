"""Persistence and review workflow for human-in-the-loop pipeline jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry, HITLStatus
from packages.core.database.connection import DatabasePool
from packages.core.hitl.models import HITLReview, JobStatus, PipelineJob, ReviewDecision
from packages.core.logging import get_logger

logger = get_logger(__name__)


def _parse_json_field(value: Any) -> dict[str, Any]:
    """Coerce a JSONB column value into a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value) if value else {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _job_from_row(data: dict[str, Any]) -> PipelineJob:
    """Build a ``PipelineJob`` from a database row dict."""
    payload = dict(data)
    payload["pipeline_inputs"] = _parse_json_field(payload.get("pipeline_inputs"))
    payload["pipeline_outputs"] = _parse_json_field(payload.get("pipeline_outputs"))
    payload["evaluator_reasoning"] = _parse_json_field(payload.get("evaluator_reasoning"))
    return PipelineJob.model_validate(payload)


class HITLManager:
    """Manages pipeline job state for human-in-the-loop review workflows."""

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        """Initialize the manager.

        Args:
            audit_logger: Optional audit logger; a new instance is created when omitted.
        """
        self._audit = audit_logger or AuditLogger()

    def save_job(self, job: PipelineJob) -> PipelineJob:
        """Persist a pipeline job to Postgres."""
        with DatabasePool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_jobs
                        (job_id, session_id, pipeline_name, status, pipeline_inputs, pipeline_outputs,
                         evaluator_reasoning, evaluator_confidence, reviewer, review_decision, review_notes,
                         created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        pipeline_outputs = EXCLUDED.pipeline_outputs,
                        evaluator_reasoning = EXCLUDED.evaluator_reasoning,
                        evaluator_confidence = EXCLUDED.evaluator_confidence,
                        reviewer = EXCLUDED.reviewer,
                        review_decision = EXCLUDED.review_decision,
                        review_notes = EXCLUDED.review_notes,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        job.job_id,
                        job.session_id,
                        job.pipeline_name,
                        job.status.value,
                        json.dumps(job.pipeline_inputs, default=str),
                        json.dumps(job.pipeline_outputs, default=str),
                        json.dumps(job.evaluator_reasoning, default=str),
                        job.evaluator_confidence,
                        job.reviewer,
                        job.review_decision.value if job.review_decision else None,
                        job.review_notes,
                        job.created_at,
                        job.updated_at,
                    ),
                )
        logger.info("pipeline_job_saved", job_id=job.job_id, status=job.status.value)
        return job

    def get_job(self, job_id: str) -> PipelineJob | None:
        """Load a pipeline job by ID."""
        with DatabasePool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM pipeline_jobs WHERE job_id = %s", (job_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                columns = [desc[0] for desc in cur.description]
                return _job_from_row(dict(zip(columns, row)))

    def submit_review(self, review: HITLReview) -> PipelineJob:
        """Process a human review decision. Updates job state and writes audit entry."""
        job = self.get_job(review.job_id)
        if job is None:
            raise ValueError(f"Job {review.job_id} not found")
        if job.status != JobStatus.PENDING_REVIEW:
            raise ValueError(f"Job {review.job_id} is {job.status.value}, not pending_review")

        new_status = (
            JobStatus.APPROVED
            if review.decision == ReviewDecision.APPROVE
            else JobStatus.REJECTED
        )
        job.status = new_status
        job.reviewer = review.reviewer
        job.review_decision = review.decision
        job.review_notes = review.notes
        job.updated_at = datetime.now(timezone.utc)
        self.save_job(job)

        self._audit.log(
            AuditEntry(
                session_id=job.session_id,
                agent_name="hitl_gate",
                action_type=ActionType.HITL_DECISION,
                input_payload={
                    "job_id": job.job_id,
                    "evaluator_confidence": job.evaluator_confidence,
                },
                output_payload={
                    "decision": review.decision.value,
                    "reviewer": review.reviewer,
                    "notes": review.notes,
                },
                confidence_score=job.evaluator_confidence,
                hitl_status=(
                    HITLStatus.APPROVED
                    if review.decision == ReviewDecision.APPROVE
                    else HITLStatus.REJECTED
                ),
                hitl_reviewer=review.reviewer,
            )
        )
        logger.info(
            "hitl_review_submitted",
            job_id=job.job_id,
            decision=review.decision.value,
            reviewer=review.reviewer,
        )
        return job

    def list_pending_jobs(self) -> list[PipelineJob]:
        """List all jobs awaiting human review."""
        with DatabasePool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM pipeline_jobs WHERE status = %s ORDER BY created_at ASC",
                    (JobStatus.PENDING_REVIEW.value,),
                )
                columns = [desc[0] for desc in cur.description]
                return [_job_from_row(dict(zip(columns, row))) for row in cur.fetchall()]
