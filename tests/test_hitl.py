"""Tests for HITL models and job manager review workflow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from packages.core.audit.models import ActionType, HITLStatus
from packages.core.hitl.manager import HITLManager
from packages.core.hitl.models import HITLReview, JobStatus, PipelineJob, ReviewDecision


def test_pipeline_job_model_defaults() -> None:
    job = PipelineJob(session_id="sess-1")
    assert job.status == JobStatus.RUNNING
    assert job.pipeline_name == "rfp_analysis"
    assert len(job.job_id) == 12
    assert job.pipeline_inputs == {}
    assert job.pipeline_outputs == {}
    assert job.reviewer is None
    assert job.review_decision is None


def test_hitl_review_model_validation() -> None:
    review = HITLReview(
        job_id="abc123def456",
        reviewer="alex",
        decision=ReviewDecision.APPROVE,
    )
    assert review.notes is None

    with pytest.raises(ValidationError):
        HITLReview(job_id="abc123def456", reviewer="alex")  # type: ignore[call-arg]


def test_submit_review_updates_job_status() -> None:
    audit = MagicMock()
    manager = HITLManager(audit_logger=audit)
    job = PipelineJob(
        job_id="abc123def456",
        session_id="sess-1",
        status=JobStatus.PENDING_REVIEW,
        evaluator_confidence=0.72,
    )
    review = HITLReview(
        job_id=job.job_id,
        reviewer="alex",
        decision=ReviewDecision.APPROVE,
        notes="looks good",
    )
    with (
        patch.object(manager, "get_job", return_value=job),
        patch.object(manager, "save_job", side_effect=lambda saved: saved) as save,
    ):
        result = manager.submit_review(review)

    assert result.status == JobStatus.APPROVED
    assert result.reviewer == "alex"
    assert result.review_decision == ReviewDecision.APPROVE
    assert result.review_notes == "looks good"
    save.assert_called_once()
    audit.log.assert_called_once()
    entry = audit.log.call_args[0][0]
    assert entry.action_type == ActionType.HITL_DECISION
    assert entry.hitl_status == HITLStatus.APPROVED
    assert entry.hitl_reviewer == "alex"


def test_submit_review_rejects_non_pending_job() -> None:
    manager = HITLManager(audit_logger=MagicMock())
    job = PipelineJob(
        job_id="abc123def456",
        session_id="sess-1",
        status=JobStatus.COMPLETED,
    )
    review = HITLReview(
        job_id=job.job_id,
        reviewer="alex",
        decision=ReviewDecision.REJECT,
    )
    with patch.object(manager, "get_job", return_value=job):
        with pytest.raises(ValueError, match="not pending_review"):
            manager.submit_review(review)
