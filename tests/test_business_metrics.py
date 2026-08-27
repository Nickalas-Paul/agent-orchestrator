"""Tests for the BusinessMetrics model."""

from __future__ import annotations

from packages.core.types import BusinessMetrics
from packages.core.types.schemas import BusinessMetrics as SchemaBusinessMetrics


def test_business_metrics_defaults() -> None:
    metrics = BusinessMetrics()
    assert metrics.cost_per_interaction_usd == 0.0
    assert metrics.task_completion_status == "completed"
    assert metrics.processing_time_ms == 0
    assert metrics.total_input_tokens == 0
    assert metrics.total_output_tokens == 0
    assert metrics.agent_call_count == 0


def test_business_metrics_explicit_values() -> None:
    metrics = BusinessMetrics(
        cost_per_interaction_usd=0.042,
        task_completion_status="pending_review",
        processing_time_ms=1500,
        total_input_tokens=800,
        total_output_tokens=200,
        agent_call_count=4,
    )
    assert metrics.cost_per_interaction_usd == 0.042
    assert metrics.task_completion_status == "pending_review"
    assert metrics.processing_time_ms == 1500
    assert metrics.total_input_tokens == 800
    assert metrics.total_output_tokens == 200
    assert metrics.agent_call_count == 4


def test_business_metrics_model_dump_keys() -> None:
    dumped = BusinessMetrics(
        cost_per_interaction_usd=0.01,
        task_completion_status="failed",
        processing_time_ms=12,
        total_input_tokens=10,
        total_output_tokens=5,
        agent_call_count=1,
    ).model_dump()
    assert dumped == {
        "cost_per_interaction_usd": 0.01,
        "task_completion_status": "failed",
        "processing_time_ms": 12,
        "total_input_tokens": 10,
        "total_output_tokens": 5,
        "agent_call_count": 1,
    }


def test_business_metrics_importable_from_core_types() -> None:
    assert BusinessMetrics is SchemaBusinessMetrics
