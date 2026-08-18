"""Tests for in-memory metrics tracking and cost estimation."""

from __future__ import annotations

from packages.core.metrics.tracker import MetricsTracker, estimate_cost


def test_track_llm_call_records_data_correctly() -> None:
    tracker = MetricsTracker()
    record = tracker.track_llm_call(
        agent_name="writer",
        model_id="anthropic.claude-sonnet-4",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=120,
        session_id="sess-a",
    )

    assert record["agent_name"] == "writer"
    assert record["model_id"] == "anthropic.claude-sonnet-4"
    assert record["input_tokens"] == 1000
    assert record["output_tokens"] == 500
    assert record["total_tokens"] == 1500
    assert record["latency_ms"] == 120
    assert record["session_id"] == "sess-a"
    assert record["estimated_cost_usd"] == pytest_approx_cost(1000, 500, 0.003, 0.015)


def test_get_session_summary_aggregates_across_calls() -> None:
    tracker = MetricsTracker()
    tracker.track_llm_call("a", "meta.llama3-8b", 1000, 1000, 10, "s1")
    tracker.track_llm_call("b", "meta.llama3-8b", 2000, 500, 20, "s1")
    tracker.track_llm_call("a", "meta.llama3-8b", 1000, 1000, 10, "s2")

    summary = tracker.get_session_summary("s1")
    assert summary["call_count"] == 2
    assert summary["input_tokens"] == 3000
    assert summary["output_tokens"] == 1500
    assert summary["total_tokens"] == 4500
    assert len(summary["per_agent"]) == 2


def test_get_agent_summary_aggregates_across_sessions() -> None:
    tracker = MetricsTracker()
    tracker.track_llm_call("researcher", "anthropic.claude-haiku", 1000, 1000, 5, "s1")
    tracker.track_llm_call("researcher", "anthropic.claude-haiku", 1000, 1000, 7, "s2")
    tracker.track_llm_call("writer", "anthropic.claude-haiku", 1000, 1000, 9, "s2")

    summary = tracker.get_agent_summary("researcher")
    assert summary["call_count"] == 2
    assert summary["input_tokens"] == 2000
    assert summary["output_tokens"] == 2000
    assert summary["total_latency_ms"] == 12


def test_cost_calculation_known_and_unknown_models() -> None:
    known = estimate_cost("anthropic.claude-sonnet-4-20250514", 1000, 1000)
    assert known == pytest_approx_cost(1000, 1000, 0.003, 0.015)

    haiku = estimate_cost("anthropic.claude-haiku-v1", 2000, 1000)
    assert haiku == pytest_approx_cost(2000, 1000, 0.00025, 0.00125)

    llama = estimate_cost("meta.llama3-70b", 1000, 1000)
    assert llama == pytest_approx_cost(1000, 1000, 0.00035, 0.0004)

    unknown = estimate_cost("vendor.unknown-model", 1000, 1000)
    assert unknown == pytest_approx_cost(1000, 1000, 0.001, 0.002)


def pytest_approx_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate: float,
    output_rate: float,
) -> float:
    return (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate
