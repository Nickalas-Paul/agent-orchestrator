"""In-memory token usage and cost tracking per agent/session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Pricing keyed by model family substring -> (input_per_1k, output_per_1k) USD.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "anthropic.claude-sonnet": (0.003, 0.015),
    "anthropic.claude-haiku": (0.00025, 0.00125),
    "meta.llama3": (0.00035, 0.0004),
}

DEFAULT_INPUT_COST_PER_1K = 0.001
DEFAULT_OUTPUT_COST_PER_1K = 0.002


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a model call using known or fallback rates.

    Args:
        model_id: Provider model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    input_rate, output_rate = DEFAULT_INPUT_COST_PER_1K, DEFAULT_OUTPUT_COST_PER_1K
    normalized = model_id.lower()
    for key, rates in MODEL_PRICING.items():
        if key in normalized:
            input_rate, output_rate = rates
            break
    return (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate


@dataclass
class MetricsTracker:
    """Tracks LLM invocations and aggregates cost/token metrics in memory."""

    _records: list[dict[str, Any]] = field(default_factory=list)

    def track_llm_call(
        self,
        agent_name: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        session_id: str,
    ) -> dict[str, Any]:
        """Record a single LLM invocation and return the stored record.

        Args:
            agent_name: Name of the calling agent.
            model_id: Model identifier used for the call.
            input_tokens: Prompt token count.
            output_tokens: Completion token count.
            latency_ms: End-to-end latency in milliseconds.
            session_id: Session correlation id.

        Returns:
            The persisted metrics record.
        """
        cost = estimate_cost(model_id, input_tokens, output_tokens)
        record: dict[str, Any] = {
            "agent_name": agent_name,
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "session_id": session_id,
            "estimated_cost_usd": cost,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._records.append(record)
        return record

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Return aggregate metrics for a session, including per-agent breakdown.

        Args:
            session_id: Session to summarize.

        Returns:
            Summary dict with totals and per-agent stats.
        """
        session_records = [r for r in self._records if r["session_id"] == session_id]
        return self._summarize(session_records, extra={"session_id": session_id})

    def get_agent_summary(self, agent_name: str) -> dict[str, Any]:
        """Return aggregate metrics for an agent across all sessions.

        Args:
            agent_name: Agent to summarize.

        Returns:
            Summary dict with totals across sessions.
        """
        agent_records = [r for r in self._records if r["agent_name"] == agent_name]
        return self._summarize(agent_records, extra={"agent_name": agent_name})

    def _summarize(
        self,
        records: list[dict[str, Any]],
        *,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a summary payload from a filtered list of records."""
        by_agent: dict[str, dict[str, Any]] = {}
        total_input = 0
        total_output = 0
        total_cost = 0.0
        total_latency = 0

        for record in records:
            total_input += int(record["input_tokens"])
            total_output += int(record["output_tokens"])
            total_cost += float(record["estimated_cost_usd"])
            total_latency += int(record["latency_ms"])

            agent = str(record["agent_name"])
            bucket = by_agent.setdefault(
                agent,
                {
                    "agent_name": agent,
                    "call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "total_latency_ms": 0,
                },
            )
            bucket["call_count"] += 1
            bucket["input_tokens"] += int(record["input_tokens"])
            bucket["output_tokens"] += int(record["output_tokens"])
            bucket["total_tokens"] += int(record["total_tokens"])
            bucket["estimated_cost_usd"] += float(record["estimated_cost_usd"])
            bucket["total_latency_ms"] += int(record["latency_ms"])

        summary: dict[str, Any] = {
            **extra,
            "call_count": len(records),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "estimated_cost_usd": total_cost,
            "total_latency_ms": total_latency,
            "per_agent": list(by_agent.values()),
        }
        return summary
