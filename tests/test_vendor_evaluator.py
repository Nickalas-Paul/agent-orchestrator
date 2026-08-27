"""Tests for the vendor output evaluator agent."""

from __future__ import annotations

import json

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_vendor.agents.vendor_evaluator import VendorOutputEvaluator


class ScriptedModelProvider(ModelProvider):
    """Returns a preconfigured content string for a single invoke."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    async def invoke(
        self,
        prompt: str,
        model_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
        *,
        agent_name: str = "unknown",
        session_id: str = "unknown",
    ) -> ModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "agent_name": agent_name,
                "session_id": session_id,
            }
        )
        return ModelResponse(
            content=self.content,
            model_id=model_id or "mock-model",
            token_usage=TokenUsage(
                input_tokens=20,
                output_tokens=40,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.005,
            ),
            latency_ms=4,
        )


def _evaluation_payload(*, overall_confidence: float = 0.9) -> dict:
    return {
        "overall_confidence": overall_confidence,
        "findings": [
            "Capability evidence is cited.",
            "Pricing figures match the source document.",
        ],
        "contradictions": [],
        "summary": "Specialist outputs are consistent and complete.",
        "recommendation": "approve" if overall_confidence >= 0.85 else "flag_for_review",
    }


def test_vendor_evaluator_constructs_with_optional_criteria() -> None:
    default_agent = VendorOutputEvaluator(
        provider=ScriptedModelProvider("{}"),
        metrics=MetricsTracker(),
    )
    assert default_agent._criteria

    custom = VendorOutputEvaluator(
        provider=ScriptedModelProvider("{}"),
        metrics=MetricsTracker(),
        evaluation_criteria=["completeness"],
    )
    assert custom._criteria == ["completeness"]


@pytest.mark.asyncio
async def test_evaluate_outputs_returns_judgment_keys() -> None:
    provider = ScriptedModelProvider(json.dumps(_evaluation_payload()))
    agent = VendorOutputEvaluator(provider=provider, metrics=MetricsTracker())
    response = await agent.evaluate_outputs(
        capability_result={"capabilities": [{"capability_name": "SOC 2"}]},
        pricing_result={"pricing_model": "tiered"},
        market_result={"market_segment": "IaaS"},
        session_id="s-eval",
    )
    assert response.agent_name == "vendor_evaluator"
    assert "overall_confidence" in response.output
    assert isinstance(response.output["findings"], list)
    assert response.output["recommendation"] in {"approve", "flag_for_review"}
    assert provider.calls[0]["temperature"] == 0.2
    assert 0.0 <= response.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_evaluate_outputs_tracks_metrics() -> None:
    provider = ScriptedModelProvider(json.dumps(_evaluation_payload()))
    metrics = MetricsTracker()
    agent = VendorOutputEvaluator(provider=provider, metrics=metrics)
    await agent.evaluate_outputs(
        capability_result={},
        pricing_result={},
        market_result={},
        session_id="s-eval-metrics",
    )
    summary = metrics.get_session_summary("s-eval-metrics")
    assert summary["call_count"] == 1
    assert summary["input_tokens"] == 20
    assert summary["output_tokens"] == 40
