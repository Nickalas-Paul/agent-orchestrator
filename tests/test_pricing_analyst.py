"""Tests for the vendor pricing analyst agent."""

from __future__ import annotations

import json

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_vendor.agents.pricing_analyst import PricingAnalystAgent


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
                input_tokens=30,
                output_tokens=50,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.008,
            ),
            latency_ms=2,
        )


def _pricing_payload() -> dict:
    return {
        "vendor_name": "CloudScale Solutions",
        "pricing_model": "tiered",
        "estimated_annual_cost": "$117,600 Professional list",
        "cost_breakdown": [
            {"item": "Professional plan", "cost": "$9,800/month", "notes": "128 vCPU"},
            {"item": "Egress overage", "cost": "$0.08/GB", "notes": "after 2 TB/month"},
        ],
        "risk_flags": ["GPU capacity is allocation-based"],
        "comparison_notes": "Reserved enterprise quotes sit between $180k and $420k annually.",
        "confidence_score": 0.86,
    }


def test_pricing_analyst_constructs() -> None:
    agent = PricingAnalystAgent(provider=ScriptedModelProvider("{}"), metrics=MetricsTracker())
    assert agent._provider is not None
    assert agent._metrics is not None


@pytest.mark.asyncio
async def test_analyze_pricing_returns_pricing_keys() -> None:
    provider = ScriptedModelProvider(json.dumps(_pricing_payload()))
    agent = PricingAnalystAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.analyze_pricing(
        vendor_name="CloudScale Solutions",
        vendor_document="Professional is $9,800 per month.",
        session_id="s-price",
    )
    assert response.agent_name == "pricing_analyst"
    assert response.output["pricing_model"] == "tiered"
    assert response.output["estimated_annual_cost"]
    assert isinstance(response.output["cost_breakdown"], list)
    assert isinstance(response.output["risk_flags"], list)
    assert provider.calls[0]["temperature"] == 0.2
    assert 0.0 <= response.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_analyze_pricing_tracks_metrics() -> None:
    provider = ScriptedModelProvider(json.dumps(_pricing_payload()))
    metrics = MetricsTracker()
    agent = PricingAnalystAgent(provider=provider, metrics=metrics)
    await agent.analyze_pricing(
        vendor_name="CloudScale Solutions",
        vendor_document="doc",
        session_id="s-price-metrics",
    )
    summary = metrics.get_session_summary("s-price-metrics")
    assert summary["call_count"] == 1
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 50
