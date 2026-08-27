"""Tests for the vendor market positioner agent."""

from __future__ import annotations

import json

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_vendor.agents.market_positioner import MarketPositionerAgent


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
                input_tokens=55,
                output_tokens=90,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.012,
            ),
            latency_ms=6,
        )


def _position_payload() -> dict:
    return {
        "vendor_name": "CloudScale Solutions",
        "reasoning": (
            "1. Segment is governed IaaS. 2. SOC 2 and reserved pricing distinguish it. "
            "3. Regional coverage is a risk versus hyperscalers."
        ),
        "market_segment": "governed mid-market IaaS",
        "differentiators": ["SOC 2 Type II", "predictable reserved tiers"],
        "competitive_advantages": ["thinner operational surface than DIY AWS"],
        "competitive_risks": ["limited APAC regions"],
        "strategic_assessment": "Viable alternative for US-residency buyers.",
        "confidence_score": 0.81,
    }


def test_market_positioner_constructs() -> None:
    agent = MarketPositionerAgent(provider=ScriptedModelProvider("{}"), metrics=MetricsTracker())
    assert agent._provider is not None
    assert agent._metrics is not None


@pytest.mark.asyncio
async def test_position_vendor_returns_market_keys() -> None:
    provider = ScriptedModelProvider(json.dumps(_position_payload()))
    agent = MarketPositionerAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.position_vendor(
        vendor_name="CloudScale Solutions",
        vendor_document="Governed hyperscale alternative.",
        capability_summary="SOC 2 and multi-region compute.",
        pricing_summary="Tiered Professional at $9,800/month.",
        session_id="s-position",
    )
    assert response.agent_name == "market_positioner"
    assert response.output["market_segment"]
    assert isinstance(response.output["differentiators"], list)
    assert isinstance(response.output["competitive_advantages"], list)
    assert isinstance(response.output["competitive_risks"], list)
    assert response.output["strategic_assessment"]
    assert response.output["reasoning"]
    assert provider.calls[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_position_vendor_includes_upstream_summaries_in_prompt() -> None:
    provider = ScriptedModelProvider(json.dumps(_position_payload()))
    agent = MarketPositionerAgent(provider=provider, metrics=MetricsTracker())
    await agent.position_vendor(
        vendor_name="CloudScale Solutions",
        vendor_document="doc",
        capability_summary="UNIQUE_CAPABILITY_SUMMARY",
        pricing_summary="UNIQUE_PRICING_SUMMARY",
        session_id="s-summaries",
    )
    prompt = str(provider.calls[0]["prompt"])
    assert "UNIQUE_CAPABILITY_SUMMARY" in prompt
    assert "UNIQUE_PRICING_SUMMARY" in prompt


@pytest.mark.asyncio
async def test_position_vendor_tracks_metrics() -> None:
    provider = ScriptedModelProvider(json.dumps(_position_payload()))
    metrics = MetricsTracker()
    agent = MarketPositionerAgent(provider=provider, metrics=metrics)
    await agent.position_vendor(
        vendor_name="CloudScale Solutions",
        vendor_document="doc",
        capability_summary="caps",
        pricing_summary="price",
        session_id="s-position-metrics",
    )
    summary = metrics.get_session_summary("s-position-metrics")
    assert summary["call_count"] == 1
    assert summary["input_tokens"] == 55
    assert summary["output_tokens"] == 90
