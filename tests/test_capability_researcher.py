"""Tests for the vendor capability researcher agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_vendor.agents.capability_researcher import CapabilityResearcherAgent


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
                "max_tokens": max_tokens,
                "agent_name": agent_name,
                "session_id": session_id,
            }
        )
        return ModelResponse(
            content=self.content,
            model_id=model_id or "mock-model",
            token_usage=TokenUsage(
                input_tokens=40,
                output_tokens=60,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.01,
            ),
            latency_ms=3,
        )


def _capability_payload() -> dict:
    return {
        "vendor_name": "CloudScale Solutions",
        "capabilities": [
            {
                "capability_name": "SOC 2 Type II",
                "evidence": "The platform is SOC 2 Type II attested.",
                "confidence_score": 0.94,
                "source_citation": "vendor_alpha_profile.txt",
            },
            {
                "capability_name": "Multi-region compute",
                "evidence": "Production footprint spans three US regions.",
                "confidence_score": 0.9,
                "source_citation": "vendor_alpha_profile.txt",
            },
        ],
        "summary": "CloudScale offers governed multi-region infrastructure with SOC 2.",
    }


def test_capability_researcher_constructs_with_optional_retriever() -> None:
    provider = ScriptedModelProvider("{}")
    metrics = MetricsTracker()
    agent = CapabilityResearcherAgent(provider=provider, metrics=metrics)
    assert agent._rag_retriever is None

    retriever = MagicMock()
    with_rag = CapabilityResearcherAgent(
        provider=provider,
        metrics=metrics,
        rag_retriever=retriever,
    )
    assert with_rag._rag_retriever is retriever


@pytest.mark.asyncio
async def test_research_capabilities_returns_expected_keys() -> None:
    provider = ScriptedModelProvider(json.dumps(_capability_payload()))
    metrics = MetricsTracker()
    agent = CapabilityResearcherAgent(provider=provider, metrics=metrics)
    response = await agent.research_capabilities(
        vendor_name="CloudScale Solutions",
        vendor_document="SOC 2 Type II attested. Three US regions.",
        session_id="s-research",
    )
    assert response.agent_name == "capability_researcher"
    assert "capabilities" in response.output
    assert "summary" in response.output
    assert response.output["vendor_name"] == "CloudScale Solutions"
    assert len(response.output["capabilities"]) == 2
    assert provider.calls[0]["temperature"] == 0.3
    assert "none available" in str(provider.calls[0]["prompt"])


@pytest.mark.asyncio
async def test_research_capabilities_tracks_metrics() -> None:
    provider = ScriptedModelProvider(json.dumps(_capability_payload()))
    metrics = MetricsTracker()
    agent = CapabilityResearcherAgent(provider=provider, metrics=metrics)
    await agent.research_capabilities(
        vendor_name="CloudScale Solutions",
        vendor_document="certified",
        session_id="s-metrics",
    )
    summary = metrics.get_session_summary("s-metrics")
    assert summary["call_count"] == 1
    assert summary["input_tokens"] == 40
    assert summary["output_tokens"] == 60


@pytest.mark.asyncio
async def test_research_capabilities_skips_rag_when_retriever_is_none() -> None:
    provider = ScriptedModelProvider(json.dumps(_capability_payload()))
    agent = CapabilityResearcherAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.research_capabilities(
        vendor_name="CloudScale Solutions",
        vendor_document="doc",
        session_id="s-no-rag",
    )
    assert response.output["retrieved_chunks"] == []


@pytest.mark.asyncio
async def test_research_capabilities_calls_retrieve_formatted_with_vendor_domain() -> None:
    provider = ScriptedModelProvider(json.dumps(_capability_payload()))
    retriever = MagicMock()
    retriever.retrieve_formatted = AsyncMock(
        return_value="[Source: kb.txt, Page: 1, Relevance: 0.91]\nSOC 2 Type II"
    )
    agent = CapabilityResearcherAgent(
        provider=provider,
        metrics=MetricsTracker(),
        rag_retriever=retriever,
    )
    response = await agent.research_capabilities(
        vendor_name="CloudScale Solutions",
        vendor_document="doc",
        session_id="s-rag",
    )
    retriever.retrieve_formatted.assert_awaited_once_with(
        query="capabilities of CloudScale Solutions",
        domain="vendor",
        top_k=5,
    )
    prompt = str(provider.calls[0]["prompt"])
    assert "Additional context from knowledge base:" in prompt
    assert "SOC 2 Type II" in prompt
    assert response.output["retrieved_chunks"]
    assert "formatted_context" in response.output["retrieved_chunks"][0]
