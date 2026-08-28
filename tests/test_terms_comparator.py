"""Tests for the terms comparator agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_contract.agents.terms_comparator import TermsComparatorAgent


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
                input_tokens=35,
                output_tokens=55,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.009,
            ),
            latency_ms=5,
        )


def _comparison_payload() -> dict:
    return {
        "deviations": [
            {
                "term_name": "liability_cap",
                "standard_language": "24 months fees; unlimited for IP",
                "contract_language": "12 months fees aggregate",
                "deviation_type": "less_favorable",
                "impact_assessment": "Lower cap increases buyer residual risk.",
                "confidence_score": 0.92,
            },
            {
                "term_name": "termination_notice",
                "standard_language": "60 days",
                "contract_language": "30 days",
                "deviation_type": "less_favorable",
                "impact_assessment": "Shorter notice reduces transition runway.",
                "confidence_score": 0.88,
            },
            {
                "term_name": "governing_law",
                "standard_language": "Delaware with arbitration",
                "contract_language": "Delaware with negotiation then litigation",
                "deviation_type": "equivalent",
                "impact_assessment": "Acceptable forum selection.",
                "confidence_score": 0.8,
            },
        ],
        "more_favorable_count": 0,
        "less_favorable_count": 2,
        "missing_count": 0,
        "equivalent_count": 1,
        "overall_deviation_score": 0.62,
        "summary": "Several less-favorable commercial terms versus playbook.",
    }


@pytest.mark.asyncio
async def test_compare_terms_without_rag_uses_default_terms() -> None:
    provider = ScriptedModelProvider(json.dumps(_comparison_payload()))
    agent = TermsComparatorAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.compare_terms(
        contract_text="Section 3.1 liability 12 months. Section 5.1 30 days notice.",
        contract_name="Sample MSA",
        session_id="s-no-rag",
    )
    assert response.agent_name == "terms_comparator"
    assert "deviations" in response.output
    assert response.output["retrieved_chunks"] == []
    prompt = str(provider.calls[0]["prompt"])
    assert "liability_cap" in prompt
    assert "none available" in prompt.lower() or "fallback" in prompt.lower()
    assert provider.calls[0]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_compare_terms_with_rag_retriever_uses_contract_domain() -> None:
    provider = ScriptedModelProvider(json.dumps(_comparison_payload()))
    retriever = MagicMock()
    retriever.retrieve_formatted = AsyncMock(
        return_value="[Source: playbook.txt, Page: 1, Relevance: 0.93]\nPreferred liability 24 months"
    )
    agent = TermsComparatorAgent(
        provider=provider,
        metrics=MetricsTracker(),
        rag_retriever=retriever,
    )
    response = await agent.compare_terms(
        contract_text="contract",
        contract_name="Sample MSA",
        session_id="s-rag",
    )
    retriever.retrieve_formatted.assert_awaited_once_with(
        query="standard contract terms for Sample MSA",
        domain="contract",
        top_k=5,
    )
    assert response.output["retrieved_chunks"]
    assert "formatted_context" in response.output["retrieved_chunks"][0]
    assert "Preferred liability 24 months" in str(provider.calls[0]["prompt"])


@pytest.mark.asyncio
async def test_compare_terms_deviation_types_are_valid() -> None:
    provider = ScriptedModelProvider(json.dumps(_comparison_payload()))
    agent = TermsComparatorAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.compare_terms(contract_text="contract", session_id="s-types")
    valid = {"more_favorable", "less_favorable", "missing", "equivalent"}
    for item in response.output["deviations"]:
        assert item["deviation_type"] in valid


@pytest.mark.asyncio
async def test_compare_terms_captures_retrieved_chunks() -> None:
    provider = ScriptedModelProvider(json.dumps(_comparison_payload()))
    retriever = MagicMock()
    retriever.retrieve_formatted = AsyncMock(return_value="standard terms block")
    agent = TermsComparatorAgent(
        provider=provider,
        metrics=MetricsTracker(),
        rag_retriever=retriever,
    )
    response = await agent.compare_terms(
        contract_text="contract",
        contract_name="Alpha",
        session_id="s-chunks",
    )
    assert isinstance(response.output["retrieved_chunks"], list)
    assert response.output["retrieved_chunks"][0]["domain"] == "contract"


@pytest.mark.asyncio
async def test_compare_terms_tracks_metrics() -> None:
    provider = ScriptedModelProvider(json.dumps(_comparison_payload()))
    metrics = MetricsTracker()
    agent = TermsComparatorAgent(provider=provider, metrics=metrics)
    await agent.compare_terms(contract_text="contract", session_id="s-metrics")
    summary = metrics.get_session_summary("s-metrics")
    assert summary["call_count"] == 1
    assert summary["input_tokens"] == 35
    assert summary["output_tokens"] == 55
