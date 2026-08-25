"""Tests for the RFP output evaluator agent."""

from __future__ import annotations

import json

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_rfp.agents.evaluator import OutputEvaluator


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
                input_tokens=50,
                output_tokens=80,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.01,
            ),
            latency_ms=4,
        )


def _evaluation_payload(*, overall_confidence: float = 0.82) -> dict:
    return {
        "overall_confidence": overall_confidence,
        "findings": [
            {
                "criterion": "completeness",
                "score": 0.85,
                "reasoning": "Requirement density matches a one-page RFP.",
            },
            {
                "criterion": "consistency",
                "score": 0.9,
                "reasoning": "Every requirement has a mapping.",
            },
            {
                "criterion": "contradiction",
                "score": 0.8,
                "reasoning": "Gap assessments align with partial matches.",
            },
            {
                "criterion": "reasoning_quality",
                "score": 0.78,
                "reasoning": "Gap traces cite numbered steps.",
            },
        ],
        "contradictions": [],
        "summary": "Outputs are coherent with one moderate support gap.",
        "recommendation": "approve",
    }


@pytest.mark.asyncio
async def test_evaluator_returns_valid_agent_response() -> None:
    provider = ScriptedModelProvider(json.dumps(_evaluation_payload()))
    agent = OutputEvaluator(provider=provider, metrics=MetricsTracker())
    response = await agent.evaluate_outputs(
        extraction_result={"requirements": [{"requirement_id": "REQ-001"}]},
        mapping_result={"mappings": [{"requirement_id": "REQ-001", "match_level": "full"}]},
        gap_result={"assessments": []},
        session_id="s-eval",
    )
    assert response.agent_name == "evaluator"
    assert 0.0 < response.confidence_score <= 1.0
    assert isinstance(response.output["findings"], list)
    assert len(response.output["findings"]) == 4
    assert response.output["recommendation"] in {"approve", "flag_for_review"}
    assert provider.calls[0]["temperature"] == 0.2
    assert provider.calls[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_evaluator_handles_json_parse_failure() -> None:
    provider = ScriptedModelProvider("this is not json")
    agent = OutputEvaluator(provider=provider, metrics=MetricsTracker())
    response = await agent.evaluate_outputs(
        extraction_result={},
        mapping_result={},
        gap_result={},
        session_id="s-eval-bad",
    )
    assert response.confidence_score == 0.3
    assert response.output["raw_response"] == "this is not json"
    assert response.output["recommendation"] == "flag_for_review"


@pytest.mark.asyncio
async def test_evaluator_confidence_clamped_to_valid_range() -> None:
    high = ScriptedModelProvider(json.dumps(_evaluation_payload(overall_confidence=1.7)))
    high_response = await OutputEvaluator(
        provider=high, metrics=MetricsTracker()
    ).evaluate_outputs({}, {}, {}, session_id="s-high")
    assert high_response.confidence_score == 1.0
    assert high_response.output["overall_confidence"] == 1.0

    low = ScriptedModelProvider(json.dumps(_evaluation_payload(overall_confidence=-0.4)))
    low_response = await OutputEvaluator(
        provider=low, metrics=MetricsTracker()
    ).evaluate_outputs({}, {}, {}, session_id="s-low")
    assert low_response.confidence_score == 0.0
    assert low_response.output["overall_confidence"] == 0.0
