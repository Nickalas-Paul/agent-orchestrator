"""Tests for the contract output evaluator agent."""

from __future__ import annotations

import json

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_contract.agents.contract_evaluator import ContractOutputEvaluator
from packages.domain_contract.prompts.evaluator_prompt import DEFAULT_EVALUATION_CRITERIA


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
        "criterion_scores": {
            "completeness": 0.9,
            "consistency": 0.88,
            "risk_identification_quality": 0.91,
            "reasoning_depth": 0.87,
            "source_citation_accuracy": 0.85,
        },
        "findings": ["Risk and compliance outputs are aligned."],
        "contradictions": [],
        "summary": "Specialist outputs are consistent and complete.",
        "recommendation": "approve" if overall_confidence >= 0.85 else "flag_for_review",
    }


def test_contract_evaluator_default_criteria_applied() -> None:
    agent = ContractOutputEvaluator(
        provider=ScriptedModelProvider("{}"),
        metrics=MetricsTracker(),
    )
    assert agent._criteria == list(DEFAULT_EVALUATION_CRITERIA)


def test_contract_evaluator_custom_criteria_override() -> None:
    agent = ContractOutputEvaluator(
        provider=ScriptedModelProvider("{}"),
        metrics=MetricsTracker(),
        evaluation_criteria=["completeness", "consistency"],
    )
    assert agent._criteria == ["completeness", "consistency"]


@pytest.mark.asyncio
async def test_evaluate_outputs_returns_confidence_and_recommendation() -> None:
    provider = ScriptedModelProvider(json.dumps(_evaluation_payload()))
    agent = ContractOutputEvaluator(provider=provider, metrics=MetricsTracker())
    response = await agent.evaluate_outputs(
        risk_result={"risks": []},
        compliance_result={"gaps": []},
        comparison_result={"deviations": []},
        session_id="s-eval",
    )
    assert response.agent_name == "contract_evaluator"
    assert "overall_confidence" in response.output
    assert response.output["recommendation"] in {"approve", "flag_for_review"}
    assert provider.calls[0]["temperature"] == 0.2
    assert 0.0 <= response.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_evaluate_outputs_clamps_confidence() -> None:
    payload = _evaluation_payload(overall_confidence=1.7)
    provider = ScriptedModelProvider(json.dumps(payload))
    agent = ContractOutputEvaluator(provider=provider, metrics=MetricsTracker())
    response = await agent.evaluate_outputs(
        risk_result={},
        compliance_result={},
        comparison_result={},
        session_id="s-clamp",
    )
    assert response.confidence_score == 1.0
    assert response.output["overall_confidence"] == 1.0
