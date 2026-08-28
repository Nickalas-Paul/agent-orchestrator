"""Tests for the compliance checker agent."""

from __future__ import annotations

import json

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_contract.agents.compliance_checker import ComplianceCheckerAgent


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
                input_tokens=25,
                output_tokens=45,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.006,
            ),
            latency_ms=3,
        )


def _compliance_payload() -> dict:
    return {
        "gaps": [
            {
                "requirement": "Governing law clause present",
                "status": "compliant",
                "clause_reference": "Section 9.1",
                "explanation": "Delaware governing law is stated.",
                "severity": "low",
            },
            {
                "requirement": "Data protection with documented controls",
                "status": "non_compliant",
                "clause_reference": "Section 8.1",
                "explanation": "Only commercially reasonable language is present.",
                "severity": "high",
            },
            {
                "requirement": "Confidentiality duration adequacy",
                "status": "partially_compliant",
                "clause_reference": "Section 6.3",
                "explanation": "Two-year term is shorter than preferred five years.",
                "severity": "medium",
            },
        ],
        "compliant_count": 1,
        "non_compliant_count": 1,
        "partially_compliant_count": 1,
        "missing_count": 0,
        "overall_compliance_score": 0.55,
        "summary": "Mixed compliance with weak data-protection language.",
    }


@pytest.mark.asyncio
async def test_check_compliance_returns_valid_agent_response() -> None:
    provider = ScriptedModelProvider(json.dumps(_compliance_payload()))
    agent = ComplianceCheckerAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.check_compliance(
        contract_text="Section 9.1 Delaware. Section 8.1 commercially reasonable.",
        session_id="s-comp",
    )
    assert response.agent_name == "compliance_checker"
    assert "gaps" in response.output
    assert len(response.output["gaps"]) == 3
    assert provider.calls[0]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_check_compliance_status_values_are_valid() -> None:
    provider = ScriptedModelProvider(json.dumps(_compliance_payload()))
    agent = ComplianceCheckerAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.check_compliance(contract_text="contract", session_id="s-status")
    valid = {"compliant", "non_compliant", "partially_compliant", "missing"}
    for gap in response.output["gaps"]:
        assert gap["status"] in valid


@pytest.mark.asyncio
async def test_check_compliance_score_in_unit_interval() -> None:
    provider = ScriptedModelProvider(json.dumps(_compliance_payload()))
    agent = ComplianceCheckerAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.check_compliance(contract_text="contract", session_id="s-score")
    score = response.output["overall_compliance_score"]
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_check_compliance_tracks_metrics() -> None:
    provider = ScriptedModelProvider(json.dumps(_compliance_payload()))
    metrics = MetricsTracker()
    agent = ComplianceCheckerAgent(provider=provider, metrics=metrics)
    await agent.check_compliance(contract_text="contract", session_id="s-metrics")
    summary = metrics.get_session_summary("s-metrics")
    assert summary["call_count"] == 1
    assert summary["input_tokens"] == 25
    assert summary["output_tokens"] == 45
