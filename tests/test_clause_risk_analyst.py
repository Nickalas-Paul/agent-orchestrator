"""Tests for the clause risk analyst agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from packages.core.audit.models import ActionType
from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage
from packages.domain_contract.agents.clause_risk_analyst import ClauseRiskAnalystAgent


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
            latency_ms=4,
        )


def _risk_payload() -> dict:
    return {
        "risks": [
            {
                "clause_id": "R1",
                "clause_text": "Liability shall not exceed 12 months fees.",
                "risk_severity": "high",
                "risk_category": "liability",
                "confidence_score": 0.91,
                "reasoning_trace": (
                    "Step 1: Cap is 12 months fees. "
                    "Step 2: Preferred playbook requires 24 months. "
                    "Step 3: Under-capping increases residual exposure. "
                    "Step 4: Severity is high for enterprise deals."
                ),
                "source_section": "Section 3.1",
                "mitigation_suggestion": "Increase cap to 24 months fees.",
            }
        ],
        "critical_count": 0,
        "high_count": 1,
        "medium_count": 0,
        "low_count": 0,
        "overall_risk_level": "high",
        "summary": "Liability cap is the primary concern.",
    }


@pytest.mark.asyncio
async def test_analyze_risks_returns_valid_agent_response() -> None:
    provider = ScriptedModelProvider(json.dumps(_risk_payload()))
    agent = ClauseRiskAnalystAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.analyze_risks(
        contract_text="Section 3.1 — liability capped at 12 months fees.",
        session_id="s-risk",
    )
    assert response.agent_name == "clause_risk_analyst"
    assert "risks" in response.output
    assert len(response.output["risks"]) == 1
    assert provider.calls[0]["temperature"] == 0.2
    assert 0.0 <= response.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_analyze_risks_includes_reasoning_trace() -> None:
    provider = ScriptedModelProvider(json.dumps(_risk_payload()))
    agent = ClauseRiskAnalystAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.analyze_risks(contract_text="contract", session_id="s-cot")
    trace = response.output["risks"][0]["reasoning_trace"]
    assert "Step 1" in trace
    assert len(trace) > 40


@pytest.mark.asyncio
async def test_analyze_risks_severity_values_are_valid() -> None:
    provider = ScriptedModelProvider(json.dumps(_risk_payload()))
    agent = ClauseRiskAnalystAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.analyze_risks(contract_text="contract", session_id="s-sev")
    for risk in response.output["risks"]:
        assert risk["risk_severity"] in {"critical", "high", "medium", "low"}
    assert response.output["overall_risk_level"] in {"critical", "high", "medium", "low"}


@pytest.mark.asyncio
async def test_analyze_risks_tracks_metrics() -> None:
    provider = ScriptedModelProvider(json.dumps(_risk_payload()))
    metrics = MetricsTracker()
    agent = ClauseRiskAnalystAgent(provider=provider, metrics=metrics)
    await agent.analyze_risks(contract_text="contract", session_id="s-metrics")
    summary = metrics.get_session_summary("s-metrics")
    assert summary["call_count"] == 1
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 50


@pytest.mark.asyncio
async def test_analyze_risks_audit_logging_when_logger_provided() -> None:
    provider = ScriptedModelProvider(json.dumps(_risk_payload()))
    audit = MagicMock()
    agent = ClauseRiskAnalystAgent(
        provider=provider,
        metrics=MetricsTracker(),
        audit_logger=audit,
    )
    await agent.analyze_risks(contract_text="contract", session_id="s-audit")
    audit.log.assert_called_once()
    entry = audit.log.call_args.args[0]
    assert entry.action_type == ActionType.INVOKE
    assert entry.agent_name == "clause_risk_analyst"
    assert entry.session_id == "s-audit"
