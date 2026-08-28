"""Tests for the shared BaseOutputEvaluator skeleton."""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.evaluation import BaseOutputEvaluator
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import TokenUsage


class ScriptedModelProvider(ModelProvider):
    """Returns a preconfigured content string for a single invoke."""

    def __init__(self, content: str) -> None:
        self.content = content

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
        del prompt, temperature, max_tokens, top_p, stop_sequences, agent_name, session_id
        return ModelResponse(
            content=self.content,
            model_id=model_id or "mock-model",
            token_usage=TokenUsage(input_tokens=5, output_tokens=7, model_id="mock-model"),
            latency_ms=1,
        )


class ConcreteEvaluator(BaseOutputEvaluator):
    """Minimal concrete subclass for isolating base-class behavior."""

    AGENT_NAME = "test_evaluator"
    DEFAULT_CRITERIA = ["completeness"]
    post_parse_calls = 0

    def _build_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps(kwargs, default=str)},
        ]

    def _post_parse(
        self, result: dict[str, Any], confidence: float
    ) -> tuple[dict[str, Any], float]:
        type(self).post_parse_calls += 1
        result = dict(result)
        result["post_parse"] = True
        return result, confidence


def test_base_evaluator_is_abstract_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        BaseOutputEvaluator(  # type: ignore[abstract]
            provider=ScriptedModelProvider("{}"),
            metrics=MetricsTracker(),
        )


def test_subclass_must_implement_build_messages() -> None:
    class Incomplete(BaseOutputEvaluator):
        AGENT_NAME = "incomplete"

    with pytest.raises(TypeError):
        Incomplete(  # type: ignore[abstract]
            provider=ScriptedModelProvider("{}"),
            metrics=MetricsTracker(),
        )


def test_parse_evaluation_extracts_confidence() -> None:
    agent = ConcreteEvaluator(
        provider=ScriptedModelProvider("{}"),
        metrics=MetricsTracker(),
    )
    payload = {
        "overall_confidence": 0.91,
        "findings": ["ok"],
        "contradictions": [],
        "summary": "fine",
        "recommendation": "approve",
    }
    result, confidence = agent._parse_evaluation(json.dumps(payload))
    assert confidence == 0.91
    assert result["recommendation"] == "approve"
    assert result["findings"] == ["ok"]


def test_parse_evaluation_fallback_on_bad_json() -> None:
    agent = ConcreteEvaluator(
        provider=ScriptedModelProvider("{}"),
        metrics=MetricsTracker(),
    )
    result, confidence = agent._parse_evaluation("not-json")
    assert confidence == 0.3
    assert result["recommendation"] == "flag_for_review"
    assert result["raw_response"] == "not-json"
    assert "parse_error" in result


@pytest.mark.asyncio
async def test_post_parse_hook_called() -> None:
    ConcreteEvaluator.post_parse_calls = 0
    provider = ScriptedModelProvider(
        json.dumps(
            {
                "overall_confidence": 0.8,
                "findings": [],
                "contradictions": [],
                "summary": "x",
                "recommendation": "approve",
            }
        )
    )
    agent = ConcreteEvaluator(provider=provider, metrics=MetricsTracker())
    response = await agent.evaluate(session_id="s-base", specialist={"a": 1})
    assert ConcreteEvaluator.post_parse_calls == 1
    assert response.output["post_parse"] is True
    assert response.agent_name == "test_evaluator"
