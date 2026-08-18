"""Tests for cloud provider configuration and adapters."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from packages.core.cloud.bedrock import BedrockProvider
from packages.core.cloud.config import ModelConfig, get_provider
from packages.core.cloud.local_provider import LocalProvider
from packages.core.metrics.tracker import MetricsTracker


def test_get_provider_returns_bedrock_when_configured() -> None:
    config = ModelConfig(provider="bedrock", aws_region="us-east-1")
    provider = get_provider(config=config, metrics=MetricsTracker())
    assert isinstance(provider, BedrockProvider)


def test_get_provider_returns_local_when_configured() -> None:
    config = ModelConfig(provider="local", local_endpoint="http://localhost:11434")
    provider = get_provider(config=config, metrics=MetricsTracker())
    assert isinstance(provider, LocalProvider)


@pytest.mark.asyncio
async def test_local_provider_graceful_mock_mode_on_unreachable_endpoint() -> None:
    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    transport = httpx.MockTransport(_raise_connect_error)
    client = httpx.AsyncClient(transport=transport)
    provider = LocalProvider(
        config=ModelConfig(
            provider="local",
            local_endpoint="http://localhost:9",
            local_model_name="llama3",
        ),
        metrics=MetricsTracker(),
        http_client=client,
    )

    response = await provider.invoke(
        prompt="Hello",
        agent_name="tester",
        session_id="sess-1",
    )
    await client.aclose()

    assert "[local mock]" in response.content
    assert response.model_id == "llama3"
    assert response.token_usage.input_tokens > 0
    assert response.token_usage.output_tokens > 0


def test_bedrock_provider_constructs_correct_request_format() -> None:
    mock_client = MagicMock()
    provider = BedrockProvider(
        config=ModelConfig(provider="bedrock", bedrock_model_id="anthropic.claude-sonnet"),
        metrics=MetricsTracker(),
        client=mock_client,
    )

    body = provider.build_request_body(
        prompt="Plan this work",
        temperature=0.2,
        max_tokens=256,
        top_p=0.8,
        stop_sequences=["END"],
    )

    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.8
    assert body["stop_sequences"] == ["END"]
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"][0] == {
        "type": "text",
        "text": "Plan this work",
    }


@pytest.mark.asyncio
async def test_bedrock_provider_invoke_uses_request_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure invoke_model is called with the Messages API JSON body."""
    captured: dict[str, Any] = {}

    class FakeBody:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }
            ).encode("utf-8")

    def fake_invoke_model(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"body": FakeBody(), "ResponseMetadata": {"HTTPHeaders": {}}}

    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = fake_invoke_model
    provider = BedrockProvider(
        config=ModelConfig(
            provider="bedrock",
            bedrock_model_id="anthropic.claude-sonnet",
        ),
        metrics=MetricsTracker(),
        client=mock_client,
    )

    result = await provider.invoke(prompt="hi", agent_name="orch", session_id="s")
    assert result.content == "ok"
    assert result.token_usage.input_tokens == 11
    assert result.token_usage.output_tokens == 7
    body = json.loads(captured["body"])
    assert body["messages"][0]["content"][0]["text"] == "hi"
    assert captured["modelId"] == "anthropic.claude-sonnet"
