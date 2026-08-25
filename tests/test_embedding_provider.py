"""Unit tests for Bedrock and local embedding providers."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from packages.core.cloud.embeddings import BedrockEmbeddingProvider, LocalEmbeddingProvider


@pytest.mark.asyncio
async def test_local_embedding_provider_calls_ollama_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3, 0.4]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = LocalEmbeddingProvider(
        endpoint="http://localhost:11434",
        model_id="nomic-embed-text",
        http_client=client,
    )
    vectors = await provider.embed(["hello"])
    await client.aclose()

    assert captured["url"] == "http://localhost:11434/api/embeddings"
    assert captured["body"] == {"model": "nomic-embed-text", "prompt": "hello"}
    assert vectors == [[0.1, 0.2, 0.3, 0.4]]


@pytest.mark.asyncio
async def test_local_embedding_provider_mock_fallback_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_MOCK_LLM", "true")

    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_raise_connect_error))
    provider = LocalEmbeddingProvider(
        endpoint="http://localhost:9",
        model_id="nomic-embed-text",
        http_client=client,
    )
    vectors = await provider.embed(["one", "two"])
    await client.aclose()

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert all(value == 0.0 for vector in vectors for value in vector)


@pytest.mark.asyncio
async def test_local_embedding_provider_raises_when_mock_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_MOCK_LLM", "false")

    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_raise_connect_error))
    provider = LocalEmbeddingProvider(
        endpoint="http://localhost:9",
        model_id="nomic-embed-text",
        http_client=client,
    )
    with pytest.raises(ConnectionError, match="ALLOW_MOCK_LLM is not enabled"):
        await provider.embed(["hello"])
    await client.aclose()


@pytest.mark.asyncio
async def test_bedrock_embedding_provider_calls_invoke_model() -> None:
    captured: list[dict[str, Any]] = []

    class FakeBody:
        def read(self) -> bytes:
            return json.dumps({"embedding": [0.5, 0.25]}).encode("utf-8")

    def fake_invoke_model(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"body": FakeBody()}

    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = fake_invoke_model
    provider = BedrockEmbeddingProvider(
        model_id="amazon.titan-embed-text-v2:0",
        client=mock_client,
    )
    await provider.embed(["alpha"])

    assert len(captured) == 1
    assert captured[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    assert captured[0]["contentType"] == "application/json"
    assert captured[0]["accept"] == "application/json"
    assert json.loads(captured[0]["body"]) == {"inputText": "alpha"}


@pytest.mark.asyncio
async def test_bedrock_embedding_provider_returns_vectors() -> None:
    class FakeBody:
        def read(self) -> bytes:
            return json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode("utf-8")

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": FakeBody()}
    provider = BedrockEmbeddingProvider(client=mock_client)
    vectors = await provider.embed(["one", "two"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert mock_client.invoke_model.call_count == 2
    assert provider.model_id == "amazon.titan-embed-text-v2:0"
    assert provider.dimensions == 1024
