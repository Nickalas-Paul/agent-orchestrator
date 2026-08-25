"""Embedding provider abstraction and implementations."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import httpx
from botocore.exceptions import BotoCoreError, ClientError

from packages.core.logging.logger import get_logger

logger = get_logger("cloud.embeddings")

_LOCAL_MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 384,
}


class BedrockEmbeddingError(Exception):
    """Raised when a Bedrock embedding request fails."""


class EmbeddingProvider(ABC):
    """Abstract interface for turning text into embedding vectors."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the embedding model in use."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return one vector per input string."""


class BedrockEmbeddingProvider(EmbeddingProvider):
    """Amazon Titan embeddings via Bedrock Runtime."""

    def __init__(
        self,
        region_name: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the Bedrock embedding provider.

        Args:
            region_name: AWS region for the Bedrock runtime client.
            model_id: Titan embedding model identifier.
            aws_access_key_id: Optional explicit access key.
            aws_secret_access_key: Optional explicit secret key.
            client: Optional pre-built boto3 client (useful for tests).
        """
        self._model_id = model_id
        self._dimensions = 1024
        if client is not None:
            self._client = client
        else:
            import boto3

            session_kwargs: dict[str, Any] = {"region_name": region_name}
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = aws_access_key_id
                session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            self._client = boto3.client("bedrock-runtime", **session_kwargs)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts one at a time using the Titan Embeddings request body."""
        vectors: list[list[float]] = []
        for text in texts:
            try:
                raw_response = self._client.invoke_model(
                    modelId=self._model_id,
                    body=json.dumps({"inputText": text}),
                    contentType="application/json",
                    accept="application/json",
                )
            except (ClientError, BotoCoreError) as exc:
                raise BedrockEmbeddingError(
                    f"Bedrock embedding request failed for model {self._model_id}: {exc}"
                ) from exc
            payload = self._parse_response_body(raw_response)
            embedding = payload.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise BedrockEmbeddingError(
                    f"Bedrock embedding response missing embedding vector for model {self._model_id}"
                )
            vectors.append([float(value) for value in embedding])
        return vectors

    def _parse_response_body(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Decode the body from a Bedrock invoke_model response."""
        body = raw_response.get("body")
        if body is None:
            raise BedrockEmbeddingError("Bedrock embedding response missing body")
        if hasattr(body, "read"):
            raw_bytes = body.read()
        else:
            raw_bytes = body
        if isinstance(raw_bytes, bytes):
            return json.loads(raw_bytes.decode("utf-8"))
        if isinstance(raw_bytes, str):
            return json.loads(raw_bytes)
        if isinstance(raw_bytes, dict):
            return raw_bytes
        raise BedrockEmbeddingError(
            f"Unexpected Bedrock embedding body type: {type(raw_bytes)}"
        )


class LocalEmbeddingProvider(EmbeddingProvider):
    """Ollama-compatible local embedding provider with mock-mode fallback."""

    def __init__(
        self,
        endpoint: str | None = None,
        model_id: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the local embedding provider.

        Args:
            endpoint: Ollama-compatible base URL.
            model_id: Embedding model name.
            http_client: Optional shared httpx client (tests can inject mocks).
        """
        self._endpoint = (
            endpoint
            or os.getenv("LOCAL_MODEL_ENDPOINT", "http://localhost:11434")
        ).rstrip("/")
        self._model_id = model_id or os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
        self._http_client = http_client
        self._dimensions = _LOCAL_MODEL_DIMENSIONS.get(self._model_id, 384)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """POST each text to ``/api/embeddings``, with optional mock fallback."""
        try:
            vectors: list[list[float]] = []
            for text in texts:
                vectors.append(await self._embed_one(text))
            return vectors
        except (httpx.HTTPError, OSError, ValueError) as exc:
            allow_mock = os.getenv("ALLOW_MOCK_LLM", "false").lower() == "true"
            if not allow_mock:
                raise ConnectionError(
                    f"Local embedding endpoint unreachable at {self._endpoint} "
                    "and ALLOW_MOCK_LLM is not enabled. "
                    "Set ALLOW_MOCK_LLM=true to allow mock embeddings in development."
                ) from exc
            logger.warning(
                "local_embedding_provider_mock_mode",
                endpoint=self._endpoint,
                model_id=self._model_id,
                error=str(exc),
                dimensions=self._dimensions,
            )
            return [[0.0] * self._dimensions for _ in texts]

    async def _embed_one(self, text: str) -> list[float]:
        """Embed a single string via the Ollama embeddings API."""
        url = f"{self._endpoint}/api/embeddings"
        payload = {"model": self._model_id, "prompt": text}
        data = await self._post_json(url, payload)
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Local embedding response missing embedding vector")
        return [float(value) for value in embedding]

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the local endpoint and return the parsed body."""
        if self._http_client is not None:
            response = await self._http_client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Local embedding response was not a JSON object")
            return data

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Local embedding response was not a JSON object")
            return data


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return a process-wide embedding provider selected by ``MODEL_PROVIDER``."""
    from packages.core.cloud.config import get_model_config

    config = get_model_config()
    if config.provider == "bedrock":
        return BedrockEmbeddingProvider(
            region_name=config.aws_region,
            model_id=config.embedding_model_id,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
    return LocalEmbeddingProvider(
        endpoint=config.local_endpoint,
        model_id=os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text"),
    )
