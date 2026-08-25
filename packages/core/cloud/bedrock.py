"""AWS Bedrock model provider adapter."""

from __future__ import annotations

import json
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.cloud.config import ModelConfig
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import estimate_cost
from packages.core.types.schemas import TokenUsage

logger = get_logger("cloud.bedrock")


class BedrockProviderError(Exception):
    """Base error for Bedrock provider failures."""


class BedrockAuthError(BedrockProviderError):
    """Raised when AWS credentials are missing or invalid."""


class BedrockModelNotFoundError(BedrockProviderError):
    """Raised when the requested model id cannot be found."""


class BedrockThrottlingError(BedrockProviderError):
    """Raised when Bedrock throttles the request."""


class BedrockProvider(ModelProvider):
    """Model provider backed by Amazon Bedrock Runtime."""

    def __init__(
        self,
        config: ModelConfig,
        client: Any | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            config: Provider configuration.
            client: Optional pre-built boto3 client (useful for tests).
        """
        self._config = config
        if client is not None:
            self._client = client
        else:
            session_kwargs: dict[str, Any] = {"region_name": config.aws_region}
            if config.aws_access_key_id and config.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = config.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = config.aws_secret_access_key
            self._client = boto3.client("bedrock-runtime", **session_kwargs)

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
        """Invoke a Bedrock model using the Messages API request shape."""
        resolved_model = model_id or self._config.bedrock_model_id
        body = self._build_request_body(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )

        started = time.perf_counter()
        try:
            raw_response = self._client.invoke_model(
                modelId=resolved_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except ClientError as exc:
            raise self._map_client_error(exc, resolved_model) from exc
        except BotoCoreError as exc:
            raise BedrockProviderError(f"Bedrock request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        payload = self._parse_response_body(raw_response)
        content = self._extract_content(payload)
        input_tokens, output_tokens = self._extract_token_counts(payload, raw_response)
        cost = estimate_cost(resolved_model, input_tokens, output_tokens)
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=resolved_model,
            estimated_cost_usd=cost,
        )
        logger.info(
            "bedrock_invoke_complete",
            model_id=resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            agent_name=agent_name,
            session_id=session_id,
        )
        return ModelResponse(
            content=content,
            model_id=resolved_model,
            token_usage=usage,
            latency_ms=latency_ms,
        )

    def build_request_body(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
    ) -> dict[str, Any]:
        """Public helper used by tests to assert request formatting."""
        return self._build_request_body(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )

    def _build_request_body(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: list[str] | None,
    ) -> dict[str, Any]:
        """Build Anthropic Messages API body accepted by Bedrock Claude models."""
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        if stop_sequences:
            body["stop_sequences"] = stop_sequences
        return body

    def _parse_response_body(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Decode the streaming body from a Bedrock invoke_model response."""
        body = raw_response.get("body")
        if body is None:
            raise BedrockProviderError("Bedrock response missing body")
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
        raise BedrockProviderError(f"Unexpected Bedrock body type: {type(raw_bytes)}")

    def _extract_content(self, payload: dict[str, Any]) -> str:
        """Extract assistant text from Messages API or text-generation payloads."""
        content = payload.get("content")
        if isinstance(content, list) and content:
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            if parts:
                return "".join(parts)
        if "completion" in payload:
            return str(payload["completion"])
        if "generation" in payload:
            return str(payload["generation"])
        raise BedrockProviderError("Unable to extract content from Bedrock response")

    def _extract_token_counts(
        self,
        payload: dict[str, Any],
        raw_response: dict[str, Any],
    ) -> tuple[int, int]:
        """Extract input/output token counts from response payload or headers."""
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("outputTokens") or 0)

        headers = raw_response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        if input_tokens == 0:
            input_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0) or 0)
        if output_tokens == 0:
            output_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0) or 0)
        return input_tokens, output_tokens

    def _map_client_error(self, exc: ClientError, model_id: str) -> BedrockProviderError:
        """Map AWS client errors to clearer domain exceptions."""
        error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = str(error.get("Code", ""))
        message = str(error.get("Message", str(exc)))

        if code in {"UnrecognizedClientException", "InvalidSignatureException", "AccessDeniedException"}:
            return BedrockAuthError(f"Bedrock authentication failed: {message}")
        if code in {"ResourceNotFoundException", "ValidationException"} and "model" in message.lower():
            return BedrockModelNotFoundError(f"Bedrock model not found: {model_id} ({message})")
        if code in {"ThrottlingException", "TooManyRequestsException", "ModelTimeoutException"}:
            return BedrockThrottlingError(f"Bedrock throttled request for {model_id}: {message}")
        return BedrockProviderError(f"Bedrock error ({code}): {message}")
