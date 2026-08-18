"""Local model provider with graceful mock fallback."""

from __future__ import annotations

import time
from typing import Any

import httpx

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.cloud.config import ModelConfig
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker, estimate_cost
from packages.core.types.schemas import TokenUsage

logger = get_logger("cloud.local")


class LocalProvider(ModelProvider):
    """Ollama-compatible local model provider with mock-mode fallback."""

    def __init__(
        self,
        config: ModelConfig,
        metrics: MetricsTracker | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the local provider.

        Args:
            config: Provider configuration.
            metrics: Optional metrics tracker.
            http_client: Optional shared httpx client (tests can inject mocks).
        """
        self._config = config
        self._metrics = metrics or MetricsTracker()
        self._http_client = http_client

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
        """Invoke a local model endpoint, falling back to mock mode if unreachable."""
        resolved_model = model_id or self._config.local_model_name
        endpoint = self._config.local_endpoint.rstrip("/")
        url = f"{endpoint}/api/generate"
        payload: dict[str, Any] = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        started = time.perf_counter()
        try:
            response_data = await self._post_json(url, payload)
            content = str(response_data.get("response", "")).strip()
            if not content:
                content = (
                    "[local mock] Empty response from local endpoint; "
                    "returning placeholder content."
                )
            input_tokens, output_tokens = self._extract_or_estimate_tokens(
                prompt=prompt,
                content=content,
                response_data=response_data,
            )
            mock_mode = False
        except (httpx.HTTPError, OSError, ValueError) as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            content = (
                "[local mock] Local model endpoint is unavailable. "
                f"Returning mock response. Reason: {exc}"
            )
            input_tokens = max(1, len(prompt) // 4)
            output_tokens = max(1, len(content) // 4)
            mock_mode = True
            logger.warning(
                "local_provider_mock_mode",
                endpoint=endpoint,
                model_id=resolved_model,
                error=str(exc),
                latency_ms=latency_ms,
            )
        else:
            latency_ms = int((time.perf_counter() - started) * 1000)

        cost = estimate_cost(resolved_model, input_tokens, output_tokens)
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=resolved_model,
            estimated_cost_usd=cost,
        )
        self._metrics.track_llm_call(
            agent_name=agent_name,
            model_id=resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            session_id=session_id,
        )
        logger.info(
            "local_invoke_complete",
            model_id=resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            mock_mode=mock_mode,
            agent_name=agent_name,
            session_id=session_id,
        )
        return ModelResponse(
            content=content,
            model_id=resolved_model,
            token_usage=usage,
            latency_ms=latency_ms,
        )

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the local endpoint and return the parsed body."""
        if self._http_client is not None:
            response = await self._http_client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Local provider response was not a JSON object")
            return data

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Local provider response was not a JSON object")
            return data

    def _extract_or_estimate_tokens(
        self,
        prompt: str,
        content: str,
        response_data: dict[str, Any],
    ) -> tuple[int, int]:
        """Use provider token counts when present; otherwise estimate from chars."""
        prompt_eval = response_data.get("prompt_eval_count")
        eval_count = response_data.get("eval_count")
        if isinstance(prompt_eval, int) and isinstance(eval_count, int):
            return prompt_eval, eval_count
        return max(1, len(prompt) // 4), max(1, len(content) // 4)
