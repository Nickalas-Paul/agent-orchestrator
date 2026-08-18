"""Abstract base class and response models for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from packages.core.types.schemas import TokenUsage


class ModelResponse(BaseModel):
    """Normalized response returned by any model provider."""

    content: str
    model_id: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0


class ModelProvider(ABC):
    """Abstract interface for invoking large language models."""

    @abstractmethod
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
        """Invoke the underlying model and return a normalized response.

        Args:
            prompt: User prompt text.
            model_id: Optional model override; providers may use a default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            stop_sequences: Optional stop sequences.
            agent_name: Calling agent name for metrics/logging.
            session_id: Session id for metrics/logging.

        Returns:
            Normalized ``ModelResponse``.
        """
