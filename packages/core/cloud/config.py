"""Provider configuration and factory helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from packages.core.cloud.base import ModelProvider
from packages.core.metrics.tracker import MetricsTracker


class ModelConfig(BaseModel):
    """Runtime configuration for selecting and connecting to a model provider."""

    provider: Literal["bedrock", "local"] = "local"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-sonnet-4-6-20250514"
    local_endpoint: str = "http://localhost:11434"
    local_model_name: str = "llama3"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None


def get_model_config(*, reload_env: bool = False) -> ModelConfig:
    """Load model configuration from environment variables.

    Args:
        reload_env: When True, reload ``.env`` before reading values.

    Returns:
        Populated ``ModelConfig``.
    """
    load_dotenv(override=reload_env)
    provider = os.getenv("MODEL_PROVIDER", "local").strip().lower()
    if provider not in {"bedrock", "local"}:
        raise ValueError(
            f"Unsupported MODEL_PROVIDER '{provider}'. Expected 'bedrock' or 'local'."
        )

    return ModelConfig(
        provider=provider,  # type: ignore[arg-type]
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-sonnet-4-6-20250514",
        ),
        local_endpoint=os.getenv("LOCAL_MODEL_ENDPOINT", "http://localhost:11434"),
        local_model_name=os.getenv("LOCAL_MODEL_NAME", "llama3"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
    )


def get_provider(
    config: ModelConfig | None = None,
    metrics: MetricsTracker | None = None,
) -> ModelProvider:
    """Factory that returns the configured ``ModelProvider`` instance.

    Args:
        config: Optional explicit config; defaults to environment config.
        metrics: Optional shared metrics tracker.

    Returns:
        A concrete ``ModelProvider`` implementation.
    """
    # Local imports avoid circular dependencies at module import time.
    from packages.core.cloud.bedrock import BedrockProvider
    from packages.core.cloud.local_provider import LocalProvider

    resolved = config or get_model_config()
    tracker = metrics or MetricsTracker()

    if resolved.provider == "bedrock":
        return BedrockProvider(config=resolved, metrics=tracker)
    return LocalProvider(config=resolved, metrics=tracker)


@lru_cache(maxsize=1)
def get_default_provider() -> ModelProvider:
    """Return a process-wide default provider instance."""
    return get_provider()
