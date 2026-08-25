"""Cloud model provider abstractions and adapters."""

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.cloud.config import ModelConfig, get_model_config, get_provider
from packages.core.cloud.embeddings import (
    BedrockEmbeddingProvider,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_provider,
)

__all__ = [
    "BedrockEmbeddingProvider",
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "ModelConfig",
    "ModelProvider",
    "ModelResponse",
    "get_embedding_provider",
    "get_model_config",
    "get_provider",
]
