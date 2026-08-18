"""AWS managed AI service adapters (Textract, Comprehend)."""

from packages.core.services.base import (
    ComprehendResult,
    DocumentProcessor,
    Entity,
    PiiEntity,
    TextAnalyzer,
    TextBlock,
    TextractResult,
)
from packages.core.services.config import (
    ServiceConfig,
    get_document_processor,
    get_service_config,
    get_text_analyzer,
)

__all__ = [
    "ComprehendResult",
    "DocumentProcessor",
    "Entity",
    "PiiEntity",
    "ServiceConfig",
    "TextAnalyzer",
    "TextBlock",
    "TextractResult",
    "get_document_processor",
    "get_service_config",
    "get_text_analyzer",
]
