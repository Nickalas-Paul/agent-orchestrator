"""Abstract interfaces for AWS managed AI service adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """A block of text extracted from a document."""

    text: str
    block_type: str  # "LINE", "WORD", "TABLE", "KEY_VALUE_SET", "PAGE"
    confidence: float  # 0-100
    page: int = 1
    geometry: dict[str, Any] | None = None


class TextractResult(BaseModel):
    """Complete result from document text extraction."""

    pages: int = 1
    blocks: list[TextBlock] = Field(default_factory=list)
    raw_text: str = ""
    tables: list[dict[str, Any]] = Field(default_factory=list)
    key_value_pairs: list[dict[str, Any]] = Field(default_factory=list)


class DocumentProcessor(ABC):
    """Abstract interface for document text extraction."""

    @abstractmethod
    async def extract_text(
        self,
        document: bytes | str,
        document_type: str = "pdf",
    ) -> TextractResult:
        """Extract structured text from a document.

        Args:
            document: Raw document bytes or file path.
            document_type: ``pdf``, ``png``, ``jpg``, or ``tiff``.

        Returns:
            Structured text extraction result.
        """


class Entity(BaseModel):
    """A named entity detected in text."""

    text: str
    entity_type: str
    confidence: float  # 0-100
    begin_offset: int
    end_offset: int


class PiiEntity(BaseModel):
    """Personally identifiable information detected in text."""

    type: str
    confidence: float  # 0-100
    begin_offset: int
    end_offset: int


class ComprehendResult(BaseModel):
    """Complete NLP analysis result."""

    entities: list[Entity] = Field(default_factory=list)
    pii_entities: list[PiiEntity] = Field(default_factory=list)
    key_phrases: list[str] = Field(default_factory=list)
    sentiment: str = "NEUTRAL"
    sentiment_scores: dict[str, float] = Field(default_factory=dict)
    language: str = "en"


class TextAnalyzer(ABC):
    """Abstract interface for NLP text analysis."""

    @abstractmethod
    async def analyze(self, text: str) -> ComprehendResult:
        """Run full NLP analysis on text.

        Returns entities, PII detection, key phrases, sentiment, and language.
        """

    @abstractmethod
    async def detect_pii(self, text: str) -> list[PiiEntity]:
        """Detect personally identifiable information in text."""

    @abstractmethod
    async def detect_entities(self, text: str) -> list[Entity]:
        """Detect named entities in text."""
