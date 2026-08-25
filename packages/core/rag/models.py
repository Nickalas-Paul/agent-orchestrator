"""Pydantic models for RAG chunks and retrieval results."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_chunk_id() -> str:
    return uuid4().hex[:16]


class ChunkMetadata(BaseModel):
    """Provenance metadata attached to a document chunk."""

    source_document: str
    document_type: str = "unknown"
    domain: str = "general"
    page_number: int | None = None
    chunk_index: int = 0
    total_chunks: int = 0
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """A text chunk, optionally with its embedding vector."""

    chunk_id: str = Field(default_factory=_new_chunk_id)
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class RetrievalResult(BaseModel):
    """A chunk returned from similarity search with score and rank."""

    chunk: DocumentChunk
    similarity_score: float
    rank: int
