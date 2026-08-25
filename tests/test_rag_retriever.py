"""Unit tests for RAGRetriever with mocked embedding and store."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.core.rag.models import ChunkMetadata, DocumentChunk, RetrievalResult
from packages.core.rag.retriever import RAGRetriever


def _meta() -> ChunkMetadata:
    return ChunkMetadata(source_document="capabilities.md", page_number=4, domain="rfp")


@pytest.mark.asyncio
async def test_ingest_document_chunks_and_embeds() -> None:
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[[0.1, 0.2]])
    store = MagicMock()
    store.store_chunks.return_value = 1
    chunker = MagicMock()
    prepared = DocumentChunk(text="capability text", metadata=_meta())
    chunker.chunk_document.return_value = [prepared]

    retriever = RAGRetriever(provider, store)
    result = await retriever.ingest_document("raw document", _meta(), chunker=chunker)

    chunker.chunk_document.assert_called_once()
    provider.embed.assert_awaited_once_with(["capability text"])
    store.store_chunks.assert_called_once()
    assert result == [prepared]


@pytest.mark.asyncio
async def test_ingest_document_assigns_embeddings_to_chunks() -> None:
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[[0.4, 0.5, 0.6]])
    store = MagicMock()
    store.store_chunks.return_value = 1
    chunker = MagicMock()
    prepared = DocumentChunk(text="to embed", metadata=_meta(), embedding=None)
    chunker.chunk_document.return_value = [prepared]

    retriever = RAGRetriever(provider, store)
    result = await retriever.ingest_document("to embed", _meta(), chunker=chunker)

    assert result[0].embedding == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_retrieve_embeds_query_and_searches() -> None:
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[[0.9, 0.1]])
    store = MagicMock()
    store.search.return_value = []
    retriever = RAGRetriever(provider, store, top_k=5)

    await retriever.retrieve("find soc2")

    provider.embed.assert_awaited_once_with(["find soc2"])
    store.search.assert_called_once_with(
        query_embedding=[0.9, 0.1],
        top_k=5,
        domain=None,
        document_type=None,
    )


@pytest.mark.asyncio
async def test_retrieve_formatted_produces_citation_string() -> None:
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[[0.2, 0.3]])
    hit = RetrievalResult(
        chunk=DocumentChunk(
            text="We maintain SOC 2 Type II certification.",
            metadata=ChunkMetadata(source_document="kb.md", page_number=7),
        ),
        similarity_score=0.91,
        rank=1,
    )
    store = MagicMock()
    store.search.return_value = [hit]
    retriever = RAGRetriever(provider, store)

    formatted = await retriever.retrieve_formatted("soc2")

    assert "kb.md" in formatted
    assert "We maintain SOC 2 Type II certification." in formatted
    assert "0.91" in formatted
    assert "Page: 7" in formatted


@pytest.mark.asyncio
async def test_retrieve_with_domain_filter() -> None:
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[[1.0]])
    store = MagicMock()
    store.search.return_value = []
    retriever = RAGRetriever(provider, store, top_k=3)

    await retriever.retrieve("query", top_k=2, domain="vendor", document_type="rfp")

    store.search.assert_called_once_with(
        query_embedding=[1.0],
        top_k=2,
        domain="vendor",
        document_type="rfp",
    )
