"""Unit tests for VectorStore with a mocked DatabasePool."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from packages.core.database.connection import DatabasePool
from packages.core.rag.models import ChunkMetadata, DocumentChunk
from packages.core.rag.vector_store import VectorStore


def _chunk(
    text: str = "hello world",
    *,
    embedding: list[float] | None = None,
    source: str = "doc.txt",
    domain: str = "rfp",
) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        metadata=ChunkMetadata(
            source_document=source,
            document_type="rfp",
            domain=domain,
            page_number=1,
            chunk_index=0,
            total_chunks=1,
        ),
        embedding=embedding,
    )


def _patched_store() -> tuple[VectorStore, MagicMock]:
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get = patch.object(DatabasePool, "get_connection")
    mock_cm = mock_get.start()
    mock_cm.return_value.__enter__.return_value = mock_conn
    store = VectorStore(embedding_dimensions=1024)
    store._table_ready = True
    return store, mock_cursor


def test_store_chunks_calls_insert() -> None:
    store, cursor = _patched_store()
    chunk = _chunk(embedding=[0.1, 0.2, 0.3])
    try:
        stored = store.store_chunks([chunk])
    finally:
        patch.stopall()

    assert stored == 1
    insert_calls = [
        call for call in cursor.execute.call_args_list if "INSERT" in call[0][0]
    ]
    assert len(insert_calls) == 1
    params = insert_calls[0][0][1]
    assert params[0] == chunk.chunk_id
    assert params[1] == "hello world"
    assert params[2] == [0.1, 0.2, 0.3]
    assert params[3] == "doc.txt"


def test_store_chunks_rejects_missing_embeddings() -> None:
    store, _cursor = _patched_store()
    try:
        with pytest.raises(ValueError, match="without embeddings"):
            store.store_chunks([_chunk(embedding=None)])
    finally:
        patch.stopall()


def test_search_returns_retrieval_results() -> None:
    store, cursor = _patched_store()
    ingested = datetime(2026, 1, 15, tzinfo=timezone.utc)
    cursor.description = [
        ("chunk_id",),
        ("text",),
        ("source_document",),
        ("document_type",),
        ("domain",),
        ("page_number",),
        ("chunk_index",),
        ("total_chunks",),
        ("ingested_at",),
        ("distance",),
    ]
    cursor.fetchall.return_value = [
        (
            "chunk-1",
            "retrieved text",
            "kb.md",
            "rfp",
            "rfp",
            2,
            0,
            1,
            ingested,
            0.25,
        )
    ]
    try:
        results = store.search([0.1] * 4, top_k=5)
    finally:
        patch.stopall()

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].similarity_score == pytest.approx(0.75)
    assert results[0].chunk.text == "retrieved text"
    assert results[0].chunk.metadata.source_document == "kb.md"
    assert results[0].chunk.metadata.page_number == 2


def test_search_with_domain_filter() -> None:
    store, cursor = _patched_store()
    cursor.description = [
        ("chunk_id",),
        ("text",),
        ("source_document",),
        ("document_type",),
        ("domain",),
        ("page_number",),
        ("chunk_index",),
        ("total_chunks",),
        ("ingested_at",),
        ("distance",),
    ]
    cursor.fetchall.return_value = []
    try:
        store.search([0.1, 0.2], top_k=3, domain="vendor")
    finally:
        patch.stopall()

    sql, params = cursor.execute.call_args[0]
    assert "WHERE" in sql
    assert "domain = %s" in sql
    assert "vendor" in params


def test_search_returns_empty_on_error() -> None:
    with patch.object(DatabasePool, "get_connection", side_effect=RuntimeError("db down")):
        store = VectorStore()
        assert store.search([0.1, 0.2]) == []


def test_delete_by_document() -> None:
    store, cursor = _patched_store()
    cursor.rowcount = 4
    try:
        deleted = store.delete_by_document("old.pdf")
    finally:
        patch.stopall()

    assert deleted == 4
    sql, params = cursor.execute.call_args[0]
    assert "DELETE FROM document_chunks" in sql
    assert params == ("old.pdf",)


def test_count_chunks() -> None:
    store, cursor = _patched_store()
    cursor.fetchone.return_value = (9,)
    try:
        assert store.count_chunks() == 9
        assert store.count_chunks(domain="rfp") == 9
    finally:
        patch.stopall()

    count_calls = [
        call for call in cursor.execute.call_args_list if "COUNT(*)" in call[0][0]
    ]
    assert any("SELECT COUNT(*) FROM document_chunks" in call[0][0] for call in count_calls)
    assert any("WHERE domain = %s" in call[0][0] for call in count_calls)
    domain_call = next(call for call in count_calls if "WHERE domain = %s" in call[0][0])
    assert domain_call[0][1] == ("rfp",)
