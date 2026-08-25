"""Unit tests for DocumentChunker strategies."""

from __future__ import annotations

from packages.core.rag.chunker import DocumentChunker
from packages.core.rag.models import ChunkMetadata


def _meta(source: str = "doc.txt") -> ChunkMetadata:
    return ChunkMetadata(source_document=source, document_type="rfp", domain="rfp")


def test_fixed_chunking_produces_correct_sizes() -> None:
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=0, strategy="fixed")
    text = "abcdefghij" * 3
    chunks = chunker.chunk_document(text, _meta())
    assert len(chunks) == 3
    assert all(len(chunk.text) == 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_fixed_chunking_overlap() -> None:
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=3, strategy="fixed")
    text = "0123456789ABCDEFGHIJ"
    chunks = chunker.chunk_document(text, _meta())
    assert len(chunks) >= 2
    assert chunks[0].text[-3:] == chunks[1].text[:3]


def test_recursive_chunking_splits_on_paragraphs() -> None:
    chunker = DocumentChunker(chunk_size=40, chunk_overlap=0, strategy="recursive")
    text = "First paragraph is short enough.\n\nSecond paragraph is also short enough."
    chunks = chunker.chunk_document(text, _meta())
    assert len(chunks) == 2
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[1].text


def test_recursive_chunking_falls_back_to_sentences() -> None:
    chunker = DocumentChunker(chunk_size=30, chunk_overlap=0, strategy="recursive")
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunker.chunk_document(text, _meta())
    assert len(chunks) >= 2
    joined = " ".join(chunk.text for chunk in chunks)
    assert "sentence one" in joined
    assert "sentence two" in joined
    assert "sentence three" in joined


def test_recursive_chunking_falls_back_to_fixed() -> None:
    chunker = DocumentChunker(chunk_size=8, chunk_overlap=0, strategy="recursive")
    text = "ABCDEFGHIJKLMNOPQRST"
    chunks = chunker.chunk_document(text, _meta())
    assert [chunk.text for chunk in chunks] == ["ABCDEFGH", "IJKLMNOP", "QRST"]


def test_chunk_metadata_propagated() -> None:
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=0, strategy="fixed")
    chunks = chunker.chunk_document("abcdefghij" * 2, _meta("rfp.pdf"))
    assert len(chunks) == 2
    for index, chunk in enumerate(chunks):
        assert chunk.metadata.source_document == "rfp.pdf"
        assert chunk.metadata.chunk_index == index
        assert chunk.metadata.total_chunks == 2


def test_empty_document_returns_empty_list() -> None:
    chunker = DocumentChunker()
    assert chunker.chunk_document("", _meta()) == []


def test_semantic_chunking_stub_falls_back_to_recursive() -> None:
    text = "Paragraph alpha is here.\n\nParagraph beta is here."
    meta = _meta()
    recursive = DocumentChunker(chunk_size=40, chunk_overlap=0, strategy="recursive")
    semantic = DocumentChunker(chunk_size=40, chunk_overlap=0, strategy="semantic")
    rec_texts = [chunk.text for chunk in recursive.chunk_document(text, meta)]
    sem_texts = [chunk.text for chunk in semantic.chunk_document(text, meta)]
    assert sem_texts == rec_texts
    assert len(sem_texts) >= 2
