"""Ingest and retrieve documents using embeddings plus the vector store."""

from __future__ import annotations

from packages.core.cloud.embeddings import EmbeddingProvider
from packages.core.logging.logger import get_logger
from packages.core.rag.chunker import DocumentChunker
from packages.core.rag.models import ChunkMetadata, DocumentChunk, RetrievalResult
from packages.core.rag.vector_store import VectorStore

logger = get_logger("rag.retriever")


class RAGRetriever:
    """Orchestrates chunking, embedding, storage, and similarity search."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedding_provider: Injected embedding adapter.
            vector_store: Injected vector store.
            top_k: Default number of results to return.
        """
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._top_k = top_k

    async def ingest_document(
        self,
        text: str,
        metadata: ChunkMetadata,
        chunker: DocumentChunker | None = None,
    ) -> list[DocumentChunk]:
        """Chunk, embed, and store a document. Returns chunks with embeddings."""
        resolved = chunker or DocumentChunker()
        chunks = resolved.chunk_document(text, metadata)
        if not chunks:
            return []
        embeddings = await self._embedding_provider.embed([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk.embedding = embedding
        stored = self._vector_store.store_chunks(chunks)
        logger.info(
            "document_ingested",
            source_document=metadata.source_document,
            chunk_count=stored,
        )
        return chunks

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        domain: str | None = None,
        document_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Embed ``query`` and search the vector store."""
        embeddings = await self._embedding_provider.embed([query])
        query_embedding = embeddings[0] if embeddings else []
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k if top_k is not None else self._top_k,
            domain=domain,
            document_type=document_type,
        )
        logger.info(
            "retrieval_complete",
            result_count=len(results),
            domain=domain,
            document_type=document_type,
        )
        return results

    async def retrieve_formatted(
        self,
        query: str,
        top_k: int | None = None,
        domain: str | None = None,
        document_type: str | None = None,
    ) -> str:
        """Retrieve chunks and format them for prompt injection with citations."""
        results = await self.retrieve(
            query=query,
            top_k=top_k,
            domain=domain,
            document_type=document_type,
        )
        blocks: list[str] = []
        for result in results:
            meta = result.chunk.metadata
            header = (
                f"[Source: {meta.source_document}, "
                f"Page: {meta.page_number}, "
                f"Relevance: {result.similarity_score:.2f}]"
            )
            blocks.append(f"{header}\n{result.chunk.text}")
        return "\n\n---\n\n".join(blocks)
