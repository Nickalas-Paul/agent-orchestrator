"""Postgres/pgvector storage for document chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from packages.core.database.connection import DatabasePool
from packages.core.logging.logger import get_logger
from packages.core.rag.models import ChunkMetadata, DocumentChunk, RetrievalResult

logger = get_logger("rag.vector_store")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id VARCHAR(32) NOT NULL UNIQUE,
    text TEXT NOT NULL,
    embedding vector(1024),
    source_document VARCHAR(512) NOT NULL,
    document_type VARCHAR(64) NOT NULL DEFAULT 'unknown',
    domain VARCHAR(64) NOT NULL DEFAULT 'general',
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON document_chunks(source_document)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_domain ON document_chunks(domain)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON document_chunks(document_type)",
    """
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
    """,
]


class VectorStore:
    """CRUD and cosine similarity search over ``document_chunks``."""

    def __init__(self, embedding_dimensions: int = 1024) -> None:
        """Initialize the store.

        Args:
            embedding_dimensions: Expected embedding size (must match the model).
        """
        self._embedding_dimensions = embedding_dimensions
        self._table_ready = False

    def _register_vector(self, conn: Any) -> None:
        """Register the pgvector adapter on a connection; never raise."""
        try:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
        except Exception as exc:  # noqa: BLE001 - defensive adapter registration
            logger.error("pgvector_register_failed", error=str(exc))

    def _ensure_table(self) -> None:
        """Create the chunks table and HNSW index if missing. Never raises."""
        if self._table_ready:
            return
        try:
            with DatabasePool.get_connection() as conn:
                self._register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_SQL)
                    for statement in _CREATE_INDEXES_SQL:
                        cur.execute(statement)
            self._table_ready = True
        except Exception as exc:  # noqa: BLE001 - same resilience as AuditLogger
            logger.error("vector_store_ensure_table_failed", error=str(exc))

    def store_chunks(self, chunks: list[DocumentChunk]) -> int:
        """INSERT chunks that already have embeddings. Returns the stored count."""
        missing = [chunk.chunk_id for chunk in chunks if chunk.embedding is None]
        if missing:
            raise ValueError(
                f"Cannot store chunks without embeddings: {missing}"
            )
        self._ensure_table()
        insert_sql = """
            INSERT INTO document_chunks
                (chunk_id, text, embedding, source_document, document_type, domain,
                 page_number, chunk_index, total_chunks, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with DatabasePool.get_connection() as conn:
            self._register_vector(conn)
            with conn.cursor() as cur:
                for chunk in chunks:
                    meta = chunk.metadata
                    cur.execute(
                        insert_sql,
                        (
                            chunk.chunk_id,
                            chunk.text,
                            chunk.embedding,
                            meta.source_document,
                            meta.document_type,
                            meta.domain,
                            meta.page_number,
                            meta.chunk_index,
                            meta.total_chunks,
                            meta.ingested_at,
                        ),
                    )
        source = chunks[0].metadata.source_document if chunks else ""
        logger.info(
            "chunks_stored",
            chunk_count=len(chunks),
            source_document=source,
        )
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        domain: str | None = None,
        document_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Cosine similarity search with optional metadata filters. Never raises."""
        self._ensure_table()
        try:
            filters: list[str] = []
            params: list[Any] = [query_embedding]
            if domain is not None:
                filters.append("domain = %s")
                params.append(domain)
            if document_type is not None:
                filters.append("document_type = %s")
                params.append(document_type)
            where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
            sql = f"""
                SELECT chunk_id, text, source_document, document_type, domain,
                       page_number, chunk_index, total_chunks, ingested_at,
                       embedding <=> %s::vector AS distance
                FROM document_chunks
                {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params.extend([query_embedding, top_k])
            with DatabasePool.get_connection() as conn:
                self._register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    columns = [desc[0] for desc in cur.description]
                    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            results: list[RetrievalResult] = []
            for rank, row in enumerate(rows, start=1):
                distance = float(row.get("distance") or 0.0)
                similarity = 1.0 - distance
                ingested_at = row.get("ingested_at") or datetime.now(timezone.utc)
                chunk = DocumentChunk(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    metadata=ChunkMetadata(
                        source_document=str(row["source_document"]),
                        document_type=str(row.get("document_type") or "unknown"),
                        domain=str(row.get("domain") or "general"),
                        page_number=row.get("page_number"),
                        chunk_index=int(row.get("chunk_index") or 0),
                        total_chunks=int(row.get("total_chunks") or 0),
                        ingested_at=ingested_at,
                    ),
                )
                results.append(
                    RetrievalResult(chunk=chunk, similarity_score=similarity, rank=rank)
                )
            return results
        except Exception as exc:  # noqa: BLE001 - search must not crash callers
            logger.error("vector_store_search_failed", error=str(exc))
            return []

    def delete_by_document(self, source_document: str) -> int:
        """DELETE all chunks for ``source_document``. Returns rows deleted."""
        self._ensure_table()
        with DatabasePool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE source_document = %s",
                    (source_document,),
                )
                deleted = cur.rowcount if cur.rowcount is not None else 0
        logger.info(
            "chunks_deleted",
            source_document=source_document,
            deleted=deleted,
        )
        return int(deleted)

    def count_chunks(self, domain: str | None = None) -> int:
        """Return the number of stored chunks, optionally filtered by domain."""
        self._ensure_table()
        with DatabasePool.get_connection() as conn:
            with conn.cursor() as cur:
                if domain is None:
                    cur.execute("SELECT COUNT(*) FROM document_chunks")
                    row = cur.fetchone()
                else:
                    cur.execute(
                        "SELECT COUNT(*) FROM document_chunks WHERE domain = %s",
                        (domain,),
                    )
                    row = cur.fetchone()
        return int(row[0]) if row else 0
