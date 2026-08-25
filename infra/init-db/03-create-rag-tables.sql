-- Batch 4: RAG document chunks with vector embeddings

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
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON document_chunks(source_document);
CREATE INDEX IF NOT EXISTS idx_chunks_domain ON document_chunks(domain);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON document_chunks(document_type);

-- HNSW index for cosine similarity search
-- vector_cosine_ops is for the <=> (cosine distance) operator
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops);
