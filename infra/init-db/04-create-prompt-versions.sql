-- Batch 4: Prompt version tracking

CREATE TABLE IF NOT EXISTS prompt_versions (
    id BIGSERIAL PRIMARY KEY,
    prompt_id VARCHAR(128) NOT NULL,
    version INTEGER NOT NULL,
    template_text TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(prompt_id, version)
);

CREATE INDEX IF NOT EXISTS idx_prompts_active ON prompt_versions(prompt_id, is_active);
