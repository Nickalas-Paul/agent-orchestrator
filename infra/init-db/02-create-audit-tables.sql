-- Batch 3: Audit log and pipeline job state tables

CREATE TABLE IF NOT EXISTS agent_audit_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    agent_name VARCHAR(128) NOT NULL,
    action_type VARCHAR(32) NOT NULL,  -- 'invoke', 'tool_call', 'evaluation', 'hitl_decision'
    input_payload JSONB DEFAULT '{}',
    output_payload JSONB DEFAULT '{}',
    confidence_score FLOAT,
    hitl_status VARCHAR(32),  -- 'pending_review', 'approved', 'rejected'
    hitl_reviewer VARCHAR(128),
    token_count INTEGER DEFAULT 0,
    cost_usd FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_session ON agent_audit_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent ON agent_audit_logs(agent_name);
CREATE INDEX IF NOT EXISTS idx_audit_created ON agent_audit_logs(created_at);

CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    pipeline_name VARCHAR(128) NOT NULL DEFAULT 'rfp_analysis',
    status VARCHAR(32) NOT NULL DEFAULT 'running',  -- 'running', 'completed', 'pending_review', 'approved', 'rejected', 'failed'
    pipeline_inputs JSONB DEFAULT '{}',
    pipeline_outputs JSONB DEFAULT '{}',
    evaluator_reasoning JSONB DEFAULT '{}',
    evaluator_confidence FLOAT,
    reviewer VARCHAR(128),
    review_decision VARCHAR(32),  -- 'approve', 'reject'
    review_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_session ON pipeline_jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON pipeline_jobs(status);
