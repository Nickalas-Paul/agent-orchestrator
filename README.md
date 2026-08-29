# Agent Orchestrator

Multi-agent AI pipeline for enterprise document analysis with built-in governance.

## What This Is

A Python monorepo that implements three document-analysis domains as **sequential specialist agent pipelines**: RFP response analysis, vendor evaluation, and contract risk review. Built around Amazon Bedrock (with a local provider option), Postgres + pgvector, and a shared governance stack — input/output guardrails, LLM-as-a-judge evaluation, human-in-the-loop review, and an immutable audit trail.

## Architecture Overview

**Core (`packages/core/`)** — provider ABCs, RAG, guardrails, `BaseOutputEvaluator`, HITL, audit, RBAC/secrets, logging, metrics, prompts, and service adapters (Textract/Comprehend).

**Domains**

- `packages/domain-rfp/` — Requirements Extractor → Capability Mapper → Gap Analyzer → OutputEvaluator
- `packages/domain-vendor/` — Capability Researcher → Pricing Analyst → Market Positioner → VendorOutputEvaluator
- `packages/domain-contract/` — Clause Risk Analyst → Compliance Checker → Terms Comparator → ContractOutputEvaluator

Pipelines run agents in **hardcoded sequential order**. `OrchestratorEngine` / `DAGExecutor` exist but are not the domain runtime. Full design notes: [Architecture](docs/ARCHITECTURE.md).

## Quickstart

```bash
git clone https://github.com/Nickalas-Paul/agent-orchestrator.git
cd agent-orchestrator

# Required: .env sets POSTGRES_PORT=5433 (and other config).
# Without it, Docker Compose defaults the host port to 5432 while
# DatabasePool defaults to 5433 — connections will fail.
cp .env.example .env

docker compose -f infra/docker-compose.yml up -d
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest   # 172 tests passing
```

**Postgres port:** application and `.env.example` use **5433**. Compose publishes `"${POSTGRES_PORT:-5432}:5432"`, so copying `.env` is mandatory for a working local stack.

**Redis:** included in Compose on port **6379** but **not used** by application code today.

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System design, agent boundaries, data flow |
| [Governance](docs/GOVERNANCE.md) | HITL design, audit trail, guardrails, confidence model |
| [Security](docs/SECURITY.md) | Access controls, secrets, audit immutability, data classification |
| [Case Study](docs/CASE_STUDY.md) | Enterprise narrative per domain |
| [AWS Service Mapping](docs/AWS_SERVICE_MAPPING.md) | Component-to-AWS-service equivalents |

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Infrastructure | Docker Compose (Postgres 16 + pgvector, Redis 7) |
| LLM Access | Amazon Bedrock via boto3; local provider via httpx |
| Embeddings | Bedrock Titan Embeddings; local `nomic-embed-text` |
| AWS Adapters | Textract, Comprehend |
| Logging | structlog (JSON, sensitivity separation) |
| Testing | pytest (172 tests) |

## Design Highlights

- **Core / domain boundary** — domains import core only; domains never import each other  
- **Provider swap** — Bedrock or local inference without changing agent code  
- **Governance stack** — guardrails → LLM-as-a-judge → HITL → immutable audit ([Governance](docs/GOVERNANCE.md))  
- **Security controls** — HITL RBAC, secrets ABC, DB audit triggers, sensitivity logging ([Security](docs/SECURITY.md))  
- **Honest gaps** — sequential pipelines (not Step Functions runtime), Redis unused, embedding dimension mismatch tracked in docs  

## Project Status

**Phase 4 — Package + Certify.** Three domain implementations complete. **172** tests passing at commit `094b419` (Batch 8 complete; Batch 9 documentation suite).

License and contribution notes are not the focus of this batch; treat the repo as a portfolio reference implementation for governed multi-agent document workflows.
