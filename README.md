# Agent Orchestrator

Enterprise-grade multi-agent orchestration framework.

**Status:** Phase 1 — Foundations

## Quickstart

### Prerequisites

- Python 3.11+
- Docker Desktop (Docker Compose v2)

### Setup

```bash
# Copy environment template
cp .env.example .env

# Start infrastructure (Postgres + Redis)
docker compose -f infra/docker-compose.yml up -d

# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install package with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

This project is under active development.
