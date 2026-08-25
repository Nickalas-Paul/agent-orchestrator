"""Run database migrations. Safe to run repeatedly — all statements use IF NOT EXISTS."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2


def run_migrations() -> None:
    """Execute every SQL file in ``infra/init-db`` in sorted filename order."""
    migration_dir = Path(__file__).parent.parent.parent.parent / "infra" / "init-db"
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "agent_orchestrator"),
        user=os.getenv("POSTGRES_USER", "agent_user"),
        password=os.getenv("POSTGRES_PASSWORD", "agent_password"),
    )
    try:
        with conn.cursor() as cur:
            for sql_file in sorted(migration_dir.glob("*.sql")):
                cur.execute(sql_file.read_text(encoding="utf-8"))
        conn.commit()
        print("Migrations complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_migrations()
