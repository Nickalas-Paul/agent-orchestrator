"""Process-level connection pool for Postgres."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from psycopg2 import pool
from psycopg2.extensions import connection as PgConnection


class DatabasePool:
    """Process-level connection pool for Postgres."""

    _pool: pool.ThreadedConnectionPool | None = None

    @classmethod
    def initialize(cls, min_conn: int = 1, max_conn: int = 5) -> None:
        """Create the shared connection pool if it does not already exist."""
        if cls._pool is not None:
            return
        cls._pool = pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5433")),
            dbname=os.getenv("POSTGRES_DB", "agent_orchestrator"),
            user=os.getenv("POSTGRES_USER", "agent_user"),
            password=os.getenv("POSTGRES_PASSWORD", "agent_password"),
        )

    @classmethod
    @contextmanager
    def get_connection(cls) -> Generator[PgConnection, None, None]:
        """Yield a pooled connection, committing on success and rolling back on error."""
        if cls._pool is None:
            cls.initialize()
        if cls._pool is None:
            raise RuntimeError("Database connection pool failed to initialize")

        conn = cls._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cls._pool.putconn(conn)

    @classmethod
    def close_all(cls) -> None:
        """Close all pooled connections and reset the process-level pool."""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None
