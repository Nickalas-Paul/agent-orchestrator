"""Register and retrieve versioned prompt templates."""

from __future__ import annotations

from typing import Any

from packages.core.database.connection import DatabasePool
from packages.core.logging.logger import get_logger
from packages.core.prompts.models import PromptVersion

logger = get_logger("prompts.registry")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    id BIGSERIAL PRIMARY KEY,
    prompt_id VARCHAR(128) NOT NULL,
    version INTEGER NOT NULL,
    template_text TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(prompt_id, version)
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prompts_active ON prompt_versions(prompt_id, is_active)
"""


class PromptRegistry:
    """Versioned prompt store with optional Postgres persistence."""

    def __init__(self, db_pool: type[DatabasePool] | None = None) -> None:
        """Initialize the registry.

        Args:
            db_pool: Optional ``DatabasePool`` class. When omitted, storage is
                in-memory only (used by unit tests).
        """
        self._db_pool = db_pool
        self._cache: dict[str, PromptVersion] = {}
        self._versions: dict[str, list[PromptVersion]] = {}
        self._table_ready = False

    def _ensure_table(self) -> None:
        """Create the prompt_versions table if missing. Never raises."""
        if self._db_pool is None or self._table_ready:
            return
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_SQL)
                    cur.execute(_CREATE_INDEX_SQL)
            self._table_ready = True
        except Exception as exc:  # noqa: BLE001 - same resilience as AuditLogger
            logger.error("prompt_registry_ensure_table_failed", error=str(exc))

    def register_prompt(
        self,
        prompt_id: str,
        template_text: str,
        description: str = "",
    ) -> PromptVersion:
        """Register a prompt version. Identical text is idempotent."""
        existing = self._find_matching_text(prompt_id, template_text)
        if existing is not None:
            return existing

        next_version = self._max_version(prompt_id) + 1
        new_prompt = PromptVersion(
            prompt_id=prompt_id,
            version=next_version,
            template_text=template_text,
            description=description,
            is_active=True,
        )
        self._deactivate_previous(prompt_id)
        self._versions.setdefault(prompt_id, []).append(new_prompt)
        self._cache[prompt_id] = new_prompt
        self._persist_new_version(new_prompt)
        logger.info(
            "prompt_registered",
            prompt_id=prompt_id,
            version=next_version,
        )
        return new_prompt

    def get_active_prompt(self, prompt_id: str) -> PromptVersion | None:
        """Return the active version for ``prompt_id``, or None if unknown."""
        cached = self._cache.get(prompt_id)
        if cached is not None:
            return cached
        if self._db_pool is None:
            return None
        self._ensure_table()
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT prompt_id, version, template_text, description,
                               is_active, created_at
                        FROM prompt_versions
                        WHERE prompt_id = %s AND is_active = TRUE
                        ORDER BY version DESC
                        LIMIT 1
                        """,
                        (prompt_id,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    prompt = self._row_to_prompt(cur, row)
                    self._cache[prompt_id] = prompt
                    return prompt
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt_registry_get_active_failed", error=str(exc))
            return None

    def get_prompt_version(self, prompt_id: str, version: int) -> PromptVersion | None:
        """Retrieve a specific historical version."""
        if self._db_pool is None:
            for prompt in self._versions.get(prompt_id, []):
                if prompt.version == version:
                    return prompt
            return None
        self._ensure_table()
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT prompt_id, version, template_text, description,
                               is_active, created_at
                        FROM prompt_versions
                        WHERE prompt_id = %s AND version = %s
                        """,
                        (prompt_id, version),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return self._row_to_prompt(cur, row)
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt_registry_get_version_failed", error=str(exc))
            return None

    def list_prompts(self) -> list[PromptVersion]:
        """Return all active prompt versions."""
        if self._db_pool is None:
            return list(self._cache.values())
        self._ensure_table()
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT prompt_id, version, template_text, description,
                               is_active, created_at
                        FROM prompt_versions
                        WHERE is_active = TRUE
                        ORDER BY prompt_id
                        """
                    )
                    return [self._row_to_prompt(cur, row) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt_registry_list_failed", error=str(exc))
            return list(self._cache.values())

    def _find_matching_text(self, prompt_id: str, template_text: str) -> PromptVersion | None:
        for prompt in self._versions.get(prompt_id, []):
            if prompt.template_text == template_text:
                return prompt
        if self._db_pool is None:
            return None
        self._ensure_table()
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT prompt_id, version, template_text, description,
                               is_active, created_at
                        FROM prompt_versions
                        WHERE prompt_id = %s AND template_text = %s
                        ORDER BY version ASC
                        LIMIT 1
                        """,
                        (prompt_id, template_text),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return self._row_to_prompt(cur, row)
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt_registry_match_failed", error=str(exc))
            return None

    def _max_version(self, prompt_id: str) -> int:
        remembered = self._versions.get(prompt_id, [])
        if remembered:
            return max(prompt.version for prompt in remembered)
        if self._db_pool is None:
            return 0
        self._ensure_table()
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COALESCE(MAX(version), 0) FROM prompt_versions WHERE prompt_id = %s",
                        (prompt_id,),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt_registry_max_version_failed", error=str(exc))
            return 0

    def _deactivate_previous(self, prompt_id: str) -> None:
        for prompt in self._versions.get(prompt_id, []):
            prompt.is_active = False
        if self._db_pool is None:
            return
        self._ensure_table()
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE prompt_versions
                        SET is_active = FALSE
                        WHERE prompt_id = %s AND is_active = TRUE
                        """,
                        (prompt_id,),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt_registry_deactivate_failed", error=str(exc))

    def _persist_new_version(self, prompt: PromptVersion) -> None:
        if self._db_pool is None:
            return
        self._ensure_table()
        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO prompt_versions
                            (prompt_id, version, template_text, description, is_active, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            prompt.prompt_id,
                            prompt.version,
                            prompt.template_text,
                            prompt.description,
                            prompt.is_active,
                            prompt.created_at,
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt_registry_persist_failed", error=str(exc))

    def _row_to_prompt(self, cur: Any, row: Any) -> PromptVersion:
        columns = [desc[0] for desc in cur.description]
        data = dict(zip(columns, row))
        return PromptVersion.model_validate(data)
