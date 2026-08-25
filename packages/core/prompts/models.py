"""Pydantic models for versioned prompt templates."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PromptVersion(BaseModel):
    """A single immutable version of a named prompt template."""

    prompt_id: str
    version: int
    template_text: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
