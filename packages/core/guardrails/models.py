"""Pydantic models and enums for the shared guardrails layer."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GuardrailAction(str, Enum):
    """What the guardrail decided."""

    PASS = "pass"
    BLOCK = "block"
    FLAG = "flag"  # flag for review but allow through


class InputGuardrailCheck(str, Enum):
    """Names of input-side guardrail checks."""

    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTION = "pii_detection"
    TOPIC_BOUNDARY = "topic_boundary"


class OutputGuardrailCheck(str, Enum):
    """Names of output-side guardrail checks."""

    CONTENT_SAFETY = "content_safety"
    PII_DETECTION = "pii_detection"
    GROUNDING = "grounding"


class GuardrailCheckResult(BaseModel):
    """Result of a single guardrail check."""

    check_name: str
    action: GuardrailAction
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class GuardrailResult(BaseModel):
    """Aggregate result of all guardrail checks on one piece of content."""

    passed: bool
    action: GuardrailAction
    checks: list[GuardrailCheckResult] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    flagged_reasons: list[str] = Field(default_factory=list)


class GuardrailConfig(BaseModel):
    """Configuration for which checks to run and their thresholds."""

    enable_input_injection_detection: bool = True
    enable_input_pii_detection: bool = True
    enable_output_content_safety: bool = True
    enable_output_pii_detection: bool = True
    enable_output_grounding: bool = False
    pii_action: GuardrailAction = GuardrailAction.FLAG
    injection_confidence_threshold: float = 0.7
    grounding_score_threshold: float = 0.5
    blocked_topics: list[str] = Field(default_factory=list)
