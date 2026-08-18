"""RFP-domain-specific Pydantic models."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid4())


class RfpRequirement(BaseModel):
    """A single extracted requirement from an RFP document."""

    requirement_id: str = Field(default_factory=_new_id)
    text: str
    category: str  # technical, compliance, timeline, budget, staffing, other
    priority: str  # must-have, should-have, nice-to-have
    source_page: int | None = None
    source_section: str | None = None
    entities: list[dict[str, Any]] | None = None
    pii_detected: bool = False


class ExtractionResult(BaseModel):
    """Output of the Requirements Extractor agent."""

    requirements: list[RfpRequirement] = Field(default_factory=list)
    total_extracted: int = 0
    document_pages: int = 1
    extraction_confidence: float = 0.0
    pii_summary: dict[str, Any] | None = None


class CapabilityMapping(BaseModel):
    """A single requirement-to-capability mapping."""

    requirement_id: str
    requirement_text: str
    match_level: str  # "full", "partial", "none"
    capability: str | None = None
    response_draft: str = ""
    confidence: float = 0.0
    gap_note: str | None = None


class MappingResult(BaseModel):
    """Output of the Capability Mapper agent."""

    mappings: list[CapabilityMapping] = Field(default_factory=list)
    fully_matched: int = 0
    partially_matched: int = 0
    unmatched: int = 0
    overall_confidence: float = 0.0


class GapAssessment(BaseModel):
    """A single gap analysis finding."""

    requirement_id: str
    requirement_text: str
    gap_description: str
    risk_severity: str  # "critical", "high", "medium", "low"
    reasoning: str
    mitigation_options: list[str] = Field(default_factory=list)
    recommendation: str  # "respond", "propose-alternative", "partner", "no-bid-risk"


class GapAnalysisResult(BaseModel):
    """Output of the Gap Analyzer agent."""

    assessments: list[GapAssessment] = Field(default_factory=list)
    critical_gaps: int = 0
    high_gaps: int = 0
    medium_gaps: int = 0
    low_gaps: int = 0
    overall_risk: str = "acceptable"
    summary: str = ""
