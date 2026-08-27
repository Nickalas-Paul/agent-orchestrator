"""Vendor-evaluation domain Pydantic models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VendorPipelineStatus(str, Enum):
    """Terminal or paused status returned by ``VendorEvaluationPipeline.evaluate``."""

    COMPLETED = "completed"
    PENDING_REVIEW = "pending_review"
    FAILED = "failed"


class CapabilityAssessment(BaseModel):
    """Single vendor capability with evidence from RAG retrieval."""

    capability_name: str
    evidence: str = ""
    confidence_score: float = 0.0
    source_citation: str = ""


class PricingAnalysis(BaseModel):
    """Vendor pricing structure analysis."""

    pricing_model: str = ""  # e.g. "per-seat", "usage-based", "tiered"
    estimated_annual_cost: str = ""
    cost_breakdown: list[dict[str, Any]] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    comparison_notes: str = ""


class MarketPosition(BaseModel):
    """Vendor competitive positioning assessment."""

    market_segment: str = ""
    differentiators: list[str] = Field(default_factory=list)
    competitive_advantages: list[str] = Field(default_factory=list)
    competitive_risks: list[str] = Field(default_factory=list)
    strategic_assessment: str = ""


class VendorEvaluationResult(BaseModel):
    """Complete result of a vendor evaluation pipeline run."""

    status: VendorPipelineStatus
    job_id: str | None = None
    vendor_name: str = ""
    capability_assessments: list[dict[str, Any]] = Field(default_factory=list)
    pricing_analysis: dict[str, Any] = Field(default_factory=dict)
    market_position: dict[str, Any] = Field(default_factory=dict)
    evaluation_result: dict[str, Any] = Field(default_factory=dict)
    business_metrics: dict[str, Any] = Field(default_factory=dict)
    agent_responses: list[dict[str, Any]] = Field(default_factory=list)
    hitl_reason: str | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
