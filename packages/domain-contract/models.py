"""Contract risk-review domain Pydantic models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContractPipelineStatus(str, Enum):
    """Terminal or paused status returned by ``ContractReviewPipeline.review``."""

    COMPLETED = "completed"
    PENDING_REVIEW = "pending_review"
    FAILED = "failed"


class RiskSeverity(str, Enum):
    """Severity level for a clause risk or compliance gap."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClauseRisk(BaseModel):
    """A single clause-level risk finding with reasoning."""

    clause_id: str
    clause_text: str
    risk_severity: RiskSeverity
    risk_category: str  # e.g., liability, ip_ownership, termination, indemnification
    confidence_score: float
    reasoning_trace: str
    source_section: str
    mitigation_suggestion: str = ""


class ClauseRiskResult(BaseModel):
    """Output of the Clause Risk Analyst agent."""

    risks: list[ClauseRisk] = Field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    overall_risk_level: RiskSeverity = RiskSeverity.LOW


class ComplianceGap(BaseModel):
    """A single compliance gap or requirement finding."""

    requirement: str
    status: str  # compliant, non_compliant, partially_compliant, missing
    clause_reference: str = ""
    explanation: str = ""
    severity: RiskSeverity = RiskSeverity.MEDIUM


class ComplianceResult(BaseModel):
    """Output of the Compliance Checker agent."""

    gaps: list[ComplianceGap] = Field(default_factory=list)
    compliant_count: int = 0
    non_compliant_count: int = 0
    partially_compliant_count: int = 0
    missing_count: int = 0
    overall_compliance_score: float = 0.0


class TermDeviation(BaseModel):
    """A single deviation between standard terms and contract language."""

    term_name: str
    standard_language: str
    contract_language: str
    deviation_type: str  # more_favorable, less_favorable, missing, equivalent
    impact_assessment: str = ""
    confidence_score: float = 0.0


class TermsComparisonResult(BaseModel):
    """Output of the Terms Comparator agent."""

    deviations: list[TermDeviation] = Field(default_factory=list)
    more_favorable_count: int = 0
    less_favorable_count: int = 0
    missing_count: int = 0
    equivalent_count: int = 0
    overall_deviation_score: float = 0.0


class ContractReviewResult(BaseModel):
    """Complete result of a contract risk-review pipeline run."""

    status: ContractPipelineStatus
    job_id: str = ""
    contract_name: str = ""
    clause_risk_result: dict[str, Any] | None = None
    compliance_result: dict[str, Any] | None = None
    terms_comparison_result: dict[str, Any] | None = None
    evaluation_result: dict[str, Any] | None = None
    business_metrics: dict[str, Any] | None = None
    agent_responses: list[dict[str, Any]] = Field(default_factory=list)
    hitl_reason: str = ""
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    prompt_versions: dict[str, int] = Field(default_factory=dict)
