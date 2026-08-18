"""Tests for RFP specialist agents with mocked model providers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.services.comprehend import LocalTextAnalyzer
from packages.core.services.textract import LocalDocumentProcessor
from packages.core.types.schemas import TokenUsage
from packages.domain_rfp.agents.capability_mapper import CapabilityMapperAgent
from packages.domain_rfp.agents.gap_analyzer import GapAnalyzerAgent
from packages.domain_rfp.agents.requirements_extractor import RequirementsExtractorAgent
from packages.domain_rfp.models import (
    CapabilityMapping,
    MappingResult,
    RfpRequirement,
)

SAMPLE_RFP = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "domain-rfp"
    / "sample_data"
    / "sample_rfp_text.txt"
)


class ScriptedModelProvider(ModelProvider):
    """Returns scripted responses in call order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def invoke(
        self,
        prompt: str,
        model_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
        *,
        agent_name: str = "unknown",
        session_id: str = "unknown",
    ) -> ModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "agent_name": agent_name,
                "session_id": session_id,
            }
        )
        content = self._responses.pop(0) if self._responses else "{}"
        return ModelResponse(
            content=content,
            model_id=model_id or "mock-model",
            token_usage=TokenUsage(
                input_tokens=100,
                output_tokens=200,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.01,
            ),
            latency_ms=5,
        )


def _extraction_payload() -> dict:
    return {
        "requirements": [
            {
                "requirement_id": "REQ-001",
                "text": "The solution shall provide a documented REST API integration capability.",
                "category": "technical",
                "priority": "must-have",
                "source_page": 1,
                "source_section": "2.1",
                "entities": [],
                "pii_detected": False,
            },
            {
                "requirement_id": "REQ-002",
                "text": "The vendor must maintain a current SOC 2 Type II certification.",
                "category": "compliance",
                "priority": "must-have",
                "source_page": 1,
                "source_section": "3.1",
                "entities": [],
                "pii_detected": False,
            },
            {
                "requirement_id": "REQ-003",
                "text": "Provide on-site technical support within four hours for Severity-1 incidents.",
                "category": "staffing",
                "priority": "should-have",
                "source_page": 1,
                "source_section": "6.2",
                "entities": [],
                "pii_detected": False,
            },
        ],
        "total_extracted": 3,
        "document_pages": 1,
        "extraction_confidence": 0.8,
        "pii_summary": {"count": 1, "types": {"EMAIL": 1}},
    }


def _mapping_payload() -> dict:
    return {
        "mappings": [
            {
                "requirement_id": "REQ-001",
                "requirement_text": "REST API integration capability",
                "match_level": "full",
                "capability": "REST API Integration Platform",
                "response_draft": "We provide documented REST APIs with OAuth2.",
                "confidence": 0.95,
                "gap_note": None,
            },
            {
                "requirement_id": "REQ-002",
                "requirement_text": "SOC 2 Type II certification",
                "match_level": "full",
                "capability": "SOC 2 Type II Certified",
                "response_draft": "We maintain SOC 2 Type II.",
                "confidence": 0.97,
                "gap_note": None,
            },
            {
                "requirement_id": "REQ-003",
                "requirement_text": "on-site technical support within four hours",
                "match_level": "partial",
                "capability": "24/7 Enterprise Support",
                "response_draft": "We offer 24/7 support and on-site where available.",
                "confidence": 0.6,
                "gap_note": "4-hour on-site not guaranteed everywhere.",
            },
            {
                "requirement_id": "REQ-004",
                "requirement_text": "Must run exclusively on-premises",
                "match_level": "none",
                "capability": None,
                "response_draft": "Not supported.",
                "confidence": 0.9,
                "gap_note": "No on-prem capability.",
            },
        ],
        "fully_matched": 2,
        "partially_matched": 1,
        "unmatched": 1,
        "overall_confidence": 0.8,
    }


def _gap_payload() -> dict:
    return {
        "assessments": [
            {
                "requirement_id": "REQ-003",
                "requirement_text": "on-site technical support within four hours",
                "gap_description": "On-site 4-hour SLA is not universally available.",
                "risk_severity": "medium",
                "reasoning": (
                    "1. The requirement demands 4-hour on-site support. "
                    "2. Mapped capability is 24/7 remote enterprise support. "
                    "3. The gap is guaranteed physical dispatch timing. "
                    "4. Business impact is moderate for a headquarters-only clause. "
                    "5. Severity is medium. "
                    "6. Mitigation includes partnering with a local field-services firm."
                ),
                "mitigation_options": [
                    "Partner with local field services",
                    "Propose enhanced remote bridge + best-effort on-site",
                ],
                "recommendation": "propose-alternative",
            },
            {
                "requirement_id": "REQ-004",
                "requirement_text": "Must run exclusively on-premises",
                "gap_description": "No on-premises-only offering exists.",
                "risk_severity": "high",
                "reasoning": (
                    "Step 1: Requirement demands exclusive on-prem. "
                    "Step 2: No capability mapped. "
                    "Step 3: Full architectural gap. "
                    "Step 4: Likely competitive disqualification risk. "
                    "Step 5: High severity. "
                    "Step 6: Partner for private-cloud appliance or no-bid."
                ),
                "mitigation_options": ["Partner for appliance", "No-bid"],
                "recommendation": "partner",
            },
        ],
        "critical_gaps": 0,
        "high_gaps": 1,
        "medium_gaps": 1,
        "low_gaps": 0,
        "overall_risk": "manageable",
        "summary": "One high and one medium gap; bid remains manageable with partners.",
    }


@pytest.mark.asyncio
async def test_requirements_extractor_structure_and_confidence() -> None:
    provider = ScriptedModelProvider([json.dumps(_extraction_payload())])
    agent = RequirementsExtractorAgent(
        provider=provider,
        metrics=MetricsTracker(),
        document_processor=LocalDocumentProcessor(),
        text_analyzer=LocalTextAnalyzer(),
    )
    response = await agent.process_document(str(SAMPLE_RFP), session_id="s-extract")
    assert response.agent_name == "requirements_extractor"
    assert response.output["total_extracted"] == 3
    assert len(response.output["requirements"]) == 3
    assert 0.0 < response.confidence_score < 1.0
    assert response.confidence_score != 1.0
    assert provider.calls[0]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_capability_mapper_categorizes_match_levels() -> None:
    requirements = [
        RfpRequirement(
            requirement_id="REQ-001",
            text="REST API",
            category="technical",
            priority="must-have",
        ),
        RfpRequirement(
            requirement_id="REQ-003",
            text="on-site support",
            category="staffing",
            priority="should-have",
        ),
        RfpRequirement(
            requirement_id="REQ-004",
            text="on-prem only",
            category="technical",
            priority="must-have",
        ),
    ]
    provider = ScriptedModelProvider([json.dumps(_mapping_payload())])
    agent = CapabilityMapperAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.map_requirements(requirements, session_id="s-map")
    assert response.agent_name == "capability_mapper"
    assert response.output["fully_matched"] == 2
    assert response.output["partially_matched"] == 1
    assert response.output["unmatched"] == 1
    levels = {m["match_level"] for m in response.output["mappings"]}
    assert levels == {"full", "partial", "none"}
    assert 0.0 < response.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_gap_analyzer_includes_reasoning_traces() -> None:
    mapping = MappingResult.model_validate(_mapping_payload())
    provider = ScriptedModelProvider([json.dumps(_gap_payload())])
    agent = GapAnalyzerAgent(provider=provider, metrics=MetricsTracker())
    response = await agent.analyze_gaps(mapping, session_id="s-gap")
    assert response.agent_name == "gap_analyzer"
    assessments = response.output["assessments"]
    assert len(assessments) == 2
    assert all(len(a["reasoning"]) > 40 for a in assessments)
    assert response.output["overall_risk"] in {
        "acceptable",
        "manageable",
        "high-risk",
        "no-bid",
    }
    assert 0.0 < response.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_json_parsing_fallback_on_malformed_output() -> None:
    provider = ScriptedModelProvider(["sorry, I cannot produce JSON today"])
    agent = RequirementsExtractorAgent(
        provider=provider,
        metrics=MetricsTracker(),
        document_processor=LocalDocumentProcessor(),
        text_analyzer=LocalTextAnalyzer(),
    )
    response = await agent.process_document(str(SAMPLE_RFP), session_id="s-bad-json")
    assert response.output["total_extracted"] == 0
    assert response.confidence_score <= 0.35
