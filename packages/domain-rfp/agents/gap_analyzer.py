"""Gap Analyzer agent — chain-of-thought prompting."""

from __future__ import annotations

import re
import time
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import AgentResponse
from packages.domain_rfp.models import GapAnalysisResult, GapAssessment, MappingResult
from packages.domain_rfp.prompts.analyzer_prompt import (
    get_analyzer_prompt,
    get_analyzer_user_prompt,
)
from packages.domain_rfp.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_rfp.gap_analyzer")

AGENT_NAME = "gap_analyzer"


class GapAnalyzerAgent:
    """Analyzes capability gaps with explicit chain-of-thought reasoning."""

    def __init__(self, provider: ModelProvider, metrics: MetricsTracker) -> None:
        """Initialize the gap analyzer.

        Args:
            provider: LLM provider for analysis.
            metrics: Metrics tracker for token/cost accounting.
        """
        self._provider = provider
        self._metrics = metrics

    async def analyze_gaps(
        self,
        mappings: MappingResult,
        session_id: str,
    ) -> AgentResponse:
        """Analyze partial and unmatched mappings for bid risk.

        Args:
            mappings: Capability mapping result from the mapper agent.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing a GapAnalysisResult payload.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        gap_mappings = [
            m.model_dump()
            for m in mappings.mappings
            if m.match_level in {"partial", "none"}
        ]

        if not gap_mappings:
            result = GapAnalysisResult(
                assessments=[],
                critical_gaps=0,
                high_gaps=0,
                medium_gaps=0,
                low_gaps=0,
                overall_risk="acceptable",
                summary="No partial or unmatched requirements; gaps are minimal.",
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return AgentResponse(
                task_id=task_id,
                agent_name=AGENT_NAME,
                output=result.model_dump(),
                confidence_score=0.85,
                processing_time_ms=elapsed_ms,
            )

        system_prompt = get_analyzer_prompt()
        user_prompt = get_analyzer_user_prompt(gap_mappings)
        prompt = f"{system_prompt}\n\n{user_prompt}"

        model_response = await self._provider.invoke(
            prompt=prompt,
            temperature=0.2,
            max_tokens=4096,
            agent_name=AGENT_NAME,
            session_id=session_id,
        )
        self._metrics.track_llm_call(
            agent_name=AGENT_NAME,
            model_id=model_response.model_id,
            input_tokens=model_response.token_usage.input_tokens,
            output_tokens=model_response.token_usage.output_tokens,
            latency_ms=model_response.latency_ms,
            session_id=session_id,
        )

        analysis, parse_ok = self._parse_analysis(model_response.content)
        analysis = self._enrich_counts_and_risk(analysis)
        if not analysis.summary:
            analysis.summary = self._generate_summary(analysis)

        confidence = self._calculate_confidence(analysis, parse_ok=parse_ok)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "gap_analysis_complete",
            session_id=session_id,
            assessment_count=len(analysis.assessments),
            overall_risk=analysis.overall_risk,
            confidence=confidence,
        )
        return AgentResponse(
            task_id=task_id,
            agent_name=AGENT_NAME,
            output=analysis.model_dump(),
            confidence_score=confidence,
            token_usage=model_response.token_usage,
            processing_time_ms=elapsed_ms,
        )

    def _parse_analysis(self, content: str) -> tuple[GapAnalysisResult, bool]:
        try:
            payload = extract_json_object(content)
            assessments_raw = payload.get("assessments", [])
            assessments: list[GapAssessment] = []
            if isinstance(assessments_raw, list):
                for item in assessments_raw:
                    if not isinstance(item, dict):
                        continue
                    assessments.append(
                        GapAssessment(
                            requirement_id=str(item.get("requirement_id", "")),
                            requirement_text=str(item.get("requirement_text", "")),
                            gap_description=str(item.get("gap_description", "")),
                            risk_severity=str(item.get("risk_severity", "medium")),
                            reasoning=str(item.get("reasoning", "")),
                            mitigation_options=[
                                str(opt) for opt in (item.get("mitigation_options") or [])
                            ],
                            recommendation=str(item.get("recommendation", "respond")),
                        )
                    )
            result = GapAnalysisResult(
                assessments=assessments,
                critical_gaps=int(payload.get("critical_gaps") or 0),
                high_gaps=int(payload.get("high_gaps") or 0),
                medium_gaps=int(payload.get("medium_gaps") or 0),
                low_gaps=int(payload.get("low_gaps") or 0),
                overall_risk=str(payload.get("overall_risk") or "manageable"),
                summary=str(payload.get("summary") or ""),
            )
            return result, True
        except Exception as exc:  # noqa: BLE001
            logger.error("gap_analysis_json_parse_failed", error=str(exc))
            return (
                GapAnalysisResult(
                    assessments=[],
                    overall_risk="high-risk",
                    summary="Gap analysis failed due to malformed model output.",
                ),
                False,
            )

    def _enrich_counts_and_risk(self, analysis: GapAnalysisResult) -> GapAnalysisResult:
        analysis.critical_gaps = sum(
            1 for a in analysis.assessments if a.risk_severity == "critical"
        )
        analysis.high_gaps = sum(
            1 for a in analysis.assessments if a.risk_severity == "high"
        )
        analysis.medium_gaps = sum(
            1 for a in analysis.assessments if a.risk_severity == "medium"
        )
        analysis.low_gaps = sum(
            1 for a in analysis.assessments if a.risk_severity == "low"
        )
        analysis.overall_risk = self._calculate_overall_risk(analysis)
        return analysis

    def _calculate_overall_risk(self, analysis: GapAnalysisResult) -> str:
        if analysis.critical_gaps > 0:
            return "no-bid"
        if analysis.high_gaps >= 2:
            return "high-risk"
        if analysis.high_gaps == 1 or analysis.medium_gaps >= 3:
            return "manageable"
        return "acceptable"

    def _generate_summary(self, analysis: GapAnalysisResult) -> str:
        return (
            f"Identified {len(analysis.assessments)} gaps "
            f"({analysis.critical_gaps} critical, {analysis.high_gaps} high, "
            f"{analysis.medium_gaps} medium, {analysis.low_gaps} low). "
            f"Overall bid risk assessed as {analysis.overall_risk}."
        )

    def _calculate_confidence(
        self,
        analysis: GapAnalysisResult,
        *,
        parse_ok: bool,
    ) -> float:
        if not parse_ok:
            return 0.2
        if not analysis.assessments:
            return 0.75

        substantive = 0
        for assessment in analysis.assessments:
            sentences = [
                s.strip()
                for s in re.split(r"[.!?]\s+", assessment.reasoning)
                if s.strip()
            ]
            has_step_refs = bool(
                re.search(r"\b(step\s*[1-6]|1\.|2\.|3\.|4\.|5\.|6\.)\b", assessment.reasoning, re.I)
            )
            if len(sentences) >= 2 and has_step_refs:
                substantive += 1
            elif len(sentences) >= 3:
                substantive += 1

        ratio = substantive / len(analysis.assessments)
        return clamp_confidence(0.45 + (0.5 * ratio))
