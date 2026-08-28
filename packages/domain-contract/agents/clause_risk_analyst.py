"""Clause Risk Analyst agent — chain-of-thought clause risk analysis."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry
from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import AgentResponse
from packages.domain_contract.prompts.analyst_prompt import (
    get_analyst_prompt,
    get_analyst_user_prompt,
)
from packages.domain_contract.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_contract.clause_risk_analyst")

AGENT_NAME = "clause_risk_analyst"

_VALID_SEVERITIES = {"critical", "high", "medium", "low"}


class ClauseRiskAnalystAgent:
    """Identifies clause-level risks with explicit reasoning traces."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize the clause risk analyst.

        Args:
            provider: LLM provider for analysis.
            metrics: Metrics tracker for token/cost accounting.
            audit_logger: Optional insert-only audit logger.
        """
        self._provider = provider
        self._metrics = metrics
        self._audit = audit_logger

    async def analyze_risks(
        self,
        contract_text: str,
        session_id: str = "",
    ) -> AgentResponse:
        """Analyze contract clauses for risk using chain-of-thought prompting.

        Args:
            contract_text: Full contract document text.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing clause risk findings.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        system_prompt = get_analyst_prompt()
        user_prompt = get_analyst_user_prompt(contract_text=contract_text)
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

        output, confidence = self._parse_risks(model_response.content)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        response = AgentResponse(
            task_id=task_id,
            agent_name=AGENT_NAME,
            output=output,
            confidence_score=confidence,
            token_usage=model_response.token_usage,
            processing_time_ms=elapsed_ms,
        )
        self._maybe_audit(
            session_id=session_id,
            input_payload={"document_chars": len(contract_text)},
            response=response,
        )
        logger.info(
            "clause_risk_analysis_complete",
            session_id=session_id,
            risk_count=len(output.get("risks", [])),
            confidence=confidence,
            processing_time_ms=elapsed_ms,
        )
        return response

    def _maybe_audit(
        self,
        *,
        session_id: str,
        input_payload: dict[str, Any],
        response: AgentResponse,
    ) -> None:
        if self._audit is None:
            return
        self._audit.log(
            AuditEntry(
                session_id=session_id,
                agent_name=AGENT_NAME,
                action_type=ActionType.INVOKE,
                input_payload=input_payload,
                output_payload=response.output,
                confidence_score=response.confidence_score,
                token_count=(
                    response.token_usage.input_tokens + response.token_usage.output_tokens
                ),
                cost_usd=response.token_usage.estimated_cost_usd,
            )
        )

    def _parse_risks(self, content: str) -> tuple[dict[str, Any], float]:
        """Parse risk JSON and normalize severities / confidence."""
        try:
            payload = extract_json_object(content)
            risks_raw = payload.get("risks", [])
            risks: list[dict[str, Any]] = []
            if isinstance(risks_raw, list):
                for item in risks_raw:
                    if not isinstance(item, dict):
                        continue
                    severity = str(item.get("risk_severity") or "medium").lower()
                    if severity not in _VALID_SEVERITIES:
                        severity = "medium"
                    risks.append(
                        {
                            "clause_id": str(item.get("clause_id") or ""),
                            "clause_text": str(item.get("clause_text") or ""),
                            "risk_severity": severity,
                            "risk_category": str(item.get("risk_category") or ""),
                            "confidence_score": clamp_confidence(
                                float(item.get("confidence_score") or 0.0)
                            ),
                            "reasoning_trace": str(item.get("reasoning_trace") or ""),
                            "source_section": str(item.get("source_section") or ""),
                            "mitigation_suggestion": str(
                                item.get("mitigation_suggestion") or ""
                            ),
                        }
                    )
            critical = sum(1 for r in risks if r["risk_severity"] == "critical")
            high = sum(1 for r in risks if r["risk_severity"] == "high")
            medium = sum(1 for r in risks if r["risk_severity"] == "medium")
            low = sum(1 for r in risks if r["risk_severity"] == "low")
            overall = str(payload.get("overall_risk_level") or "low").lower()
            if overall not in _VALID_SEVERITIES:
                if critical:
                    overall = "critical"
                elif high:
                    overall = "high"
                elif medium:
                    overall = "medium"
                else:
                    overall = "low"
            if risks:
                confidence = clamp_confidence(
                    sum(float(r["confidence_score"]) for r in risks) / len(risks)
                )
            else:
                confidence = 0.3
            output = {
                "risks": risks,
                "critical_count": int(payload.get("critical_count") or critical),
                "high_count": int(payload.get("high_count") or high),
                "medium_count": int(payload.get("medium_count") or medium),
                "low_count": int(payload.get("low_count") or low),
                "overall_risk_level": overall,
                "summary": str(payload.get("summary") or ""),
            }
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("clause_risk_json_parse_failed", error=str(exc))
            return (
                {
                    "risks": [],
                    "critical_count": 0,
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": 0,
                    "overall_risk_level": "medium",
                    "summary": "Unable to extract risks due to malformed model output.",
                    "raw_response": content,
                },
                0.2,
            )
