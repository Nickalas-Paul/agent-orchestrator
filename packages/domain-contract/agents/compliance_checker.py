"""Compliance Checker agent — few-shot compliance gap analysis."""

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
from packages.domain_contract.prompts.checker_prompt import (
    get_checker_prompt,
    get_checker_user_prompt,
)
from packages.domain_contract.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_contract.compliance_checker")

AGENT_NAME = "compliance_checker"

_VALID_STATUSES = {
    "compliant",
    "non_compliant",
    "partially_compliant",
    "missing",
}
_VALID_SEVERITIES = {"critical", "high", "medium", "low"}


class ComplianceCheckerAgent:
    """Checks contracts against regulatory and completeness requirements."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize the compliance checker.

        Args:
            provider: LLM provider for analysis.
            metrics: Metrics tracker for token/cost accounting.
            audit_logger: Optional insert-only audit logger.
        """
        self._provider = provider
        self._metrics = metrics
        self._audit = audit_logger

    async def check_compliance(
        self,
        contract_text: str,
        session_id: str = "",
    ) -> AgentResponse:
        """Check contract compliance using few-shot prompting.

        Args:
            contract_text: Full contract document text.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing compliance gap findings.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        system_prompt = get_checker_prompt()
        user_prompt = get_checker_user_prompt(contract_text=contract_text)
        prompt = f"{system_prompt}\n\n{user_prompt}"

        model_response = await self._provider.invoke(
            prompt=prompt,
            temperature=0.3,
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

        output, confidence = self._parse_compliance(model_response.content)
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
            "compliance_check_complete",
            session_id=session_id,
            gap_count=len(output.get("gaps", [])),
            score=output.get("overall_compliance_score"),
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

    def _parse_compliance(self, content: str) -> tuple[dict[str, Any], float]:
        """Parse compliance JSON and normalize statuses / scores."""
        try:
            payload = extract_json_object(content)
            gaps_raw = payload.get("gaps", [])
            gaps: list[dict[str, Any]] = []
            if isinstance(gaps_raw, list):
                for item in gaps_raw:
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status") or "missing").lower()
                    if status not in _VALID_STATUSES:
                        status = "missing"
                    severity = str(item.get("severity") or "medium").lower()
                    if severity not in _VALID_SEVERITIES:
                        severity = "medium"
                    gaps.append(
                        {
                            "requirement": str(item.get("requirement") or ""),
                            "status": status,
                            "clause_reference": str(item.get("clause_reference") or ""),
                            "explanation": str(item.get("explanation") or ""),
                            "severity": severity,
                        }
                    )
            compliant = sum(1 for g in gaps if g["status"] == "compliant")
            non_compliant = sum(1 for g in gaps if g["status"] == "non_compliant")
            partial = sum(1 for g in gaps if g["status"] == "partially_compliant")
            missing = sum(1 for g in gaps if g["status"] == "missing")
            score = clamp_confidence(float(payload.get("overall_compliance_score") or 0.0))
            if gaps and score == 0.0 and compliant:
                score = clamp_confidence(compliant / len(gaps))
            confidence = score if gaps else 0.3
            output = {
                "gaps": gaps,
                "compliant_count": int(payload.get("compliant_count") or compliant),
                "non_compliant_count": int(payload.get("non_compliant_count") or non_compliant),
                "partially_compliant_count": int(
                    payload.get("partially_compliant_count") or partial
                ),
                "missing_count": int(payload.get("missing_count") or missing),
                "overall_compliance_score": score,
                "summary": str(payload.get("summary") or ""),
            }
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("compliance_json_parse_failed", error=str(exc))
            return (
                {
                    "gaps": [],
                    "compliant_count": 0,
                    "non_compliant_count": 0,
                    "partially_compliant_count": 0,
                    "missing_count": 0,
                    "overall_compliance_score": 0.2,
                    "summary": "Unable to check compliance due to malformed model output.",
                    "raw_response": content,
                },
                0.2,
            )
