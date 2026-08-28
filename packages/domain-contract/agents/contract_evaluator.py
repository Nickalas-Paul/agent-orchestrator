"""Contract output evaluator — LLM-as-a-judge over specialist pipeline results."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import AgentResponse
from packages.domain_contract.prompts.evaluator_prompt import (
    DEFAULT_EVALUATION_CRITERIA,
    get_contract_evaluator_prompt,
    get_contract_evaluator_user_prompt,
)
from packages.domain_contract.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_contract.contract_evaluator")

AGENT_NAME = "contract_evaluator"


class ContractOutputEvaluator:
    """Evaluates combined contract-specialist outputs using LLM-as-a-judge."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        evaluation_criteria: list[str] | None = None,
    ) -> None:
        """Initialize the contract output evaluator.

        Args:
            provider: LLM provider for evaluation.
            metrics: Metrics tracker for token/cost accounting.
            evaluation_criteria: Optional criterion names; defaults to contract five-point list.
        """
        self._provider = provider
        self._metrics = metrics
        self._criteria = (
            list(evaluation_criteria)
            if evaluation_criteria
            else list(DEFAULT_EVALUATION_CRITERIA)
        )

    async def evaluate_outputs(
        self,
        risk_result: dict[str, Any],
        compliance_result: dict[str, Any],
        comparison_result: dict[str, Any],
        session_id: str = "",
    ) -> AgentResponse:
        """Evaluate combined specialist outputs and return a structured judgment.

        Args:
            risk_result: Clause risk analyst output.
            compliance_result: Compliance checker output.
            comparison_result: Terms comparator output.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing evaluation findings and recommendation.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        system_prompt = get_contract_evaluator_prompt(self._criteria)
        user_prompt = get_contract_evaluator_user_prompt(
            risk_result=risk_result,
            compliance_result=compliance_result,
            comparison_result=comparison_result,
        )
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

        output, confidence = self._parse_evaluation(model_response.content)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "contract_evaluation_complete",
            session_id=session_id,
            confidence=confidence,
            recommendation=output.get("recommendation"),
            processing_time_ms=elapsed_ms,
        )
        return AgentResponse(
            task_id=task_id,
            agent_name=AGENT_NAME,
            output=output,
            confidence_score=confidence,
            token_usage=model_response.token_usage,
            processing_time_ms=elapsed_ms,
        )

    def _parse_evaluation(self, content: str) -> tuple[dict[str, Any], float]:
        """Parse evaluator JSON, clamping overall_confidence into ``[0.0, 1.0]``."""
        try:
            payload = extract_json_object(content)
            confidence = clamp_confidence(float(payload.get("overall_confidence") or 0.0))
            findings = payload.get("findings", [])
            if not isinstance(findings, list):
                findings = []
            contradictions = payload.get("contradictions", [])
            if not isinstance(contradictions, list):
                contradictions = []
            criterion_scores_raw = payload.get("criterion_scores", {})
            criterion_scores: dict[str, float] = {}
            if isinstance(criterion_scores_raw, dict):
                for key, value in criterion_scores_raw.items():
                    try:
                        criterion_scores[str(key)] = clamp_confidence(float(value))
                    except (TypeError, ValueError):
                        continue
            recommendation = str(payload.get("recommendation") or "flag_for_review")
            if recommendation not in {"approve", "flag_for_review"}:
                recommendation = "flag_for_review"
            output = {
                "overall_confidence": confidence,
                "criterion_scores": criterion_scores,
                "findings": findings,
                "contradictions": contradictions,
                "summary": str(payload.get("summary") or ""),
                "recommendation": recommendation,
            }
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("contract_evaluation_json_parse_failed", error=str(exc))
            return (
                {
                    "overall_confidence": 0.3,
                    "criterion_scores": {},
                    "findings": [],
                    "contradictions": [],
                    "summary": "Evaluation failed due to malformed model output.",
                    "recommendation": "flag_for_review",
                    "raw_response": content,
                },
                0.3,
            )
