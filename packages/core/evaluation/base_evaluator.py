"""Shared LLM-as-a-judge evaluator skeleton for domain packages."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import AgentResponse
from packages.core.utils import clamp_confidence, extract_json_object


class BaseOutputEvaluator(ABC):
    """Shared LLM-as-a-judge evaluator skeleton.

    Subclasses override:
    - ``AGENT_NAME`` — logger/metrics identifier
    - ``DEFAULT_CRITERIA`` — domain-specific evaluation criteria
    - ``_build_messages`` — prompt construction from specialist outputs
    - ``_post_parse`` (optional) — domain-specific enrichment (e.g. criterion_scores)
    """

    AGENT_NAME: str = "base_evaluator"
    DEFAULT_CRITERIA: list[str] = []

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        evaluation_criteria: list[str] | None = None,
    ) -> None:
        """Initialize the shared evaluator.

        Args:
            provider: LLM provider for evaluation.
            metrics: Metrics tracker for token/cost accounting.
            evaluation_criteria: Optional criterion names; defaults to ``DEFAULT_CRITERIA``.
        """
        self._provider = provider
        self._metrics = metrics
        self._criteria = (
            list(evaluation_criteria)
            if evaluation_criteria
            else list(self.DEFAULT_CRITERIA)
        )
        self._logger = get_logger(self.AGENT_NAME)

    @abstractmethod
    def _build_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """Build system + user messages for the evaluation prompt.

        Returns:
            Message dicts: ``[{"role": "system", "content": ...}, {"role": "user", ...}]``.
        """

    def _post_parse(
        self, result: dict[str, Any], confidence: float
    ) -> tuple[dict[str, Any], float]:
        """Optional hook for subclass-specific post-parse enrichment.

        Default is identity (no-op). Contract evaluator clamps ``criterion_scores``.
        """
        return result, confidence

    async def evaluate(
        self, *, session_id: str = "", **specialist_outputs: Any
    ) -> AgentResponse:
        """Run LLM-as-a-judge evaluation on specialist outputs.

        Args:
            session_id: Session identifier for metrics tracking.
            **specialist_outputs: Domain-specific specialist result dicts.

        Returns:
            AgentResponse with evaluation result and confidence score.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        messages = self._build_messages(**specialist_outputs)
        prompt = "\n\n".join(message["content"] for message in messages)

        model_response = await self._provider.invoke(
            prompt=prompt,
            temperature=0.2,
            max_tokens=4096,
            agent_name=self.AGENT_NAME,
            session_id=session_id,
        )
        self._metrics.track_llm_call(
            agent_name=self.AGENT_NAME,
            model_id=model_response.model_id,
            input_tokens=model_response.token_usage.input_tokens,
            output_tokens=model_response.token_usage.output_tokens,
            latency_ms=model_response.latency_ms,
            session_id=session_id,
        )

        output, confidence = self._parse_evaluation(model_response.content)
        output, confidence = self._post_parse(output, confidence)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        self._logger.info(
            f"{self.AGENT_NAME}_evaluation_complete",
            session_id=session_id,
            confidence=confidence,
            parse_ok="parse_error" not in output,
            recommendation=output.get("recommendation"),
            processing_time_ms=elapsed_ms,
        )
        return AgentResponse(
            task_id=task_id,
            agent_name=self.AGENT_NAME,
            output=output,
            confidence_score=confidence,
            token_usage=model_response.token_usage,
            processing_time_ms=elapsed_ms,
        )

    def _parse_evaluation(self, content: str) -> tuple[dict[str, Any], float]:
        """Parse evaluator JSON, clamping overall_confidence into ``[0.0, 1.0]``.

        Returns:
            ``(result_dict, confidence)``. On failure confidence is ``0.3`` and the
            result includes ``raw_response`` / ``parse_error``.
        """
        try:
            payload = extract_json_object(content)
            confidence = clamp_confidence(float(payload.get("overall_confidence") or 0.0))
            findings = payload.get("findings", [])
            if not isinstance(findings, list):
                findings = []
            contradictions = payload.get("contradictions", [])
            if not isinstance(contradictions, list):
                contradictions = []
            recommendation = str(payload.get("recommendation") or "flag_for_review")
            if recommendation not in {"approve", "flag_for_review"}:
                recommendation = "flag_for_review"
            output: dict[str, Any] = {
                "overall_confidence": confidence,
                "findings": findings,
                "contradictions": contradictions,
                "summary": str(payload.get("summary") or ""),
                "recommendation": recommendation,
            }
            if "criterion_scores" in payload:
                output["criterion_scores"] = payload.get("criterion_scores")
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            self._logger.error(
                f"{self.AGENT_NAME}_evaluation_json_parse_failed",
                error=str(exc),
            )
            return (
                {
                    "overall_confidence": 0.3,
                    "findings": [],
                    "contradictions": [],
                    "summary": "Evaluation failed due to malformed model output.",
                    "recommendation": "flag_for_review",
                    "raw_response": content,
                    "parse_error": str(exc),
                },
                0.3,
            )
