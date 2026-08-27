"""Market Positioner agent — chain-of-thought competitive positioning."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import AgentResponse
from packages.domain_vendor.prompts.positioner_prompt import (
    get_market_positioner_prompt,
    get_market_positioner_user_prompt,
)
from packages.domain_vendor.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_vendor.market_positioner")

AGENT_NAME = "market_positioner"


class MarketPositionerAgent:
    """Assesses competitive positioning using higher-temperature CoT reasoning."""

    def __init__(self, provider: ModelProvider, metrics: MetricsTracker) -> None:
        """Initialize the market positioner.

        Args:
            provider: LLM provider for analysis.
            metrics: Metrics tracker for token/cost accounting.
        """
        self._provider = provider
        self._metrics = metrics

    async def position_vendor(
        self,
        vendor_name: str,
        vendor_document: str,
        capability_summary: str,
        pricing_summary: str,
        session_id: str,
    ) -> AgentResponse:
        """Produce a strategic market-position assessment.

        Args:
            vendor_name: Vendor being positioned.
            vendor_document: Source vendor profile or document text.
            capability_summary: Summary from the capability researcher.
            pricing_summary: Summary from the pricing analyst.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing a market-position payload.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        system_prompt = get_market_positioner_prompt()
        user_prompt = get_market_positioner_user_prompt(
            vendor_name=vendor_name,
            vendor_document=vendor_document,
            capability_summary=capability_summary,
            pricing_summary=pricing_summary,
        )
        prompt = f"{system_prompt}\n\n{user_prompt}"

        model_response = await self._provider.invoke(
            prompt=prompt,
            temperature=0.7,
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

        output, confidence = self._parse_position(
            content=model_response.content,
            vendor_name=vendor_name,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "market_positioning_complete",
            session_id=session_id,
            vendor_name=vendor_name,
            market_segment=output.get("market_segment"),
            confidence=confidence,
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

    def _as_str_list(self, value: Any) -> list[str]:
        """Coerce a JSON field into a list of strings."""
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def _parse_position(
        self,
        content: str,
        vendor_name: str,
    ) -> tuple[dict[str, Any], float]:
        """Parse positioning JSON and clamp confidence."""
        try:
            payload = extract_json_object(content)
            confidence = clamp_confidence(float(payload.get("confidence_score") or 0.0))
            output = {
                "vendor_name": str(payload.get("vendor_name") or vendor_name),
                "reasoning": str(payload.get("reasoning") or ""),
                "market_segment": str(payload.get("market_segment") or ""),
                "differentiators": self._as_str_list(payload.get("differentiators")),
                "competitive_advantages": self._as_str_list(
                    payload.get("competitive_advantages")
                ),
                "competitive_risks": self._as_str_list(payload.get("competitive_risks")),
                "strategic_assessment": str(payload.get("strategic_assessment") or ""),
                "confidence_score": confidence,
            }
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("market_positioning_json_parse_failed", error=str(exc))
            return (
                {
                    "vendor_name": vendor_name,
                    "reasoning": "",
                    "market_segment": "",
                    "differentiators": [],
                    "competitive_advantages": [],
                    "competitive_risks": [],
                    "strategic_assessment": (
                        "Unable to assess market position due to malformed model output."
                    ),
                    "confidence_score": 0.2,
                    "raw_response": content,
                },
                0.2,
            )
