"""Pricing Analyst agent — zero-shot structured pricing extraction."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import AgentResponse
from packages.domain_vendor.prompts.analyst_prompt import (
    get_pricing_analyst_prompt,
    get_pricing_analyst_user_prompt,
)
from packages.domain_vendor.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_vendor.pricing_analyst")

AGENT_NAME = "pricing_analyst"


class PricingAnalystAgent:
    """Analyzes vendor pricing structures with low-temperature structured output."""

    def __init__(self, provider: ModelProvider, metrics: MetricsTracker) -> None:
        """Initialize the pricing analyst.

        Args:
            provider: LLM provider for analysis.
            metrics: Metrics tracker for token/cost accounting.
        """
        self._provider = provider
        self._metrics = metrics

    async def analyze_pricing(
        self,
        vendor_name: str,
        vendor_document: str,
        session_id: str,
    ) -> AgentResponse:
        """Extract and analyze pricing from a vendor document.

        Args:
            vendor_name: Vendor being analyzed.
            vendor_document: Source vendor profile or document text.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing a pricing analysis payload.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        system_prompt = get_pricing_analyst_prompt()
        user_prompt = get_pricing_analyst_user_prompt(
            vendor_name=vendor_name,
            vendor_document=vendor_document,
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

        output, confidence = self._parse_pricing(
            content=model_response.content,
            vendor_name=vendor_name,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "pricing_analysis_complete",
            session_id=session_id,
            vendor_name=vendor_name,
            pricing_model=output.get("pricing_model"),
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

    def _parse_pricing(
        self,
        content: str,
        vendor_name: str,
    ) -> tuple[dict[str, Any], float]:
        """Parse pricing JSON and clamp confidence."""
        try:
            payload = extract_json_object(content)
            breakdown_raw = payload.get("cost_breakdown", [])
            cost_breakdown: list[dict[str, Any]] = []
            if isinstance(breakdown_raw, list):
                for item in breakdown_raw:
                    if not isinstance(item, dict):
                        continue
                    cost_breakdown.append(
                        {
                            "item": str(item.get("item") or ""),
                            "cost": str(item.get("cost") or ""),
                            "notes": str(item.get("notes") or ""),
                        }
                    )
            risk_flags = payload.get("risk_flags", [])
            if not isinstance(risk_flags, list):
                risk_flags = []
            confidence = clamp_confidence(float(payload.get("confidence_score") or 0.0))
            output = {
                "vendor_name": str(payload.get("vendor_name") or vendor_name),
                "pricing_model": str(payload.get("pricing_model") or ""),
                "estimated_annual_cost": str(payload.get("estimated_annual_cost") or ""),
                "cost_breakdown": cost_breakdown,
                "risk_flags": [str(flag) for flag in risk_flags],
                "comparison_notes": str(payload.get("comparison_notes") or ""),
                "confidence_score": confidence,
            }
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("pricing_analysis_json_parse_failed", error=str(exc))
            return (
                {
                    "vendor_name": vendor_name,
                    "pricing_model": "",
                    "estimated_annual_cost": "",
                    "cost_breakdown": [],
                    "risk_flags": ["JSON parse failure"],
                    "comparison_notes": "Unable to analyze pricing due to malformed model output.",
                    "confidence_score": 0.2,
                    "raw_response": content,
                },
                0.2,
            )
