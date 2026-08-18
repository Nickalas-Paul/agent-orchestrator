"""Capability Mapper agent — few-shot prompting."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.types.schemas import AgentResponse
from packages.domain_rfp.models import CapabilityMapping, MappingResult, RfpRequirement
from packages.domain_rfp.prompts.mapper_prompt import (
    DEFAULT_CAPABILITIES,
    get_mapper_prompt,
    get_mapper_user_prompt,
)
from packages.domain_rfp.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_rfp.capability_mapper")

AGENT_NAME = "capability_mapper"

_PRIORITY_WEIGHT = {
    "must-have": 2.0,
    "should-have": 1.0,
    "nice-to-have": 0.5,
}


class CapabilityMapperAgent:
    """Maps extracted RFP requirements to known enterprise capabilities."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        capabilities: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the capability mapper.

        Args:
            provider: LLM provider for mapping.
            metrics: Metrics tracker for token/cost accounting.
            capabilities: Optional capability knowledge base override.
        """
        self._provider = provider
        self._metrics = metrics
        self._capabilities = capabilities or list(DEFAULT_CAPABILITIES)

    async def map_requirements(
        self,
        requirements: list[RfpRequirement],
        session_id: str,
    ) -> AgentResponse:
        """Map requirements to capabilities using few-shot prompting.

        Args:
            requirements: Extracted RFP requirements.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing a MappingResult payload.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        system_prompt = get_mapper_prompt()
        user_prompt = get_mapper_user_prompt(
            requirements=[r.model_dump() for r in requirements],
            capabilities=self._capabilities,
        )
        prompt = f"{system_prompt}\n\n{user_prompt}"

        model_response = await self._provider.invoke(
            prompt=prompt,
            temperature=0.4,
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

        mapping_result, parse_ok = self._parse_mapping(
            content=model_response.content,
            requirements=requirements,
        )
        confidence = self._calculate_confidence(
            mapping_result=mapping_result,
            requirements=requirements,
            parse_ok=parse_ok,
        )
        mapping_result.overall_confidence = confidence
        mapping_result.fully_matched = sum(
            1 for m in mapping_result.mappings if m.match_level == "full"
        )
        mapping_result.partially_matched = sum(
            1 for m in mapping_result.mappings if m.match_level == "partial"
        )
        mapping_result.unmatched = sum(
            1 for m in mapping_result.mappings if m.match_level == "none"
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "capability_mapping_complete",
            session_id=session_id,
            fully_matched=mapping_result.fully_matched,
            partially_matched=mapping_result.partially_matched,
            unmatched=mapping_result.unmatched,
            confidence=confidence,
        )
        return AgentResponse(
            task_id=task_id,
            agent_name=AGENT_NAME,
            output=mapping_result.model_dump(),
            confidence_score=confidence,
            token_usage=model_response.token_usage,
            processing_time_ms=elapsed_ms,
        )

    def _parse_mapping(
        self,
        content: str,
        requirements: list[RfpRequirement],
    ) -> tuple[MappingResult, bool]:
        try:
            payload = extract_json_object(content)
            mappings_raw = payload.get("mappings", [])
            mappings: list[CapabilityMapping] = []
            if isinstance(mappings_raw, list):
                for item in mappings_raw:
                    if not isinstance(item, dict):
                        continue
                    mappings.append(
                        CapabilityMapping(
                            requirement_id=str(item.get("requirement_id", "")),
                            requirement_text=str(item.get("requirement_text", "")),
                            match_level=str(item.get("match_level", "none")),
                            capability=item.get("capability"),
                            response_draft=str(item.get("response_draft", "")),
                            confidence=float(item.get("confidence", 0.0) or 0.0),
                            gap_note=item.get("gap_note"),
                        )
                    )
            return MappingResult(mappings=mappings), True
        except Exception as exc:  # noqa: BLE001
            logger.error("mapping_json_parse_failed", error=str(exc))
            fallback = [
                CapabilityMapping(
                    requirement_id=req.requirement_id,
                    requirement_text=req.text,
                    match_level="none",
                    capability=None,
                    response_draft="Unable to map due to malformed model output.",
                    confidence=0.1,
                    gap_note="JSON parse failure",
                )
                for req in requirements
            ]
            return MappingResult(mappings=fallback), False

    def _calculate_confidence(
        self,
        mapping_result: MappingResult,
        requirements: list[RfpRequirement],
        parse_ok: bool,
    ) -> float:
        if not parse_ok:
            return 0.2
        if not mapping_result.mappings:
            return 0.3

        priority_by_id = {r.requirement_id: r.priority for r in requirements}
        weighted_sum = 0.0
        weight_total = 0.0
        for mapping in mapping_result.mappings:
            priority = priority_by_id.get(mapping.requirement_id, "should-have")
            weight = _PRIORITY_WEIGHT.get(priority, 1.0)
            weighted_sum += clamp_confidence(mapping.confidence) * weight
            weight_total += weight
        if weight_total <= 0:
            return 0.3
        return clamp_confidence(weighted_sum / weight_total)
