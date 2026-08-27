"""Capability Researcher agent — few-shot prompting with optional RAG context."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.rag.retriever import RAGRetriever
from packages.core.types.schemas import AgentResponse
from packages.domain_vendor.prompts.researcher_prompt import (
    get_capability_researcher_prompt,
    get_capability_researcher_user_prompt,
)
from packages.domain_vendor.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_vendor.capability_researcher")

AGENT_NAME = "capability_researcher"


class CapabilityResearcherAgent:
    """Researches vendor capabilities from documents, optionally using RAG."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        rag_retriever: RAGRetriever | None = None,
    ) -> None:
        """Initialize the capability researcher.

        Args:
            provider: LLM provider for extraction.
            metrics: Metrics tracker for token/cost accounting.
            rag_retriever: Optional retriever for vendor-domain knowledge chunks.
        """
        self._provider = provider
        self._metrics = metrics
        self._rag_retriever = rag_retriever

    async def research_capabilities(
        self,
        vendor_name: str,
        vendor_document: str,
        session_id: str,
    ) -> AgentResponse:
        """Extract capabilities from a vendor document, injecting RAG context when available.

        Args:
            vendor_name: Vendor being researched.
            vendor_document: Source vendor profile or document text.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing capability assessments and optional retrieved chunks.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        rag_context = ""
        retrieved_chunks: list[dict[str, Any]] = []
        if self._rag_retriever is not None:
            rag_context = await self._rag_retriever.retrieve_formatted(
                query=f"capabilities of {vendor_name}",
                domain="vendor",
                top_k=5,
            )
            if rag_context:
                retrieved_chunks = [
                    {
                        "formatted_context": rag_context,
                        "query": f"capabilities of {vendor_name}",
                        "domain": "vendor",
                    }
                ]

        system_prompt = get_capability_researcher_prompt()
        user_prompt = get_capability_researcher_user_prompt(
            vendor_name=vendor_name,
            vendor_document=vendor_document,
            rag_context=rag_context,
        )
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

        output, confidence = self._parse_research(
            content=model_response.content,
            vendor_name=vendor_name,
        )
        output["retrieved_chunks"] = retrieved_chunks

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "capability_research_complete",
            session_id=session_id,
            vendor_name=vendor_name,
            capability_count=len(output.get("capabilities", [])),
            confidence=confidence,
            rag_used=self._rag_retriever is not None,
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

    def _parse_research(
        self,
        content: str,
        vendor_name: str,
    ) -> tuple[dict[str, Any], float]:
        """Parse researcher JSON and clamp per-capability confidence scores."""
        try:
            payload = extract_json_object(content)
            capabilities_raw = payload.get("capabilities", [])
            capabilities: list[dict[str, Any]] = []
            if isinstance(capabilities_raw, list):
                for item in capabilities_raw:
                    if not isinstance(item, dict):
                        continue
                    capabilities.append(
                        {
                            "capability_name": str(item.get("capability_name") or ""),
                            "evidence": str(item.get("evidence") or ""),
                            "confidence_score": clamp_confidence(
                                float(item.get("confidence_score") or 0.0)
                            ),
                            "source_citation": str(item.get("source_citation") or ""),
                        }
                    )
            if capabilities:
                confidence = clamp_confidence(
                    sum(float(item["confidence_score"]) for item in capabilities)
                    / len(capabilities)
                )
            else:
                confidence = 0.3
            output = {
                "vendor_name": str(payload.get("vendor_name") or vendor_name),
                "capabilities": capabilities,
                "summary": str(payload.get("summary") or ""),
            }
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("capability_research_json_parse_failed", error=str(exc))
            return (
                {
                    "vendor_name": vendor_name,
                    "capabilities": [],
                    "summary": "Unable to extract capabilities due to malformed model output.",
                    "raw_response": content,
                },
                0.2,
            )
