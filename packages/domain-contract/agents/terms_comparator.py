"""Terms Comparator agent — CoT comparison with optional RAG standard terms."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry
from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.rag.retriever import RAGRetriever
from packages.core.types.schemas import AgentResponse
from packages.domain_contract.prompts.comparator_prompt import (
    get_comparator_prompt,
    get_comparator_user_prompt,
)
from packages.domain_contract.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_contract.terms_comparator")

AGENT_NAME = "terms_comparator"

_VALID_DEVIATIONS = {
    "more_favorable",
    "less_favorable",
    "missing",
    "equivalent",
}


class TermsComparatorAgent:
    """Compares contract language to standard/preferred terms, optionally via RAG."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        rag_retriever: RAGRetriever | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize the terms comparator.

        Args:
            provider: LLM provider for comparison.
            metrics: Metrics tracker for token/cost accounting.
            rag_retriever: Optional retriever for standard contract terms.
            audit_logger: Optional insert-only audit logger.
        """
        self._provider = provider
        self._metrics = metrics
        self._rag_retriever = rag_retriever
        self._audit = audit_logger

    async def compare_terms(
        self,
        contract_text: str,
        contract_name: str = "",
        session_id: str = "",
    ) -> AgentResponse:
        """Compare contract terms against standard/preferred language.

        Args:
            contract_text: Full contract document text.
            contract_name: Contract label used for RAG query context.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing term deviations and optional retrieved chunks.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        rag_context = ""
        retrieved_chunks: list[dict[str, Any]] = []
        label = contract_name or "agreement"
        if self._rag_retriever is not None:
            rag_context = await self._rag_retriever.retrieve_formatted(
                query=f"standard contract terms for {label}",
                domain="contract",
                top_k=5,
            )
            if rag_context:
                retrieved_chunks = [
                    {
                        "formatted_context": rag_context,
                        "query": f"standard contract terms for {label}",
                        "domain": "contract",
                    }
                ]

        system_prompt = get_comparator_prompt()
        user_prompt = get_comparator_user_prompt(
            contract_text=contract_text,
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

        output, confidence = self._parse_comparison(model_response.content)
        output["retrieved_chunks"] = retrieved_chunks

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
            input_payload={
                "contract_name": contract_name,
                "document_chars": len(contract_text),
                "rag_used": self._rag_retriever is not None,
            },
            response=response,
        )
        logger.info(
            "terms_comparison_complete",
            session_id=session_id,
            contract_name=contract_name,
            deviation_count=len(output.get("deviations", [])),
            confidence=confidence,
            rag_used=self._rag_retriever is not None,
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

    def _parse_comparison(self, content: str) -> tuple[dict[str, Any], float]:
        """Parse comparison JSON and normalize deviation types / scores."""
        try:
            payload = extract_json_object(content)
            deviations_raw = payload.get("deviations", [])
            deviations: list[dict[str, Any]] = []
            if isinstance(deviations_raw, list):
                for item in deviations_raw:
                    if not isinstance(item, dict):
                        continue
                    dtype = str(item.get("deviation_type") or "equivalent").lower()
                    if dtype not in _VALID_DEVIATIONS:
                        dtype = "equivalent"
                    deviations.append(
                        {
                            "term_name": str(item.get("term_name") or ""),
                            "standard_language": str(item.get("standard_language") or ""),
                            "contract_language": str(item.get("contract_language") or ""),
                            "deviation_type": dtype,
                            "impact_assessment": str(item.get("impact_assessment") or ""),
                            "confidence_score": clamp_confidence(
                                float(item.get("confidence_score") or 0.0)
                            ),
                        }
                    )
            more_fav = sum(1 for d in deviations if d["deviation_type"] == "more_favorable")
            less_fav = sum(1 for d in deviations if d["deviation_type"] == "less_favorable")
            missing = sum(1 for d in deviations if d["deviation_type"] == "missing")
            equivalent = sum(1 for d in deviations if d["deviation_type"] == "equivalent")
            deviation_score = clamp_confidence(
                float(payload.get("overall_deviation_score") or 0.0)
            )
            if deviations:
                confidence = clamp_confidence(
                    sum(float(d["confidence_score"]) for d in deviations) / len(deviations)
                )
            else:
                confidence = 0.3
            output = {
                "deviations": deviations,
                "more_favorable_count": int(payload.get("more_favorable_count") or more_fav),
                "less_favorable_count": int(payload.get("less_favorable_count") or less_fav),
                "missing_count": int(payload.get("missing_count") or missing),
                "equivalent_count": int(payload.get("equivalent_count") or equivalent),
                "overall_deviation_score": deviation_score,
                "summary": str(payload.get("summary") or ""),
            }
            return output, confidence
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("terms_comparison_json_parse_failed", error=str(exc))
            return (
                {
                    "deviations": [],
                    "more_favorable_count": 0,
                    "less_favorable_count": 0,
                    "missing_count": 0,
                    "equivalent_count": 0,
                    "overall_deviation_score": 0.5,
                    "summary": "Unable to compare terms due to malformed model output.",
                    "raw_response": content,
                },
                0.2,
            )
