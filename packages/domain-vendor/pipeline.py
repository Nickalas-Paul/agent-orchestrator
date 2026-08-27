"""End-to-end vendor evaluation pipeline wiring specialist agents, evaluator, and HITL."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry, HITLStatus
from packages.core.cloud.base import ModelProvider
from packages.core.hitl.manager import HITLManager
from packages.core.hitl.models import JobStatus, PipelineJob
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.prompts.registry import PromptRegistry
from packages.core.rag.retriever import RAGRetriever
from packages.core.types.schemas import AgentResponse, BusinessMetrics
from packages.domain_vendor.agents.capability_researcher import CapabilityResearcherAgent
from packages.domain_vendor.agents.market_positioner import MarketPositionerAgent
from packages.domain_vendor.agents.pricing_analyst import PricingAnalystAgent
from packages.domain_vendor.agents.vendor_evaluator import VendorOutputEvaluator
from packages.domain_vendor.models import VendorEvaluationResult, VendorPipelineStatus
from packages.domain_vendor.prompts.analyst_prompt import get_pricing_analyst_prompt
from packages.domain_vendor.prompts.evaluator_prompt import get_vendor_evaluator_prompt
from packages.domain_vendor.prompts.positioner_prompt import get_market_positioner_prompt
from packages.domain_vendor.prompts.researcher_prompt import get_capability_researcher_prompt

logger = get_logger("domain_vendor.pipeline")


class VendorEvaluationPipeline:
    """Runs research → price → position → evaluate for a vendor document."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        rag_retriever: RAGRetriever | None = None,
        audit_logger: AuditLogger | None = None,
        hitl_manager: HITLManager | None = None,
        confidence_threshold: float = 0.85,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        """Initialize the pipeline and optionally register prompt versions.

        Args:
            provider: Shared model provider for all agents.
            metrics: Shared metrics tracker.
            rag_retriever: Optional RAG retriever passed to the capability researcher.
            audit_logger: Optional insert-only audit logger. When omitted, DB writes are skipped.
            hitl_manager: Optional HITL job manager. When omitted, jobs are not persisted.
            confidence_threshold: Evaluator confidence required to skip human review.
            prompt_registry: Optional prompt registry for version tracking.
        """
        self._provider = provider
        self._metrics = metrics
        self._rag = rag_retriever
        self._audit = audit_logger
        self._hitl = hitl_manager
        self._confidence_threshold = confidence_threshold
        self._prompt_versions: dict[str, str] = {}
        self._researcher = CapabilityResearcherAgent(
            provider=provider,
            metrics=metrics,
            rag_retriever=rag_retriever,
        )
        self._analyst = PricingAnalystAgent(provider=provider, metrics=metrics)
        self._positioner = MarketPositionerAgent(provider=provider, metrics=metrics)
        self._evaluator = VendorOutputEvaluator(provider=provider, metrics=metrics)

        if prompt_registry is not None:
            self._register_prompts(prompt_registry)

    def _register_prompts(self, registry: PromptRegistry) -> None:
        """Register vendor system prompts and record active versions."""
        registrations = (
            ("capability_researcher", "vendor_researcher_v1", get_capability_researcher_prompt()),
            ("pricing_analyst", "vendor_analyst_v1", get_pricing_analyst_prompt()),
            ("market_positioner", "vendor_positioner_v1", get_market_positioner_prompt()),
            ("vendor_evaluator", "vendor_evaluator_v1", get_vendor_evaluator_prompt()),
        )
        for agent_name, prompt_id, template_text in registrations:
            version = registry.register_prompt(
                prompt_id=prompt_id,
                template_text=template_text,
                description=f"{agent_name} system prompt",
            )
            self._prompt_versions[agent_name] = str(version.version)

    async def evaluate(
        self,
        vendor_name: str,
        vendor_document: str,
        session_id: str,
    ) -> VendorEvaluationResult:
        """Execute the full vendor evaluation pipeline.

        Args:
            vendor_name: Vendor being evaluated.
            vendor_document: Source vendor profile or document text.
            session_id: Session correlation id.

        Returns:
            ``VendorEvaluationResult`` with specialist outputs, evaluation, and HITL status.
        """
        start_time = time.perf_counter()
        job_id = uuid4().hex[:12]
        logger.info(
            "vendor_pipeline_start",
            session_id=session_id,
            vendor_name=vendor_name,
            job_id=job_id,
        )

        capability_response = await self._researcher.research_capabilities(
            vendor_name=vendor_name,
            vendor_document=vendor_document,
            session_id=session_id,
        )
        self._log_agent_action(
            session_id=session_id,
            agent_name=capability_response.agent_name,
            input_payload={"vendor_name": vendor_name, "document_chars": len(vendor_document)},
            response=capability_response,
        )

        pricing_response = await self._analyst.analyze_pricing(
            vendor_name=vendor_name,
            vendor_document=vendor_document,
            session_id=session_id,
        )
        self._log_agent_action(
            session_id=session_id,
            agent_name=pricing_response.agent_name,
            input_payload={"vendor_name": vendor_name},
            response=pricing_response,
        )

        capability_summary = self._capability_summary(capability_response.output)
        pricing_summary = self._pricing_summary(pricing_response.output)

        market_response = await self._positioner.position_vendor(
            vendor_name=vendor_name,
            vendor_document=vendor_document,
            capability_summary=capability_summary,
            pricing_summary=pricing_summary,
            session_id=session_id,
        )
        self._log_agent_action(
            session_id=session_id,
            agent_name=market_response.agent_name,
            input_payload={
                "vendor_name": vendor_name,
                "capability_summary": capability_summary,
                "pricing_summary": pricing_summary,
            },
            response=market_response,
        )

        eval_response = await self._evaluator.evaluate_outputs(
            capability_result=capability_response.output,
            pricing_result=pricing_response.output,
            market_result=market_response.output,
            session_id=session_id,
        )
        self._log_agent_action(
            session_id=session_id,
            agent_name=eval_response.agent_name,
            action_type=ActionType.EVALUATION,
            input_payload={"specialist_agents": 3},
            response=eval_response,
        )

        hitl_triggered = eval_response.confidence_score < self._confidence_threshold
        status = (
            VendorPipelineStatus.PENDING_REVIEW
            if hitl_triggered
            else VendorPipelineStatus.COMPLETED
        )
        hitl_reason: str | None = None
        persisted_job_id: str | None = None
        if hitl_triggered:
            hitl_reason = str(
                eval_response.output.get("summary")
                or eval_response.output.get("recommendation")
                or "Evaluator confidence below threshold"
            )

        pipeline_outputs = {
            "capability_result": capability_response.output,
            "pricing_result": pricing_response.output,
            "market_result": market_response.output,
            "evaluation_result": eval_response.output,
        }

        if self._hitl is not None:
            job = PipelineJob(
                job_id=job_id,
                session_id=session_id,
                pipeline_name="vendor_evaluation",
                status=JobStatus.PENDING_REVIEW if hitl_triggered else JobStatus.COMPLETED,
                pipeline_inputs={"vendor_name": vendor_name, "document_chars": len(vendor_document)},
                pipeline_outputs=pipeline_outputs,
                evaluator_reasoning=eval_response.output,
                evaluator_confidence=eval_response.confidence_score,
            )
            self._hitl.save_job(job)
            persisted_job_id = job.job_id

        if hitl_triggered and self._audit is not None:
            self._audit.log(
                AuditEntry(
                    session_id=session_id,
                    agent_name="vendor_evaluator",
                    action_type=ActionType.HITL_DECISION,
                    input_payload={
                        "confidence_threshold": self._confidence_threshold,
                        "job_id": persisted_job_id or job_id,
                    },
                    output_payload=eval_response.output,
                    confidence_score=eval_response.confidence_score,
                    hitl_status=HITLStatus.PENDING_REVIEW,
                    token_count=(
                        eval_response.token_usage.input_tokens
                        + eval_response.token_usage.output_tokens
                    ),
                    cost_usd=eval_response.token_usage.estimated_cost_usd,
                )
            )
            logger.warning(
                "hitl_triggered",
                session_id=session_id,
                job_id=persisted_job_id or job_id,
                evaluator_confidence=eval_response.confidence_score,
                threshold=self._confidence_threshold,
            )

        session_summary = self._metrics.get_session_summary(session_id)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        business_metrics = BusinessMetrics(
            cost_per_interaction_usd=float(session_summary.get("estimated_cost_usd", 0.0)),
            task_completion_status=status.value,
            processing_time_ms=elapsed_ms,
            total_input_tokens=int(session_summary.get("input_tokens", 0)),
            total_output_tokens=int(session_summary.get("output_tokens", 0)),
            agent_call_count=int(session_summary.get("call_count", 0)),
        )
        retrieved_chunks = capability_response.output.get("retrieved_chunks")
        if not isinstance(retrieved_chunks, list):
            retrieved_chunks = []

        logger.info(
            "vendor_pipeline_complete",
            session_id=session_id,
            vendor_name=vendor_name,
            status=status.value,
            hitl_triggered=hitl_triggered,
            processing_time_ms=elapsed_ms,
            cost_per_interaction_usd=business_metrics.cost_per_interaction_usd,
        )

        return VendorEvaluationResult(
            status=status,
            job_id=persisted_job_id,
            vendor_name=vendor_name,
            capability_assessments=list(capability_response.output.get("capabilities") or []),
            pricing_analysis=pricing_response.output,
            market_position=market_response.output,
            evaluation_result=eval_response.output,
            business_metrics=business_metrics.model_dump(),
            agent_responses=[
                capability_response.model_dump(),
                pricing_response.model_dump(),
                market_response.model_dump(),
                eval_response.model_dump(),
            ],
            hitl_reason=hitl_reason,
            retrieved_chunks=retrieved_chunks,
            prompt_versions=dict(self._prompt_versions),
        )

    def _capability_summary(self, output: dict[str, Any]) -> str:
        """Build a short capability summary for the market positioner."""
        summary = str(output.get("summary") or "").strip()
        if summary:
            return summary
        names = [
            str(item.get("capability_name") or "")
            for item in output.get("capabilities") or []
            if isinstance(item, dict)
        ]
        names = [name for name in names if name]
        return ", ".join(names) if names else "No capabilities extracted."

    def _pricing_summary(self, output: dict[str, Any]) -> str:
        """Build a short pricing summary for the market positioner."""
        notes = str(output.get("comparison_notes") or "").strip()
        if notes:
            return notes
        pricing_model = str(output.get("pricing_model") or "unknown")
        annual = str(output.get("estimated_annual_cost") or "not stated")
        return f"Model: {pricing_model}; estimated annual cost: {annual}"

    def _log_agent_action(
        self,
        *,
        session_id: str,
        agent_name: str,
        input_payload: dict[str, Any],
        response: AgentResponse,
        action_type: ActionType = ActionType.INVOKE,
    ) -> None:
        """Write an audit entry for one agent invocation when an audit logger is configured."""
        if self._audit is None:
            return
        self._audit.log(
            AuditEntry(
                session_id=session_id,
                agent_name=agent_name,
                action_type=action_type,
                input_payload=input_payload,
                output_payload=response.output,
                confidence_score=response.confidence_score,
                token_count=(
                    response.token_usage.input_tokens + response.token_usage.output_tokens
                ),
                cost_usd=response.token_usage.estimated_cost_usd,
            )
        )
