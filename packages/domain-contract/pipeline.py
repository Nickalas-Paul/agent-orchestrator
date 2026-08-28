"""End-to-end contract risk-review pipeline wiring specialist agents, evaluator, and HITL."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry, HITLStatus
from packages.core.cloud.base import ModelProvider
from packages.core.guardrails import GuardrailAction, GuardrailsEngine
from packages.core.hitl.manager import HITLManager
from packages.core.hitl.models import JobStatus, PipelineJob
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.prompts.registry import PromptRegistry
from packages.core.rag.retriever import RAGRetriever
from packages.core.types.schemas import AgentResponse, BusinessMetrics
from packages.domain_contract.agents.clause_risk_analyst import ClauseRiskAnalystAgent
from packages.domain_contract.agents.compliance_checker import ComplianceCheckerAgent
from packages.domain_contract.agents.contract_evaluator import ContractOutputEvaluator
from packages.domain_contract.agents.terms_comparator import TermsComparatorAgent
from packages.domain_contract.models import ContractPipelineStatus, ContractReviewResult
from packages.domain_contract.prompts.analyst_prompt import get_analyst_prompt
from packages.domain_contract.prompts.checker_prompt import get_checker_prompt
from packages.domain_contract.prompts.comparator_prompt import get_comparator_prompt
from packages.domain_contract.prompts.evaluator_prompt import get_contract_evaluator_prompt

logger = get_logger("domain_contract.pipeline")


class ContractReviewPipeline:
    """Runs risk → compliance → terms → evaluate for a contract document."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        rag_retriever: RAGRetriever | None = None,
        audit_logger: AuditLogger | None = None,
        hitl_manager: HITLManager | None = None,
        confidence_threshold: float = 0.85,
        prompt_registry: PromptRegistry | None = None,
        guardrails: GuardrailsEngine | None = None,
    ) -> None:
        """Initialize the pipeline and optionally register prompt versions.

        Args:
            provider: Shared model provider for all agents.
            metrics: Shared metrics tracker.
            rag_retriever: Optional RAG retriever passed to the terms comparator.
            audit_logger: Optional insert-only audit logger. When omitted, DB writes are skipped.
            hitl_manager: Optional HITL job manager. When omitted, jobs are not persisted.
            confidence_threshold: Evaluator confidence required to skip human review.
            prompt_registry: Optional prompt registry for version tracking.
            guardrails: Optional shared guardrails engine for input/output safety checks.
        """
        self._provider = provider
        self._metrics = metrics
        self._rag = rag_retriever
        self._audit = audit_logger
        self._hitl = hitl_manager
        self._confidence_threshold = confidence_threshold
        self._guardrails = guardrails
        self._prompt_versions: dict[str, int] = {}
        self._risk_analyst = ClauseRiskAnalystAgent(provider=provider, metrics=metrics)
        self._compliance_checker = ComplianceCheckerAgent(
            provider=provider,
            metrics=metrics,
        )
        self._terms_comparator = TermsComparatorAgent(
            provider=provider,
            metrics=metrics,
            rag_retriever=rag_retriever,
        )
        self._evaluator = ContractOutputEvaluator(provider=provider, metrics=metrics)

        if prompt_registry is not None:
            self._register_prompts(prompt_registry)

    def _register_prompts(self, registry: PromptRegistry) -> None:
        """Register contract system prompts and record active versions."""
        registrations = (
            ("clause_risk_analyst", "contract_analyst_v1", get_analyst_prompt()),
            ("compliance_checker", "contract_checker_v1", get_checker_prompt()),
            ("terms_comparator", "contract_comparator_v1", get_comparator_prompt()),
            ("contract_evaluator", "contract_evaluator_v1", get_contract_evaluator_prompt()),
        )
        for agent_name, prompt_id, template_text in registrations:
            version = registry.register_prompt(
                prompt_id=prompt_id,
                template_text=template_text,
                description=f"{agent_name} system prompt",
            )
            self._prompt_versions[agent_name] = int(version.version)

    async def review(
        self,
        contract_name: str,
        contract_text: str,
        session_id: str = "",
    ) -> ContractReviewResult:
        """Execute the full contract risk-review pipeline.

        Args:
            contract_name: Contract being reviewed.
            contract_text: Source contract document text.
            session_id: Session correlation id.

        Returns:
            ``ContractReviewResult`` with specialist outputs, evaluation, and HITL status.
        """
        start_time = time.perf_counter()
        job_id = str(uuid4())[:12]
        logger.info(
            "contract_pipeline_start",
            session_id=session_id,
            contract_name=contract_name,
            job_id=job_id,
        )

        try:
            return await self._run_review(
                contract_name=contract_name,
                contract_text=contract_text,
                session_id=session_id,
                job_id=job_id,
                start_time=start_time,
            )
        except Exception as exc:  # noqa: BLE001 - pipeline must return FAILED, not raise
            logger.error(
                "contract_pipeline_failed",
                session_id=session_id,
                contract_name=contract_name,
                error=str(exc),
            )
            return ContractReviewResult(
                status=ContractPipelineStatus.FAILED,
                job_id=job_id,
                contract_name=contract_name,
                hitl_reason=f"Pipeline error: {exc}",
                prompt_versions=dict(self._prompt_versions),
            )

    async def _run_review(
        self,
        *,
        contract_name: str,
        contract_text: str,
        session_id: str,
        job_id: str,
        start_time: float,
    ) -> ContractReviewResult:
        """Core review flow after job_id allocation."""
        if self._guardrails is not None:
            input_result = await self._guardrails.check_input(
                text=contract_text,
                domain="contract",
                session_id=session_id,
            )
            if not input_result.passed:
                logger.warning(
                    "contract_input_guardrail_blocked",
                    session_id=session_id,
                    contract_name=contract_name,
                    reasons=input_result.blocked_reasons,
                )
                return ContractReviewResult(
                    status=ContractPipelineStatus.FAILED,
                    job_id="",
                    contract_name=contract_name,
                    hitl_reason=(
                        "Input blocked by guardrails: "
                        + "; ".join(input_result.blocked_reasons)
                    ),
                    prompt_versions=dict(self._prompt_versions),
                )

        risk_response = await self._risk_analyst.analyze_risks(
            contract_text=contract_text,
            session_id=session_id,
        )
        self._log_agent_action(
            session_id=session_id,
            agent_name=risk_response.agent_name,
            input_payload={
                "contract_name": contract_name,
                "document_chars": len(contract_text),
            },
            response=risk_response,
        )

        compliance_response = await self._compliance_checker.check_compliance(
            contract_text=contract_text,
            session_id=session_id,
        )
        self._log_agent_action(
            session_id=session_id,
            agent_name=compliance_response.agent_name,
            input_payload={"contract_name": contract_name},
            response=compliance_response,
        )

        comparison_response = await self._terms_comparator.compare_terms(
            contract_text=contract_text,
            contract_name=contract_name,
            session_id=session_id,
        )
        self._log_agent_action(
            session_id=session_id,
            agent_name=comparison_response.agent_name,
            input_payload={"contract_name": contract_name},
            response=comparison_response,
        )

        eval_response = await self._evaluator.evaluate_outputs(
            risk_result=risk_response.output,
            compliance_result=compliance_response.output,
            comparison_result=comparison_response.output,
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
        hitl_reason = ""
        if hitl_triggered:
            hitl_reason = str(
                eval_response.output.get("summary")
                or eval_response.output.get("recommendation")
                or "Evaluator confidence below threshold"
            )

        retrieved_chunks = comparison_response.output.get("retrieved_chunks")
        if not isinstance(retrieved_chunks, list):
            retrieved_chunks = []

        if self._guardrails is not None:
            output_result = await self._guardrails.check_output(
                text=str(eval_response.output),
                retrieved_chunks=retrieved_chunks if retrieved_chunks else None,
                domain="contract",
                session_id=session_id,
            )
            if output_result.action != GuardrailAction.PASS:
                hitl_triggered = True
                hitl_reason = (
                    "Output flagged by guardrails: "
                    + "; ".join(output_result.blocked_reasons + output_result.flagged_reasons)
                )

        status = (
            ContractPipelineStatus.PENDING_REVIEW
            if hitl_triggered
            else ContractPipelineStatus.COMPLETED
        )
        persisted_job_id = ""

        pipeline_outputs = {
            "clause_risk_result": risk_response.output,
            "compliance_result": compliance_response.output,
            "terms_comparison_result": comparison_response.output,
            "evaluation_result": eval_response.output,
        }

        if self._hitl is not None:
            job = PipelineJob(
                job_id=job_id,
                session_id=session_id,
                pipeline_name="contract_review",
                status=JobStatus.PENDING_REVIEW if hitl_triggered else JobStatus.COMPLETED,
                pipeline_inputs={
                    "contract_name": contract_name,
                    "document_chars": len(contract_text),
                },
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
                    agent_name="contract_evaluator",
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

        logger.info(
            "contract_pipeline_complete",
            session_id=session_id,
            contract_name=contract_name,
            status=status.value,
            hitl_triggered=hitl_triggered,
            processing_time_ms=elapsed_ms,
            cost_per_interaction_usd=business_metrics.cost_per_interaction_usd,
        )

        return ContractReviewResult(
            status=status,
            job_id=persisted_job_id,
            contract_name=contract_name,
            clause_risk_result=risk_response.output,
            compliance_result=compliance_response.output,
            terms_comparison_result=comparison_response.output,
            evaluation_result=eval_response.output,
            business_metrics=business_metrics.model_dump(),
            agent_responses=[
                risk_response.model_dump(),
                compliance_response.model_dump(),
                comparison_response.model_dump(),
                eval_response.model_dump(),
            ],
            hitl_reason=hitl_reason,
            retrieved_chunks=retrieved_chunks,
            prompt_versions=dict(self._prompt_versions),
        )

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
