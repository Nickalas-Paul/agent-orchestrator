"""End-to-end RFP analysis pipeline wiring specialist agents, evaluator, and HITL."""

from __future__ import annotations

import time
from typing import Any

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry, HITLStatus
from packages.core.cloud.base import ModelProvider
from packages.core.guardrails import GuardrailAction, GuardrailsEngine
from packages.core.hitl.manager import HITLManager
from packages.core.hitl.models import JobStatus, PipelineJob
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.orchestrator.engine import OrchestratorEngine
from packages.core.orchestrator.models import AgentCapability
from packages.core.services.base import DocumentProcessor, TextAnalyzer
from packages.core.types.schemas import AgentResponse
from packages.domain_rfp.agents.capability_mapper import CapabilityMapperAgent
from packages.domain_rfp.agents.evaluator import OutputEvaluator
from packages.domain_rfp.agents.gap_analyzer import GapAnalyzerAgent
from packages.domain_rfp.agents.requirements_extractor import RequirementsExtractorAgent
from packages.domain_rfp.models import (
    ExtractionResult,
    MappingResult,
    PipelineResult,
    PipelineStatus,
    RfpRequirement,
)

logger = get_logger("domain_rfp.pipeline")

RFP_AGENT_CAPABILITIES: list[AgentCapability] = [
    AgentCapability(
        agent_name="requirements_extractor",
        description="Extracts structured requirements from RFP documents using OCR/NLP and LLM analysis.",
        supported_task_types=["rfp_extract", "requirements_extraction"],
    ),
    AgentCapability(
        agent_name="capability_mapper",
        description="Maps RFP requirements to known enterprise capabilities with full/partial/none match levels.",
        supported_task_types=["rfp_map", "capability_mapping"],
    ),
    AgentCapability(
        agent_name="gap_analyzer",
        description="Performs chain-of-thought gap analysis and bid-risk assessment for unmatched requirements.",
        supported_task_types=["rfp_gap", "gap_analysis"],
    ),
    AgentCapability(
        agent_name="evaluator",
        description="Evaluates specialist outputs for completeness, consistency, contradiction, and reasoning quality.",
        supported_task_types=["rfp_evaluate", "output_evaluation"],
    ),
]


def register_rfp_agents(engine: OrchestratorEngine) -> None:
    """Register all RFP specialist agents with an orchestrator engine.

    Args:
        engine: Orchestrator engine that should know about RFP agents.
    """
    for capability in RFP_AGENT_CAPABILITIES:
        engine.register_agent(capability)


def _document_ref(document: bytes | str) -> dict[str, Any]:
    """Build a JSON-safe reference to the pipeline input document."""
    if isinstance(document, str):
        return {"document": document, "input_kind": "path_or_text"}
    return {"document_bytes": len(document), "input_kind": "bytes"}


class RfpAnalysisPipeline:
    """Runs extract → map → gap-analyze → evaluate for an RFP document."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        document_processor: DocumentProcessor,
        text_analyzer: TextAnalyzer,
        orchestrator: OrchestratorEngine | None = None,
        audit_logger: AuditLogger | None = None,
        hitl_manager: HITLManager | None = None,
        confidence_threshold: float = 0.85,
        guardrails: GuardrailsEngine | None = None,
    ) -> None:
        """Initialize the pipeline and optionally register agents.

        Args:
            provider: Shared model provider for all agents.
            metrics: Shared metrics tracker.
            document_processor: Document extraction adapter.
            text_analyzer: NLP analysis adapter.
            orchestrator: Optional orchestrator to register agent capabilities on.
            audit_logger: Optional insert-only audit logger. When omitted, DB writes are skipped.
            hitl_manager: Optional HITL job manager. When omitted, jobs are not persisted.
            confidence_threshold: Evaluator confidence required to skip human review.
            guardrails: Optional shared guardrails engine for input/output safety checks.
        """
        self._provider = provider
        self._metrics = metrics
        self._audit = audit_logger
        self._hitl = hitl_manager
        self._confidence_threshold = confidence_threshold
        self._guardrails = guardrails
        self._extractor = RequirementsExtractorAgent(
            provider=provider,
            metrics=metrics,
            document_processor=document_processor,
            text_analyzer=text_analyzer,
        )
        self._mapper = CapabilityMapperAgent(provider=provider, metrics=metrics)
        self._analyzer = GapAnalyzerAgent(provider=provider, metrics=metrics)
        self._evaluator = OutputEvaluator(provider=provider, metrics=metrics)

        if orchestrator is not None:
            register_rfp_agents(orchestrator)

    async def analyze(self, document: bytes | str, session_id: str) -> PipelineResult:
        """Execute the full RFP analysis pipeline.

        Args:
            document: Document bytes or filesystem path.
            session_id: Session correlation id.

        Returns:
            ``PipelineResult`` with specialist outputs, evaluation, and HITL status.
        """
        started = time.perf_counter()
        logger.info("rfp_pipeline_start", session_id=session_id)

        # Input guardrails (when document is text / path string)
        if self._guardrails is not None and isinstance(document, str):
            input_result = await self._guardrails.check_input(
                text=document,
                domain="rfp",
                session_id=session_id,
            )
            if not input_result.passed:
                logger.warning(
                    "rfp_input_guardrail_blocked",
                    session_id=session_id,
                    reasons=input_result.blocked_reasons,
                )
                return PipelineResult(
                    status=PipelineStatus.FAILED,
                    job_id=None,
                    hitl_reason=(
                        "Input blocked by guardrails: "
                        + "; ".join(input_result.blocked_reasons)
                    ),
                )

        extraction_response = await self._extractor.process_document(
            document=document,
            session_id=session_id,
        )
        if extraction_response.confidence_score < 0.5:
            logger.warning(
                "extraction_low_confidence",
                session_id=session_id,
                confidence=extraction_response.confidence_score,
            )

        extraction = ExtractionResult.model_validate(extraction_response.output)
        requirements = [
            RfpRequirement.model_validate(item)
            if isinstance(item, dict)
            else item
            for item in extraction.requirements
        ]

        mapping_response = await self._mapper.map_requirements(
            requirements=requirements,
            session_id=session_id,
        )
        mapping = MappingResult.model_validate(mapping_response.output)

        gap_response = await self._analyzer.analyze_gaps(
            mappings=mapping,
            session_id=session_id,
        )

        eval_response = await self._evaluator.evaluate_outputs(
            extraction_result=extraction_response.output,
            mapping_result=mapping_response.output,
            gap_result=gap_response.output,
            session_id=session_id,
        )

        hitl_triggered = eval_response.confidence_score < self._confidence_threshold
        hitl_reason: str | None = None
        if hitl_triggered:
            hitl_reason = str(
                eval_response.output.get("summary")
                or eval_response.output.get("recommendation")
                or "Evaluator confidence below threshold"
            )

        # Output guardrails on evaluator response — FLAG or BLOCK forces HITL
        if self._guardrails is not None:
            output_result = await self._guardrails.check_output(
                text=str(eval_response.output),
                domain="rfp",
                session_id=session_id,
            )
            if output_result.action != GuardrailAction.PASS:
                hitl_triggered = True
                hitl_reason = (
                    "Output flagged by guardrails: "
                    + "; ".join(output_result.blocked_reasons + output_result.flagged_reasons)
                )

        self._log_agent_invoke(
            session_id=session_id,
            agent_name=extraction_response.agent_name,
            input_payload=_document_ref(document),
            response=extraction_response,
        )
        self._log_agent_invoke(
            session_id=session_id,
            agent_name=mapping_response.agent_name,
            input_payload={"requirement_count": len(requirements)},
            response=mapping_response,
        )
        self._log_agent_invoke(
            session_id=session_id,
            agent_name=gap_response.agent_name,
            input_payload={
                "fully_matched": mapping.fully_matched,
                "partially_matched": mapping.partially_matched,
                "unmatched": mapping.unmatched,
            },
            response=gap_response,
        )
        self._log_agent_invoke(
            session_id=session_id,
            agent_name=eval_response.agent_name,
            action_type=ActionType.EVALUATION,
            input_payload={"specialist_agents": 3},
            response=eval_response,
        )

        status = (
            PipelineStatus.PENDING_REVIEW if hitl_triggered else PipelineStatus.COMPLETED
        )
        job_id: str | None = None

        pipeline_outputs = {
            "extraction_result": extraction_response.output,
            "mapping_result": mapping_response.output,
            "gap_analysis_result": gap_response.output,
            "evaluation_result": eval_response.output,
        }

        if self._hitl is not None:
            job = PipelineJob(
                session_id=session_id,
                pipeline_name="rfp_analysis",
                status=JobStatus.PENDING_REVIEW if hitl_triggered else JobStatus.COMPLETED,
                pipeline_inputs=_document_ref(document),
                pipeline_outputs=pipeline_outputs,
                evaluator_reasoning=eval_response.output,
                evaluator_confidence=eval_response.confidence_score,
            )
            self._hitl.save_job(job)
            job_id = job.job_id

        if hitl_triggered and self._audit is not None:
            self._audit.log(
                AuditEntry(
                    session_id=session_id,
                    agent_name="evaluator",
                    action_type=ActionType.EVALUATION,
                    input_payload={
                        "confidence_threshold": self._confidence_threshold,
                        "job_id": job_id,
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
                job_id=job_id,
                evaluator_confidence=eval_response.confidence_score,
                threshold=self._confidence_threshold,
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        summary = self._metrics.get_session_summary(session_id)
        pipeline_metrics = {
            "total_time_ms": elapsed_ms,
            "total_tokens": int(summary.get("total_tokens", 0)),
            "total_cost_usd": float(summary.get("estimated_cost_usd", 0.0)),
            "agent_count": 4,
            "evaluator_confidence": eval_response.confidence_score,
            "hitl_triggered": hitl_triggered,
            "status": status.value,
        }
        logger.info(
            "rfp_pipeline_complete",
            session_id=session_id,
            total_time_ms=pipeline_metrics["total_time_ms"],
            total_tokens=pipeline_metrics["total_tokens"],
            total_cost_usd=pipeline_metrics["total_cost_usd"],
            status=status.value,
            hitl_triggered=hitl_triggered,
        )

        return PipelineResult(
            status=status,
            job_id=job_id,
            extraction_result=extraction_response.output,
            mapping_result=mapping_response.output,
            gap_analysis_result=gap_response.output,
            evaluation_result=eval_response.output,
            pipeline_metrics=pipeline_metrics,
            agent_responses=[
                extraction_response.model_dump(),
                mapping_response.model_dump(),
                gap_response.model_dump(),
                eval_response.model_dump(),
            ],
            hitl_reason=hitl_reason,
        )

    def _log_agent_invoke(
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
