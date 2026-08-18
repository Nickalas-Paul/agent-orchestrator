"""End-to-end RFP analysis pipeline wiring the three specialist agents."""

from __future__ import annotations

import time
from typing import Any

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.orchestrator.engine import OrchestratorEngine
from packages.core.orchestrator.models import AgentCapability
from packages.core.services.base import DocumentProcessor, TextAnalyzer
from packages.core.types.schemas import AgentResponse
from packages.domain_rfp.agents.capability_mapper import CapabilityMapperAgent
from packages.domain_rfp.agents.gap_analyzer import GapAnalyzerAgent
from packages.domain_rfp.agents.requirements_extractor import RequirementsExtractorAgent
from packages.domain_rfp.models import ExtractionResult, MappingResult, RfpRequirement

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
]


def register_rfp_agents(engine: OrchestratorEngine) -> None:
    """Register all RFP specialist agents with an orchestrator engine.

    Args:
        engine: Orchestrator engine that should know about RFP agents.
    """
    for capability in RFP_AGENT_CAPABILITIES:
        engine.register_agent(capability)


class RfpAnalysisPipeline:
    """Runs extract → map → gap-analyze for an RFP document."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        document_processor: DocumentProcessor,
        text_analyzer: TextAnalyzer,
        orchestrator: OrchestratorEngine | None = None,
    ) -> None:
        """Initialize the pipeline and optionally register agents.

        Args:
            provider: Shared model provider for all agents.
            metrics: Shared metrics tracker.
            document_processor: Document extraction adapter.
            text_analyzer: NLP analysis adapter.
            orchestrator: Optional orchestrator to register agent capabilities on.
        """
        self._provider = provider
        self._metrics = metrics
        self._extractor = RequirementsExtractorAgent(
            provider=provider,
            metrics=metrics,
            document_processor=document_processor,
            text_analyzer=text_analyzer,
        )
        self._mapper = CapabilityMapperAgent(provider=provider, metrics=metrics)
        self._analyzer = GapAnalyzerAgent(provider=provider, metrics=metrics)

        if orchestrator is not None:
            register_rfp_agents(orchestrator)

    async def analyze(self, document: bytes | str, session_id: str) -> dict[str, Any]:
        """Execute the full RFP analysis pipeline.

        Args:
            document: Document bytes or filesystem path.
            session_id: Session correlation id.

        Returns:
            Dict with extraction, mapping, gap analysis results and pipeline metrics.
        """
        started = time.perf_counter()
        logger.info("rfp_pipeline_start", session_id=session_id)

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

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        summary = self._metrics.get_session_summary(session_id)
        pipeline_metrics = {
            "total_time_ms": elapsed_ms,
            "total_tokens": int(summary.get("total_tokens", 0)),
            "total_cost_usd": float(summary.get("estimated_cost_usd", 0.0)),
            "agent_count": 3,
        }
        logger.info(
            "rfp_pipeline_complete",
            session_id=session_id,
            total_time_ms=pipeline_metrics["total_time_ms"],
            total_tokens=pipeline_metrics["total_tokens"],
            total_cost_usd=pipeline_metrics["total_cost_usd"],
        )

        return {
            "extraction_result": extraction_response.output,
            "mapping_result": mapping_response.output,
            "gap_analysis_result": gap_response.output,
            "pipeline_metrics": pipeline_metrics,
            "agent_responses": [
                extraction_response.model_dump(),
                mapping_response.model_dump(),
                gap_response.model_dump(),
            ],
        }
