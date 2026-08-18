"""Requirements Extractor agent — zero-shot with negative prompts."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any
from uuid import uuid4

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.services.base import DocumentProcessor, TextAnalyzer
from packages.core.types.schemas import AgentResponse
from packages.domain_rfp.models import ExtractionResult, RfpRequirement
from packages.domain_rfp.prompts.extractor_prompt import (
    get_extractor_prompt,
    get_extractor_user_prompt,
)
from packages.domain_rfp.utils import clamp_confidence, extract_json_object

logger = get_logger("domain_rfp.requirements_extractor")

AGENT_NAME = "requirements_extractor"


class RequirementsExtractorAgent:
    """Extracts structured requirements from RFP documents."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker,
        document_processor: DocumentProcessor,
        text_analyzer: TextAnalyzer,
    ) -> None:
        """Initialize the requirements extractor.

        Args:
            provider: LLM provider for extraction.
            metrics: Metrics tracker for token/cost accounting.
            document_processor: Document text extraction adapter.
            text_analyzer: NLP analysis adapter.
        """
        self._provider = provider
        self._metrics = metrics
        self._document_processor = document_processor
        self._text_analyzer = text_analyzer

    async def process_document(
        self,
        document: bytes | str,
        session_id: str,
    ) -> AgentResponse:
        """Extract requirements from a document end-to-end.

        Args:
            document: Raw document bytes or filesystem path.
            session_id: Session correlation id.

        Returns:
            AgentResponse containing an ExtractionResult payload.
        """
        started = time.perf_counter()
        task_id = str(uuid4())

        textract_result = await self._document_processor.extract_text(document)
        nlp = await self._text_analyzer.analyze(textract_result.raw_text)

        pii_types = Counter(p.type for p in nlp.pii_entities)
        logger.info(
            "pii_findings",
            session_id=session_id,
            pii_count=len(nlp.pii_entities),
            pii_types=dict(pii_types),
        )

        entity_dicts = [e.model_dump() for e in nlp.entities]
        system_prompt = get_extractor_prompt()
        user_prompt = get_extractor_user_prompt(
            document_text=textract_result.raw_text,
            entities=entity_dicts,
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

        extraction, parse_ok = self._parse_extraction(
            content=model_response.content,
            document_pages=textract_result.pages,
            entity_dicts=entity_dicts,
            pii_types=dict(pii_types),
        )
        confidence = self._calculate_confidence(
            extraction=extraction,
            textract_confidences=[
                b.confidence for b in textract_result.blocks if b.block_type == "LINE"
            ],
            parse_ok=parse_ok,
        )
        extraction.extraction_confidence = confidence
        extraction.total_extracted = len(extraction.requirements)
        extraction.document_pages = textract_result.pages

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "requirements_extraction_complete",
            session_id=session_id,
            requirement_count=extraction.total_extracted,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
        )
        return AgentResponse(
            task_id=task_id,
            agent_name=AGENT_NAME,
            output=extraction.model_dump(),
            confidence_score=confidence,
            token_usage=model_response.token_usage,
            processing_time_ms=elapsed_ms,
        )

    def _parse_extraction(
        self,
        content: str,
        document_pages: int,
        entity_dicts: list[dict[str, Any]],
        pii_types: dict[str, int],
    ) -> tuple[ExtractionResult, bool]:
        try:
            payload = extract_json_object(content)
            requirements_raw = payload.get("requirements", [])
            requirements: list[RfpRequirement] = []
            if isinstance(requirements_raw, list):
                for index, item in enumerate(requirements_raw, start=1):
                    if not isinstance(item, dict):
                        continue
                    requirements.append(
                        RfpRequirement(
                            requirement_id=str(item.get("requirement_id") or f"REQ-{index:03d}"),
                            text=str(item.get("text", "")).strip(),
                            category=str(item.get("category", "other")),
                            priority=str(item.get("priority", "should-have")),
                            source_page=item.get("source_page"),
                            source_section=item.get("source_section"),
                            entities=item.get("entities") or entity_dicts[:3] or None,
                            pii_detected=bool(item.get("pii_detected", False)),
                        )
                    )
            result = ExtractionResult(
                requirements=[r for r in requirements if r.text],
                total_extracted=len(requirements),
                document_pages=int(payload.get("document_pages") or document_pages),
                extraction_confidence=float(payload.get("extraction_confidence") or 0.0),
                pii_summary=payload.get("pii_summary")
                or {"count": sum(pii_types.values()), "types": pii_types},
            )
            return result, True
        except Exception as exc:  # noqa: BLE001 - intentional low-confidence fallback
            logger.error("extraction_json_parse_failed", error=str(exc))
            return (
                ExtractionResult(
                    requirements=[],
                    total_extracted=0,
                    document_pages=document_pages,
                    extraction_confidence=0.15,
                    pii_summary={"count": sum(pii_types.values()), "types": pii_types},
                ),
                False,
            )

    def _calculate_confidence(
        self,
        extraction: ExtractionResult,
        textract_confidences: list[float],
        parse_ok: bool,
    ) -> float:
        if not parse_ok:
            return 0.2

        pages = max(1, extraction.document_pages)
        density = extraction.total_extracted / pages
        # Flag under-extraction (<0.5 req/page) and over-extraction (>20/page).
        if density < 0.5:
            density_score = 0.45
        elif density > 20:
            density_score = 0.4
        elif 1.0 <= density <= 12.0:
            density_score = 0.9
        else:
            density_score = 0.7

        if textract_confidences:
            avg_textract = sum(textract_confidences) / len(textract_confidences) / 100.0
        else:
            avg_textract = 0.75

        combined = (0.55 * density_score) + (0.45 * avg_textract)
        if extraction.total_extracted == 0:
            combined = min(combined, 0.35)
        return clamp_confidence(combined)
