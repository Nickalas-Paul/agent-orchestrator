"""Tests for Textract and Comprehend service adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.core.services.base import ComprehendResult, TextractResult
from packages.core.services.comprehend import LocalTextAnalyzer
from packages.core.services.config import (
    ServiceConfig,
    get_document_processor,
    get_text_analyzer,
)
from packages.core.services.textract import LocalDocumentProcessor, TextractProcessor
from packages.core.services.comprehend import ComprehendAnalyzer

SAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "packages" / "domain-rfp" / "sample_data"
)
SAMPLE_RFP = SAMPLE_DIR / "sample_rfp_text.txt"


@pytest.mark.asyncio
async def test_local_document_processor_from_text_string() -> None:
    processor = LocalDocumentProcessor()
    result = await processor.extract_text(
        "Line one requirement.\nLine two requirement.\n",
        document_type="txt",
    )
    assert isinstance(result, TextractResult)
    assert result.pages >= 1
    assert result.raw_text
    assert any(block.block_type == "LINE" for block in result.blocks)
    assert any(block.block_type == "WORD" for block in result.blocks)
    assert all(85.0 <= b.confidence <= 99.0 for b in result.blocks if b.block_type == "LINE")


@pytest.mark.asyncio
async def test_local_document_processor_from_sample_rfp_file() -> None:
    processor = LocalDocumentProcessor()
    result = await processor.extract_text(str(SAMPLE_RFP), document_type="txt")
    assert isinstance(result, TextractResult)
    assert "REST API" in result.raw_text or "SOC 2" in result.raw_text
    assert result.tables == []
    assert result.key_value_pairs == []


@pytest.mark.asyncio
async def test_local_text_analyzer_returns_valid_result() -> None:
    analyzer = LocalTextAnalyzer()
    text = SAMPLE_RFP.read_text(encoding="utf-8")
    result = await analyzer.analyze(text)
    assert isinstance(result, ComprehendResult)
    assert result.language == "en"
    assert result.sentiment == "NEUTRAL"
    assert "positive" in result.sentiment_scores
    assert result.key_phrases


@pytest.mark.asyncio
async def test_local_text_analyzer_detects_email_as_pii() -> None:
    analyzer = LocalTextAnalyzer()
    text = "Contact us at bids@vendor-example.com for questions."
    pii = await analyzer.detect_pii(text)
    assert any(item.type == "EMAIL" for item in pii)


@pytest.mark.asyncio
async def test_local_text_analyzer_detects_date_entities() -> None:
    analyzer = LocalTextAnalyzer()
    text = "Implementation must complete by December 31, 2026 and pilot in Q3 2026."
    entities = await analyzer.detect_entities(text)
    assert any(item.entity_type == "DATE" for item in entities)


def test_factory_returns_local_implementations_by_default() -> None:
    config = ServiceConfig(
        textract_provider="local",
        comprehend_provider="local",
        aws_region="us-east-1",
    )
    assert isinstance(get_document_processor(config), LocalDocumentProcessor)
    assert isinstance(get_text_analyzer(config), LocalTextAnalyzer)


def test_factory_returns_aws_implementations_when_configured() -> None:
    config = ServiceConfig(
        textract_provider="aws",
        comprehend_provider="aws",
        aws_region="us-east-1",
    )
    assert isinstance(get_document_processor(config), TextractProcessor)
    assert isinstance(get_text_analyzer(config), ComprehendAnalyzer)
