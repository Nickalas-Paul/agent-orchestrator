"""Amazon Comprehend text analyzer adapters (AWS + local mock)."""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from packages.core.logging.logger import get_logger
from packages.core.services.base import (
    ComprehendResult,
    Entity,
    PiiEntity,
    TextAnalyzer,
)

logger = get_logger("services.comprehend")

_COMPREHEND_MAX_BYTES = 100_000


class ComprehendAnalyzerError(Exception):
    """Base error for Comprehend analyzer failures."""


class ComprehendTextTooLongError(ComprehendAnalyzerError):
    """Raised when input text exceeds Comprehend size limits."""


class ComprehendThrottlingError(ComprehendAnalyzerError):
    """Raised when Comprehend throttles the request."""


class ComprehendAnalyzer(TextAnalyzer):
    """Real AWS Comprehend implementation."""

    def __init__(self, region_name: str = "us-east-1", client: Any | None = None) -> None:
        """Initialize the AWS Comprehend analyzer.

        Args:
            region_name: AWS region for the Comprehend client.
            client: Optional pre-built boto3 client (useful for tests).
        """
        self._client = client or boto3.client("comprehend", region_name=region_name)
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def analyze(self, text: str) -> ComprehendResult:
        """Run entity, PII, key phrase, and sentiment detection."""
        self._validate_text_length(text)
        language = "en"

        entities_task = asyncio.to_thread(self._detect_entities_sync, text, language)
        pii_task = asyncio.to_thread(self._detect_pii_sync, text, language)
        phrases_task = asyncio.to_thread(self._detect_key_phrases_sync, text, language)
        sentiment_task = asyncio.to_thread(self._detect_sentiment_sync, text, language)

        entities, pii_entities, key_phrases, sentiment_payload = await asyncio.gather(
            entities_task,
            pii_task,
            phrases_task,
            sentiment_task,
        )
        result = ComprehendResult(
            entities=entities,
            pii_entities=pii_entities,
            key_phrases=key_phrases,
            sentiment=str(sentiment_payload.get("Sentiment", "NEUTRAL")),
            sentiment_scores={
                k.lower().replace("score", ""): float(v)
                for k, v in (sentiment_payload.get("SentimentScore") or {}).items()
            },
            language=language,
        )
        logger.info(
            "comprehend_analyze_complete",
            entity_count=len(result.entities),
            pii_count=len(result.pii_entities),
            sentiment=result.sentiment,
        )
        return result

    async def detect_pii(self, text: str) -> list[PiiEntity]:
        """Detect PII entities in text."""
        self._validate_text_length(text)
        return await asyncio.to_thread(self._detect_pii_sync, text, "en")

    async def detect_entities(self, text: str) -> list[Entity]:
        """Detect named entities in text."""
        self._validate_text_length(text)
        return await asyncio.to_thread(self._detect_entities_sync, text, "en")

    def _validate_text_length(self, text: str) -> None:
        if len(text.encode("utf-8")) > _COMPREHEND_MAX_BYTES:
            raise ComprehendTextTooLongError(
                f"Text exceeds Comprehend {_COMPREHEND_MAX_BYTES} byte limit"
            )

    def _detect_entities_sync(self, text: str, language: str) -> list[Entity]:
        try:
            response = self._client.detect_entities(Text=text, LanguageCode=language)
        except ClientError as exc:
            raise self._map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise ComprehendAnalyzerError(str(exc)) from exc
        return [
            Entity(
                text=str(item.get("Text", "")),
                entity_type=str(item.get("Type", "OTHER")),
                confidence=float(item.get("Score", 0.0) or 0.0) * 100.0,
                begin_offset=int(item.get("BeginOffset", 0) or 0),
                end_offset=int(item.get("EndOffset", 0) or 0),
            )
            for item in response.get("Entities", [])
        ]

    def _detect_pii_sync(self, text: str, language: str) -> list[PiiEntity]:
        try:
            response = self._client.detect_pii_entities(Text=text, LanguageCode=language)
        except ClientError as exc:
            raise self._map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise ComprehendAnalyzerError(str(exc)) from exc
        return [
            PiiEntity(
                type=str(item.get("Type", "OTHER")),
                confidence=float(item.get("Score", 0.0) or 0.0) * 100.0,
                begin_offset=int(item.get("BeginOffset", 0) or 0),
                end_offset=int(item.get("EndOffset", 0) or 0),
            )
            for item in response.get("Entities", [])
        ]

    def _detect_key_phrases_sync(self, text: str, language: str) -> list[str]:
        try:
            response = self._client.detect_key_phrases(Text=text, LanguageCode=language)
        except ClientError as exc:
            raise self._map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise ComprehendAnalyzerError(str(exc)) from exc
        return [str(item.get("Text", "")) for item in response.get("KeyPhrases", [])]

    def _detect_sentiment_sync(self, text: str, language: str) -> dict[str, Any]:
        try:
            return self._client.detect_sentiment(Text=text, LanguageCode=language)
        except ClientError as exc:
            raise self._map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise ComprehendAnalyzerError(str(exc)) from exc

    def _map_client_error(self, exc: ClientError) -> ComprehendAnalyzerError:
        error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = str(error.get("Code", ""))
        message = str(error.get("Message", str(exc)))
        if code in {"TextSizeLimitExceededException", "InvalidRequestException"}:
            return ComprehendTextTooLongError(message)
        if code in {"ThrottlingException", "TooManyRequestsException"}:
            return ComprehendThrottlingError(message)
        return ComprehendAnalyzerError(f"Comprehend error ({code}): {message}")


class LocalTextAnalyzer(TextAnalyzer):
    """Regex-based local/mock NLP analyzer for development without AWS."""

    _DATE_PATTERNS = [
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:January|February|March|April|May|June|July|August|"
                   r"September|October|November|December)\s+\d{4}\b", re.IGNORECASE),
        re.compile(r"\bQ[1-4]\s+\d{4}\b", re.IGNORECASE),
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    ]
    _EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _ORG_PATTERN = re.compile(
        r"\b([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.']*)*\s+"
        r"(?:Inc\.|Corp\.|LLC|Ltd\.|Company|Corporation))\b"
    )
    _MONEY_PATTERN = re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
    _PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
    _SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    async def analyze(self, text: str) -> ComprehendResult:
        """Run local regex-based NLP analysis."""
        logger.info("local_text_analyzer_mode", text_length=len(text))
        entities = await self.detect_entities(text)
        pii_entities = await self.detect_pii(text)
        key_phrases = self._extract_key_phrases(text)
        return ComprehendResult(
            entities=entities,
            pii_entities=pii_entities,
            key_phrases=key_phrases,
            sentiment="NEUTRAL",
            sentiment_scores={
                "positive": 0.25,
                "negative": 0.25,
                "neutral": 0.45,
                "mixed": 0.05,
            },
            language="en",
        )

    async def detect_pii(self, text: str) -> list[PiiEntity]:
        """Detect email, phone, and SSN-like PII via regex."""
        findings: list[PiiEntity] = []
        for match in self._EMAIL_PATTERN.finditer(text):
            findings.append(
                PiiEntity(
                    type="EMAIL",
                    confidence=95.0,
                    begin_offset=match.start(),
                    end_offset=match.end(),
                )
            )
        for match in self._PHONE_PATTERN.finditer(text):
            findings.append(
                PiiEntity(
                    type="PHONE",
                    confidence=90.0,
                    begin_offset=match.start(),
                    end_offset=match.end(),
                )
            )
        for match in self._SSN_PATTERN.finditer(text):
            findings.append(
                PiiEntity(
                    type="SSN",
                    confidence=92.0,
                    begin_offset=match.start(),
                    end_offset=match.end(),
                )
            )
        return findings

    async def detect_entities(self, text: str) -> list[Entity]:
        """Detect dates, organizations, and monetary quantities via regex."""
        entities: list[Entity] = []
        for pattern in self._DATE_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    Entity(
                        text=match.group(0),
                        entity_type="DATE",
                        confidence=88.0,
                        begin_offset=match.start(),
                        end_offset=match.end(),
                    )
                )
        for match in self._ORG_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0),
                    entity_type="ORGANIZATION",
                    confidence=85.0,
                    begin_offset=match.start(),
                    end_offset=match.end(),
                )
            )
        for match in self._MONEY_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0),
                    entity_type="QUANTITY",
                    confidence=90.0,
                    begin_offset=match.start(),
                    end_offset=match.end(),
                )
            )
        for match in self._EMAIL_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0),
                    entity_type="OTHER",
                    confidence=95.0,
                    begin_offset=match.start(),
                    end_offset=match.end(),
                )
            )
        return entities

    def _extract_key_phrases(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        phrases: list[str] = []
        for paragraph in paragraphs:
            sentence = re.split(r"(?<=[.!?])\s+", paragraph)[0].strip()
            if sentence:
                phrases.append(sentence[:160])
        return phrases[:8] or ([text[:120].strip()] if text.strip() else [])
