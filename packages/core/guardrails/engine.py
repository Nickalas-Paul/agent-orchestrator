"""Shared guardrails engine for input validation and output safety."""

from __future__ import annotations

import re
from typing import Any

from packages.core.audit.logger import AuditLogger
from packages.core.audit.models import ActionType, AuditEntry
from packages.core.guardrails.models import (
    GuardrailAction,
    GuardrailCheckResult,
    GuardrailConfig,
    GuardrailResult,
    InputGuardrailCheck,
    OutputGuardrailCheck,
)
from packages.core.guardrails.patterns import (
    CONTENT_SAFETY_PATTERNS,
    INJECTION_PATTERNS,
    STOPWORDS,
)
from packages.core.logging.logger import get_logger
from packages.core.services.base import TextAnalyzer

logger = get_logger("core.guardrails")

_ACTION_RANK = {
    GuardrailAction.PASS: 0,
    GuardrailAction.FLAG: 1,
    GuardrailAction.BLOCK: 2,
}


class GuardrailsEngine:
    """Rule-based input/output safety checks for pipeline wiring."""

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        text_analyzer: TextAnalyzer | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            config: Optional check toggles and thresholds.
            text_analyzer: Optional NLP adapter for PII detection.
            audit_logger: Optional insert-only audit logger for FLAG/BLOCK results.
        """
        self._config = config or GuardrailConfig()
        self._text_analyzer = text_analyzer
        self._audit_logger = audit_logger

    async def check_input(
        self,
        text: str,
        domain: str = "",
        session_id: str = "",
    ) -> GuardrailResult:
        """Run all enabled input guardrail checks. Called before agent processing."""
        checks: list[GuardrailCheckResult] = []

        if self._config.enable_input_injection_detection:
            checks.append(self._check_injection(text))

        if self._config.enable_input_pii_detection and self._text_analyzer is not None:
            checks.append(await self._check_pii(text))

        topic_check = self._check_topic_boundary(text, domain)
        if topic_check is not None:
            checks.append(topic_check)

        result = self._aggregate(checks)
        self._maybe_audit(result=result, session_id=session_id, domain=domain, stage="input")
        return result

    async def check_output(
        self,
        text: str,
        retrieved_chunks: list[Any] | None = None,
        domain: str = "",
        session_id: str = "",
    ) -> GuardrailResult:
        """Run all enabled output guardrail checks. Called after agent processing."""
        checks: list[GuardrailCheckResult] = []

        if self._config.enable_output_content_safety:
            checks.append(self._check_content_safety(text))

        if self._config.enable_output_pii_detection and self._text_analyzer is not None:
            checks.append(await self._check_pii(text))

        if (
            self._config.enable_output_grounding
            and retrieved_chunks is not None
            and len(retrieved_chunks) > 0
        ):
            checks.append(self._check_grounding(text, retrieved_chunks))

        result = self._aggregate(checks)
        self._maybe_audit(result=result, session_id=session_id, domain=domain, stage="output")
        return result

    def _check_injection(self, text: str) -> GuardrailCheckResult:
        lowered = text.lower()
        matched = [pattern for pattern in INJECTION_PATTERNS if pattern in lowered]
        pattern_count = len(matched)
        if pattern_count == 0:
            confidence = 0.0
            action = GuardrailAction.PASS
            reason = "No prompt-injection patterns detected"
        elif pattern_count == 1:
            confidence = 0.7
            action = (
                GuardrailAction.BLOCK
                if confidence >= self._config.injection_confidence_threshold
                else GuardrailAction.PASS
            )
            reason = f"Prompt injection pattern detected: {matched[0]}"
        else:
            confidence = 0.9
            action = (
                GuardrailAction.BLOCK
                if confidence >= self._config.injection_confidence_threshold
                else GuardrailAction.PASS
            )
            reason = f"Multiple prompt injection patterns detected ({pattern_count})"

        return GuardrailCheckResult(
            check_name=InputGuardrailCheck.PROMPT_INJECTION.value,
            action=action,
            reason=reason,
            confidence=confidence,
            details={"matched_patterns": matched, "pattern_count": pattern_count},
        )

    async def _check_pii(self, text: str) -> GuardrailCheckResult:
        assert self._text_analyzer is not None
        entities = await self._text_analyzer.detect_pii(text)
        if not entities:
            return GuardrailCheckResult(
                check_name=InputGuardrailCheck.PII_DETECTION.value,
                action=GuardrailAction.PASS,
                reason="No PII entities detected",
                confidence=0.0,
                details={"pii_entities": []},
            )

        max_confidence = max(float(entity.confidence) for entity in entities) / 100.0
        action = self._config.pii_action
        entity_summaries = [
            {"type": entity.type, "confidence": entity.confidence} for entity in entities
        ]
        return GuardrailCheckResult(
            check_name=InputGuardrailCheck.PII_DETECTION.value,
            action=action,
            reason=f"Detected {len(entities)} PII entit{'y' if len(entities) == 1 else 'ies'}",
            confidence=max_confidence,
            details={"pii_entities": entity_summaries},
        )

    def _check_topic_boundary(
        self,
        text: str,
        domain: str,
    ) -> GuardrailCheckResult | None:
        if not self._config.blocked_topics:
            return None

        lowered = text.lower()
        matched: list[str] = []
        for topic in self._config.blocked_topics:
            topic_lower = topic.lower()
            phrase = topic_lower.replace("_", " ")
            if phrase in lowered or topic_lower in lowered:
                matched.append(topic)

        if matched:
            return GuardrailCheckResult(
                check_name=InputGuardrailCheck.TOPIC_BOUNDARY.value,
                action=GuardrailAction.BLOCK,
                reason=f"Blocked topic(s) detected: {', '.join(matched)}",
                confidence=0.95,
                details={"matched_topics": matched, "domain": domain},
            )
        return GuardrailCheckResult(
            check_name=InputGuardrailCheck.TOPIC_BOUNDARY.value,
            action=GuardrailAction.PASS,
            reason="No blocked topics detected",
            confidence=0.0,
            details={"matched_topics": [], "domain": domain},
        )

    def _check_content_safety(self, text: str) -> GuardrailCheckResult:
        lowered = text.lower()
        matched = [pattern for pattern in CONTENT_SAFETY_PATTERNS if pattern in lowered]
        indicator_count = len(matched)
        if indicator_count == 0:
            return GuardrailCheckResult(
                check_name=OutputGuardrailCheck.CONTENT_SAFETY.value,
                action=GuardrailAction.PASS,
                reason="No content-safety indicators detected",
                confidence=0.0,
                details={"matched_indicators": [], "indicator_count": 0},
            )

        confidence = 0.6 if indicator_count == 1 else 0.85
        action = GuardrailAction.BLOCK if confidence >= 0.6 else GuardrailAction.PASS
        return GuardrailCheckResult(
            check_name=OutputGuardrailCheck.CONTENT_SAFETY.value,
            action=action,
            reason=f"Content safety indicators matched ({indicator_count})",
            confidence=confidence,
            details={"matched_indicators": matched, "indicator_count": indicator_count},
        )

    def _check_grounding(
        self,
        text: str,
        retrieved_chunks: list[Any],
    ) -> GuardrailCheckResult:
        chunk_texts = [self._chunk_to_text(chunk) for chunk in retrieved_chunks]
        chunk_texts = [chunk for chunk in chunk_texts if chunk]
        sentences = self._split_sentences(text)
        if not sentences:
            return GuardrailCheckResult(
                check_name=OutputGuardrailCheck.GROUNDING.value,
                action=GuardrailAction.PASS,
                reason="No sentences to ground",
                confidence=1.0,
                details={
                    "grounding_score": 1.0,
                    "total_sentences": 0,
                    "grounded_sentences": 0,
                    "ungrounded_sentences": [],
                },
            )

        ungrounded: list[str] = []
        grounded_count = 0
        for sentence in sentences:
            if self._has_chunk_overlap(sentence, chunk_texts):
                grounded_count += 1
            else:
                ungrounded.append(sentence)

        total = len(sentences)
        grounding_score = grounded_count / total if total else 1.0
        if grounding_score < self._config.grounding_score_threshold:
            action = GuardrailAction.FLAG
            reason = (
                f"Low grounding score {grounding_score:.2f} "
                f"(threshold {self._config.grounding_score_threshold})"
            )
        else:
            action = GuardrailAction.PASS
            reason = f"Grounding score {grounding_score:.2f} meets threshold"

        return GuardrailCheckResult(
            check_name=OutputGuardrailCheck.GROUNDING.value,
            action=action,
            reason=reason,
            confidence=1.0 - grounding_score if action == GuardrailAction.FLAG else grounding_score,
            details={
                "grounding_score": grounding_score,
                "total_sentences": total,
                "grounded_sentences": grounded_count,
                "ungrounded_sentences": ungrounded,
            },
        )

    def _chunk_to_text(self, chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("text", "content", "chunk"):
                value = chunk.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return str(chunk)
        text_attr = getattr(chunk, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        nested = getattr(chunk, "chunk", None)
        nested_text = getattr(nested, "text", None) if nested is not None else None
        if isinstance(nested_text, str):
            return nested_text
        return str(chunk)

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _has_chunk_overlap(self, sentence: str, chunk_texts: list[str]) -> bool:
        sentence_tokens = self._tokenize(sentence)
        if len(sentence_tokens) < 3:
            # Short sentences: require all non-stopword tokens to appear in a chunk
            if not sentence_tokens:
                return True
            for chunk in chunk_texts:
                chunk_tokens = self._tokenize(chunk)
                if sentence_tokens.issubset(chunk_tokens):
                    return True
            return False

        for chunk in chunk_texts:
            chunk_tokens = self._tokenize(chunk)
            overlap = sentence_tokens & chunk_tokens
            if len(overlap) >= 3:
                return True
        return False

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {token for token in tokens if token not in STOPWORDS and len(token) > 1}

    def _aggregate(self, checks: list[GuardrailCheckResult]) -> GuardrailResult:
        if not checks:
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.PASS,
                checks=[],
                blocked_reasons=[],
                flagged_reasons=[],
            )

        worst = GuardrailAction.PASS
        blocked_reasons: list[str] = []
        flagged_reasons: list[str] = []
        for check in checks:
            if _ACTION_RANK[check.action] > _ACTION_RANK[worst]:
                worst = check.action
            if check.action == GuardrailAction.BLOCK:
                blocked_reasons.append(check.reason)
            elif check.action == GuardrailAction.FLAG:
                flagged_reasons.append(check.reason)

        return GuardrailResult(
            passed=worst != GuardrailAction.BLOCK,
            action=worst,
            checks=checks,
            blocked_reasons=blocked_reasons,
            flagged_reasons=flagged_reasons,
        )

    def _maybe_audit(
        self,
        *,
        result: GuardrailResult,
        session_id: str,
        domain: str,
        stage: str,
    ) -> None:
        if self._audit_logger is None:
            return
        if result.action not in {GuardrailAction.BLOCK, GuardrailAction.FLAG}:
            return

        self._audit_logger.log(
            AuditEntry(
                session_id=session_id or "unknown",
                agent_name="guardrails",
                action_type=ActionType.GUARDRAIL_CHECK,
                input_payload={"domain": domain, "stage": stage},
                output_payload=result.model_dump(),
                confidence_score=None,
            )
        )
        logger.info(
            "guardrail_check_audited",
            session_id=session_id,
            domain=domain,
            stage=stage,
            action=result.action.value,
        )
