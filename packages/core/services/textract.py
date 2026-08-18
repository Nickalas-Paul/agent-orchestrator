"""Amazon Textract document processor adapters (AWS + local mock)."""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from packages.core.logging.logger import get_logger
from packages.core.services.base import DocumentProcessor, TextBlock, TextractResult

logger = get_logger("services.textract")


class TextractProcessorError(Exception):
    """Base error for Textract processor failures."""


class TextractDocumentTooLargeError(TextractProcessorError):
    """Raised when the document exceeds Textract size limits."""


class TextractUnsupportedFormatError(TextractProcessorError):
    """Raised when the document format is not supported."""


class TextractThrottlingError(TextractProcessorError):
    """Raised when Textract throttles the request."""


class TextractProcessor(DocumentProcessor):
    """Real AWS Textract implementation."""

    def __init__(self, region_name: str = "us-east-1", client: Any | None = None) -> None:
        """Initialize the AWS Textract processor.

        Args:
            region_name: AWS region for the Textract client.
            client: Optional pre-built boto3 client (useful for tests).
        """
        self._client = client or boto3.client("textract", region_name=region_name)

    async def extract_text(
        self,
        document: bytes | str,
        document_type: str = "pdf",
    ) -> TextractResult:
        """Extract structured text via Textract detect + analyze APIs."""
        document_bytes = self._load_bytes(document)
        try:
            detect_response = self._client.detect_document_text(
                Document={"Bytes": document_bytes}
            )
            analyze_response = self._client.analyze_document(
                Document={"Bytes": document_bytes},
                FeatureTypes=["TABLES", "FORMS"],
            )
        except ClientError as exc:
            raise self._map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise TextractProcessorError(f"Textract request failed: {exc}") from exc

        result = self._parse_responses(detect_response, analyze_response)
        logger.info(
            "textract_extract_complete",
            pages=result.pages,
            block_count=len(result.blocks),
            document_type=document_type,
        )
        return result

    def _load_bytes(self, document: bytes | str) -> bytes:
        if isinstance(document, bytes):
            return document
        path = Path(document)
        if not path.exists():
            raise TextractUnsupportedFormatError(f"Document path not found: {document}")
        return path.read_bytes()

    def _parse_responses(
        self,
        detect_response: dict[str, Any],
        analyze_response: dict[str, Any],
    ) -> TextractResult:
        blocks_raw = list(detect_response.get("Blocks", [])) + list(
            analyze_response.get("Blocks", [])
        )
        # De-duplicate by Id when present.
        seen: set[str] = set()
        unique_blocks: list[dict[str, Any]] = []
        for block in blocks_raw:
            block_id = str(block.get("Id", ""))
            if block_id and block_id in seen:
                continue
            if block_id:
                seen.add(block_id)
            unique_blocks.append(block)

        parsed_blocks: list[TextBlock] = []
        pages = 1
        for block in unique_blocks:
            block_type = str(block.get("BlockType", "WORD"))
            page = int(block.get("Page", 1) or 1)
            pages = max(pages, page)
            text = str(block.get("Text", "")) if "Text" in block else ""
            if block_type in {"LINE", "WORD"} and not text:
                continue
            parsed_blocks.append(
                TextBlock(
                    text=text,
                    block_type=block_type,
                    confidence=float(block.get("Confidence", 0.0) or 0.0),
                    page=page,
                    geometry=block.get("Geometry"),
                )
            )

        line_blocks = [b for b in parsed_blocks if b.block_type == "LINE"]
        raw_text = "\n".join(b.text for b in line_blocks if b.text)
        tables = self._extract_tables(analyze_response.get("Blocks", []))
        key_values = self._extract_key_values(analyze_response.get("Blocks", []))
        return TextractResult(
            pages=pages,
            blocks=parsed_blocks,
            raw_text=raw_text,
            tables=tables,
            key_value_pairs=key_values,
        )

    def _extract_tables(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a simplified list of table metadata (full cell reconstruction deferred)."""
        tables: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("BlockType") == "TABLE":
                tables.append(
                    {
                        "id": block.get("Id"),
                        "page": block.get("Page", 1),
                        "confidence": block.get("Confidence", 0.0),
                        "rows": [],
                    }
                )
        return tables

    def _extract_key_values(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return simplified KEY_VALUE_SET entries."""
        pairs: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("BlockType") == "KEY_VALUE_SET":
                pairs.append(
                    {
                        "id": block.get("Id"),
                        "entity_types": block.get("EntityTypes", []),
                        "confidence": block.get("Confidence", 0.0),
                        "page": block.get("Page", 1),
                    }
                )
        return pairs

    def _map_client_error(self, exc: ClientError) -> TextractProcessorError:
        error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = str(error.get("Code", ""))
        message = str(error.get("Message", str(exc)))
        if code in {"DocumentTooLargeException", "LimitExceededException"}:
            return TextractDocumentTooLargeError(message)
        if code in {"UnsupportedDocumentException", "BadDocumentException", "InvalidParameterException"}:
            return TextractUnsupportedFormatError(message)
        if code in {"ProvisionedThroughputExceededException", "ThrottlingException"}:
            return TextractThrottlingError(message)
        return TextractProcessorError(f"Textract error ({code}): {message}")


class LocalDocumentProcessor(DocumentProcessor):
    """Local/mock document processor for development without AWS credentials."""

    async def extract_text(
        self,
        document: bytes | str,
        document_type: str = "pdf",
    ) -> TextractResult:
        """Extract text locally from files, with mock fallback for bytes/failures."""
        logger.info(
            "local_document_processor_mode",
            document_type=document_type,
            input_kind="path" if isinstance(document, str) else "bytes",
        )

        text: str | None = None
        if isinstance(document, str):
            path = Path(document)
            if path.exists() and path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8")
            elif path.exists() and path.suffix.lower() == ".pdf":
                text = self._extract_pdf_text(path)
            elif path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = None
        elif isinstance(document, bytes):
            try:
                decoded = document.decode("utf-8")
                if decoded.strip() and "\x00" not in decoded[:200]:
                    text = decoded
            except UnicodeDecodeError:
                text = None

        if not text:
            text = self._mock_document_text(document, document_type)

        return self._text_to_result(text)

    def _extract_pdf_text(self, path: Path) -> str | None:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("pypdf2_unavailable", path=str(path))
            return None
        try:
            reader = PdfReader(str(path))
            parts = [(page.extract_text() or "") for page in reader.pages]
            joined = "\n".join(parts).strip()
            return joined or None
        except Exception as exc:  # noqa: BLE001 - fall back to mock
            logger.warning("pdf_extraction_failed", path=str(path), error=str(exc))
            return None

    def _mock_document_text(self, document: bytes | str, document_type: str) -> str:
        seed = hashlib.sha256(
            document if isinstance(document, bytes) else document.encode("utf-8")
        ).hexdigest()[:8]
        return (
            "LOCAL MOCK DOCUMENT EXTRACTION\n"
            f"Document type: {document_type}\n"
            f"Seed: {seed}\n\n"
            "1. The vendor shall provide REST API integration capabilities.\n"
            "2. System availability must meet a 99.9% monthly uptime SLA.\n"
            "3. All data at rest must be encrypted using AES-256.\n"
            "4. The solution must maintain SOC 2 Type II compliance.\n"
            "5. Implementation must be completed by December 31, 2026.\n"
        )

    def _text_to_result(self, text: str) -> TextractResult:
        lines = [line for line in text.splitlines() if line.strip()]
        pages = max(1, (len(lines) // 40) + 1)
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        blocks: list[TextBlock] = []

        for page_num in range(1, pages + 1):
            blocks.append(
                TextBlock(
                    text="",
                    block_type="PAGE",
                    confidence=99.0,
                    page=page_num,
                    geometry={"BoundingBox": {"Width": 1.0, "Height": 1.0, "Left": 0.0, "Top": 0.0}},
                )
            )

        for index, line in enumerate(lines):
            page = min(pages, (index // 40) + 1)
            line_confidence = round(rng.uniform(85.0, 99.0), 2)
            blocks.append(
                TextBlock(
                    text=line,
                    block_type="LINE",
                    confidence=line_confidence,
                    page=page,
                    geometry={
                        "BoundingBox": {
                            "Width": 0.8,
                            "Height": 0.02,
                            "Left": 0.1,
                            "Top": 0.05 + (index % 40) * 0.02,
                        }
                    },
                )
            )
            for word in re.findall(r"\S+", line):
                blocks.append(
                    TextBlock(
                        text=word,
                        block_type="WORD",
                        confidence=round(min(99.0, line_confidence + rng.uniform(-2, 2)), 2),
                        page=page,
                        geometry=None,
                    )
                )

        return TextractResult(
            pages=pages,
            blocks=blocks,
            raw_text="\n".join(lines),
            tables=[],
            key_value_pairs=[],
        )
