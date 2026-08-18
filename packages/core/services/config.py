"""Service configuration and factory helpers for managed AI adapters."""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from packages.core.services.base import DocumentProcessor, TextAnalyzer


class ServiceConfig(BaseModel):
    """Runtime configuration for Textract and Comprehend providers."""

    textract_provider: Literal["aws", "local"] = "local"
    comprehend_provider: Literal["aws", "local"] = "local"
    aws_region: str = "us-east-1"


def get_service_config(*, reload_env: bool = False) -> ServiceConfig:
    """Load service adapter configuration from environment variables.

    Args:
        reload_env: When True, reload ``.env`` before reading values.

    Returns:
        Populated ``ServiceConfig``.
    """
    load_dotenv(override=reload_env)
    textract = os.getenv("TEXTRACT_PROVIDER", "local").strip().lower()
    comprehend = os.getenv("COMPREHEND_PROVIDER", "local").strip().lower()
    if textract not in {"aws", "local"}:
        raise ValueError(
            f"Unsupported TEXTRACT_PROVIDER '{textract}'. Expected 'aws' or 'local'."
        )
    if comprehend not in {"aws", "local"}:
        raise ValueError(
            f"Unsupported COMPREHEND_PROVIDER '{comprehend}'. Expected 'aws' or 'local'."
        )
    return ServiceConfig(
        textract_provider=textract,  # type: ignore[arg-type]
        comprehend_provider=comprehend,  # type: ignore[arg-type]
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
    )


def get_document_processor(config: ServiceConfig | None = None) -> DocumentProcessor:
    """Factory that returns the configured document processor.

    Args:
        config: Optional explicit config; defaults to environment config.

    Returns:
        Concrete ``DocumentProcessor`` implementation.
    """
    from packages.core.services.textract import LocalDocumentProcessor, TextractProcessor

    resolved = config or get_service_config()
    if resolved.textract_provider == "aws":
        return TextractProcessor(region_name=resolved.aws_region)
    return LocalDocumentProcessor()


def get_text_analyzer(config: ServiceConfig | None = None) -> TextAnalyzer:
    """Factory that returns the configured text analyzer.

    Args:
        config: Optional explicit config; defaults to environment config.

    Returns:
        Concrete ``TextAnalyzer`` implementation.
    """
    from packages.core.services.comprehend import ComprehendAnalyzer, LocalTextAnalyzer

    resolved = config or get_service_config()
    if resolved.comprehend_provider == "aws":
        return ComprehendAnalyzer(region_name=resolved.aws_region)
    return LocalTextAnalyzer()
