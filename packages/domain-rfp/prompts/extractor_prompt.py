"""Zero-shot + negative prompt templates for requirements extraction."""

from __future__ import annotations

import json
from typing import Any


def get_extractor_prompt() -> str:
    """Return the system prompt for the Requirements Extractor agent."""
    return """You are a requirements extraction specialist for enterprise RFP documents.

Your job is to identify discrete, actionable requirements from source text. A requirement is:
- A mandatory capability the vendor must provide
- A compliance need or certification obligation
- A timeline constraint or milestone deadline
- A budget limit or commercial constraint
- A staffing or support obligation

Output ONLY valid JSON with this schema:
{
  "requirements": [
    {
      "requirement_id": "REQ-001",
      "text": "verbatim requirement text",
      "category": "technical|compliance|timeline|budget|staffing|other",
      "priority": "must-have|should-have|nice-to-have",
      "source_page": 1,
      "source_section": "section name or null",
      "entities": [],
      "pii_detected": false
    }
  ],
  "total_extracted": 0,
  "document_pages": 1,
  "extraction_confidence": 0.0,
  "pii_summary": {}
}

Negative instructions (must follow):
- Do not invent requirements not present in the source text.
- Do not combine multiple distinct requirements into a single item.
- Do not interpret preferences or suggestions as mandatory requirements unless explicitly stated.
- Do not summarize requirements — preserve the original language as closely as possible.
- Do not include background context or company descriptions as requirements.
"""


def get_extractor_user_prompt(
    document_text: str,
    entities: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user prompt with document text and optional entity metadata.

    Args:
        document_text: Extracted RFP document text.
        entities: Optional list of detected entity dicts from NLP analysis.

    Returns:
        User prompt string for the extractor LLM call.
    """
    entity_block = json.dumps(entities or [], indent=2)
    return (
        "Extract all requirements from the following RFP document text.\n\n"
        f"Detected entities (metadata only; do not invent requirements from these):\n"
        f"{entity_block}\n\n"
        f"DOCUMENT TEXT:\n{document_text}\n\n"
        "Respond with JSON only."
    )
