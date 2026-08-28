"""Chain-of-thought prompt templates for standard-terms comparison."""

from __future__ import annotations

from typing import Any

DEFAULT_STANDARD_TERMS: list[dict[str, Any]] = [
    {
        "term_name": "liability_cap",
        "standard_language": (
            "Unlimited for IP infringement and confidentiality breaches; "
            "otherwise 24 months of fees paid."
        ),
    },
    {
        "term_name": "termination_notice",
        "standard_language": "60 days' prior written notice for convenience termination.",
    },
    {
        "term_name": "ip_ownership",
        "standard_language": (
            "Client owns all deliverables and custom work product upon payment; "
            "provider retains only pre-existing tools."
        ),
    },
    {
        "term_name": "confidentiality_duration",
        "standard_language": "5 years from disclosure; indefinite for trade secrets.",
    },
    {
        "term_name": "indemnification_scope",
        "standard_language": (
            "Broad indemnification including IP infringement, negligence, and "
            "third-party claims — not limited to willful misconduct."
        ),
    },
    {
        "term_name": "data_protection",
        "standard_language": (
            "Documented SOC 2 Type II (or equivalent) controls; GDPR/CCPA compliance "
            "as applicable — not merely commercially reasonable care."
        ),
    },
    {
        "term_name": "governing_law",
        "standard_language": "Delaware law with structured dispute resolution (negotiation then arbitration).",
    },
    {
        "term_name": "force_majeure",
        "standard_language": (
            "Prompt notice within 5 business days; payment obligations not excused; "
            "termination right if outage exceeds 60 days."
        ),
    },
]


def get_comparator_prompt() -> str:
    """Return the system prompt for the Terms Comparator agent."""
    terms_lines = "\n".join(
        f"- {item['term_name']}: {item['standard_language']}" for item in DEFAULT_STANDARD_TERMS
    )
    return f"""You are a contract terms comparator for enterprise legal review.

Compare the contract against preferred/standard terms using chain-of-thought reasoning.
For each term:
1. Restate the standard/preferred position.
2. Extract the contract's actual language.
3. Classify deviation_type: more_favorable | less_favorable | missing | equivalent
4. Assess commercial impact.

Fallback standard terms (use when no RAG context is provided):
{terms_lines}

When RAG / retrieved standard terms are provided in the user prompt, treat them as
the authoritative preferred position and prefer them over the fallback list.

Output ONLY valid JSON with this schema:
{{
  "deviations": [
    {{
      "term_name": "string",
      "standard_language": "string",
      "contract_language": "string",
      "deviation_type": "more_favorable|less_favorable|missing|equivalent",
      "impact_assessment": "string",
      "confidence_score": 0.0
    }}
  ],
  "more_favorable_count": 0,
  "less_favorable_count": 0,
  "missing_count": 0,
  "equivalent_count": 0,
  "overall_deviation_score": 0.0,
  "summary": "string"
}}

overall_deviation_score is 0.0 (all standard) to 1.0 (highly deviant).
confidence_score must be between 0.0 and 1.0. Counts must match the deviations array.
"""


def get_comparator_user_prompt(contract_text: str, rag_context: str = "") -> str:
    """Build the user prompt with contract text and optional RAG standard terms.

    Args:
        contract_text: Full contract document text.
        rag_context: Optional formatted retrieval of standard/preferred terms.

    Returns:
        User prompt string for the comparator LLM call.
    """
    rag_block = (
        "Authoritative standard/preferred terms from knowledge base:\n"
        f"{rag_context}\n\n"
        if rag_context.strip()
        else "Authoritative standard/preferred terms from knowledge base: none available. "
        "Use the fallback standard terms from the system prompt.\n\n"
    )
    return (
        f"{rag_block}"
        f"CONTRACT TEXT:\n{contract_text}\n\n"
        "Compare terms using chain-of-thought reasoning. Respond with JSON only."
    )
