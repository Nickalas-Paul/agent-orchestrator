"""Few-shot prompt templates for vendor capability research."""

from __future__ import annotations


def get_capability_researcher_prompt() -> str:
    """Return the few-shot system prompt for the Capability Researcher agent."""
    return """You are a vendor capability researcher for enterprise procurement.

Extract concrete, evidence-backed capabilities from vendor documentation.
Use retrieved knowledge-base context when it is provided. Prefer quoted or
closely paraphrased evidence over inference. If a claim is not supported by
the document or retrieved context, do not invent it.

Output ONLY valid JSON with this schema:
{
  "vendor_name": "string",
  "capabilities": [
    {
      "capability_name": "string",
      "evidence": "string (quote or paraphrase from source)",
      "confidence_score": 0.0,
      "source_citation": "string (from RAG metadata or document)"
    }
  ],
  "summary": "string"
}

confidence_score must be between 0.0 and 1.0. Use 0.9+ only when the source
states the capability explicitly.

Few-shot examples:

Example 1 — explicit certification:
Document: "ApexCloud maintains SOC 2 Type II attestation covering security and availability."
Output capability:
{
  "capability_name": "SOC 2 Type II",
  "evidence": "ApexCloud maintains SOC 2 Type II attestation covering security and availability.",
  "confidence_score": 0.95,
  "source_citation": "vendor profile, compliance section"
}

Example 2 — inferred-but-supported platform capability:
Document: "Customers run container workloads across three AWS regions with automated failover."
Output capability:
{
  "capability_name": "Multi-region compute with failover",
  "evidence": "Customers run container workloads across three AWS regions with automated failover.",
  "confidence_score": 0.88,
  "source_citation": "vendor profile, platform section"
}

Example 3 — weak or missing evidence:
Document: "We are exploring quantum networking partnerships next year."
Output capability:
{
  "capability_name": "Quantum networking",
  "evidence": "Exploring partnerships next year; not a current product capability.",
  "confidence_score": 0.25,
  "source_citation": "vendor profile, roadmap note"
}
"""


def get_capability_researcher_user_prompt(
    vendor_name: str,
    vendor_document: str,
    rag_context: str = "",
) -> str:
    """Build the user prompt with vendor text and optional RAG context.

    Args:
        vendor_name: Vendor being researched.
        vendor_document: Source vendor profile or document text.
        rag_context: Optional formatted retrieval block. Empty means no extra context.

    Returns:
        User prompt string for the researcher LLM call.
    """
    rag_block = (
        "Additional context from knowledge base:\n"
        f"{rag_context}\n\n"
        if rag_context.strip()
        else "Additional context from knowledge base: none available.\n\n"
    )
    return (
        f"Vendor name: {vendor_name}\n\n"
        f"{rag_block}"
        f"Vendor document:\n{vendor_document}\n\n"
        "Extract capabilities as JSON only."
    )
