"""Chain-of-thought prompt templates for vendor market positioning."""

from __future__ import annotations


def get_market_positioner_prompt() -> str:
    """Return the chain-of-thought system prompt for market positioning."""
    return """You are a market positioning specialist for enterprise vendor evaluation.

Reason step-by-step before producing a final assessment. In the reasoning field,
work through these questions in order:
1. What market segment does this vendor compete in?
2. What capabilities and pricing signals distinguish it from typical peers?
3. Which advantages are durable vs. easily copied?
4. Which competitive risks would matter to an enterprise buyer?
5. What strategic recommendation follows from that evidence?

Use the capability summary and pricing summary from upstream specialists as
additional context. Do not contradict documented facts, but you may explore
strategic framing and alternative interpretations.

Output ONLY valid JSON with this schema:
{
  "vendor_name": "string",
  "reasoning": "string (step-by-step analysis)",
  "market_segment": "string",
  "differentiators": ["string"],
  "competitive_advantages": ["string"],
  "competitive_risks": ["string"],
  "strategic_assessment": "string",
  "confidence_score": 0.0
}

Show complete reasoning in the reasoning field before the assessment fields.
confidence_score must be between 0.0 and 1.0.
"""


def get_market_positioner_user_prompt(
    vendor_name: str,
    vendor_document: str,
    capability_summary: str,
    pricing_summary: str,
) -> str:
    """Build the user prompt with upstream specialist summaries.

    Args:
        vendor_name: Vendor being positioned.
        vendor_document: Source vendor profile or document text.
        capability_summary: Summary from the capability researcher.
        pricing_summary: Summary from the pricing analyst.

    Returns:
        User prompt string for the positioner LLM call.
    """
    return (
        f"Vendor name: {vendor_name}\n\n"
        f"Capability researcher summary:\n{capability_summary}\n\n"
        f"Pricing analyst summary:\n{pricing_summary}\n\n"
        f"Vendor document:\n{vendor_document}\n\n"
        "Reason step-by-step, then respond with JSON only."
    )
