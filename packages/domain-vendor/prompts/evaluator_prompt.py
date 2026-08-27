"""LLM-as-a-judge prompt templates for vendor pipeline evaluation."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_EVALUATION_CRITERIA: list[str] = [
    "completeness",
    "consistency",
    "contradiction",
    "evidence_quality",
]

_CRITERION_GUIDANCE: dict[str, str] = {
    "completeness": (
        "Did the capability researcher extract a reasonable set of capabilities? "
        "Did pricing analysis cover model, costs, and risks? "
        "Did market positioning include segment, advantages, and risks?"
    ),
    "consistency": (
        "Do capability, pricing, and market outputs describe the same vendor? "
        "Do pricing claims match capabilities (e.g., enterprise tier vs. stated features)?"
    ),
    "contradiction": (
        "Does market positioning contradict capability evidence or pricing findings? "
        "Are risk flags ignored by the strategic assessment?"
    ),
    "evidence_quality": (
        "Are capabilities backed by evidence and citations? "
        "Are pricing figures sourced rather than invented? "
        "Does market reasoning support the assessment?"
    ),
}


def get_vendor_evaluator_prompt(criteria: list[str] | None = None) -> str:
    """Return the system prompt for the vendor output evaluator.

    Args:
        criteria: Evaluation criteria names. Defaults to the vendor four-point list.

    Returns:
        System prompt string instructing JSON-only judgment.
    """
    resolved = list(criteria) if criteria else list(DEFAULT_EVALUATION_CRITERIA)
    criterion_lines: list[str] = []
    for index, name in enumerate(resolved, start=1):
        guidance = _CRITERION_GUIDANCE.get(name, f"Evaluate the pipeline outputs for {name}.")
        criterion_lines.append(f"{index}. {name} — {guidance}")
    criteria_block = "\n".join(criterion_lines)

    return f"""You are an output evaluator for a multi-agent vendor evaluation pipeline.
You judge the combined work of three specialist agents using chain-of-thought reasoning.

Evaluate the pipeline outputs against these criteria:
{criteria_block}

For each criterion, think through the evidence in the specialist outputs before scoring.
Scores must be between 0.0 and 1.0. overall_confidence is your calibrated aggregate
judgment (not a naive mean if one criterion is severely failed).

Set recommendation to "approve" when the outputs are coherent and complete enough
to proceed. Set recommendation to "flag_for_review" when confidence is low, criteria
fail, or contradictions require a human reviewer.

Output ONLY valid JSON with this schema:
{{
  "overall_confidence": 0.0,
  "findings": ["string"],
  "contradictions": ["string"],
  "summary": "string",
  "recommendation": "approve | flag_for_review"
}}
"""


def get_vendor_evaluator_user_prompt(
    capability_result: dict[str, Any],
    pricing_result: dict[str, Any],
    market_result: dict[str, Any],
) -> str:
    """Build the user prompt from all three specialist agent outputs.

    Args:
        capability_result: Capability researcher output dict.
        pricing_result: Pricing analyst output dict.
        market_result: Market positioner output dict.

    Returns:
        User prompt string for the evaluator LLM call.
    """
    return (
        "Evaluate the following specialist-agent outputs.\n"
        "Use explicit reasoning for each criterion before scoring.\n\n"
        f"CAPABILITY RESEARCH RESULT:\n{json.dumps(capability_result, indent=2, default=str)}\n\n"
        f"PRICING ANALYSIS RESULT:\n{json.dumps(pricing_result, indent=2, default=str)}\n\n"
        f"MARKET POSITION RESULT:\n{json.dumps(market_result, indent=2, default=str)}\n\n"
        "Respond with JSON only."
    )
