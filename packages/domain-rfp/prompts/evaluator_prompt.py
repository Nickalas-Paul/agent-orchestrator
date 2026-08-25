"""Chain-of-thought prompt templates for pipeline output evaluation."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_EVALUATION_CRITERIA: list[str] = [
    "completeness",
    "consistency",
    "contradiction",
    "reasoning_quality",
]

_CRITERION_GUIDANCE: dict[str, str] = {
    "completeness": (
        "Does the number of extracted requirements seem reasonable for the document size? "
        "Were any obvious categories missing (e.g., zero compliance requirements in a government RFP)?"
    ),
    "consistency": (
        "Does every extracted requirement have a corresponding capability mapping? "
        "Are there orphaned requirements or mappings?"
    ),
    "contradiction": (
        "Do any gap assessments contradict the capability mappings? "
        "(e.g., mapper says \"full match\" but gap analyzer flags it as a gap)"
    ),
    "reasoning_quality": (
        "Do the gap analyzer's reasoning traces support their conclusions? "
        "Are risk severity ratings justified?"
    ),
}


def get_evaluator_prompt(criteria: list[str] | None = None) -> str:
    """Return the system prompt for the Output Evaluator agent.

    Args:
        criteria: Evaluation criteria names. Defaults to the RFP four-point list.

    Returns:
        System prompt string instructing JSON-only chain-of-thought evaluation.
    """
    resolved = list(criteria) if criteria else list(DEFAULT_EVALUATION_CRITERIA)
    criterion_lines: list[str] = []
    for index, name in enumerate(resolved, start=1):
        guidance = _CRITERION_GUIDANCE.get(name, f"Evaluate the pipeline outputs for {name}.")
        criterion_lines.append(f"{index}. {name} — {guidance}")
    criteria_block = "\n".join(criterion_lines)

    return f"""You are an output evaluator for a multi-agent RFP analysis pipeline.
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
  "findings": [
    {{
      "criterion": "criterion_name",
      "score": 0.0,
      "reasoning": "step-by-step evidence-based reasoning"
    }}
  ],
  "contradictions": [],
  "summary": "executive summary of evaluation",
  "recommendation": "approve|flag_for_review"
}}

Include one findings entry for every criterion listed above.
"""


def get_evaluator_user_prompt(
    extraction_result: dict[str, Any],
    mapping_result: dict[str, Any],
    gap_result: dict[str, Any],
) -> str:
    """Build the user prompt from all three specialist agent outputs.

    Args:
        extraction_result: Requirements extractor output dict.
        mapping_result: Capability mapper output dict.
        gap_result: Gap analyzer output dict.

    Returns:
        User prompt string for the evaluator LLM call.
    """
    return (
        "Evaluate the following specialist-agent outputs.\n"
        "Use explicit reasoning for each criterion before scoring.\n\n"
        f"EXTRACTION RESULT:\n{json.dumps(extraction_result, indent=2, default=str)}\n\n"
        f"MAPPING RESULT:\n{json.dumps(mapping_result, indent=2, default=str)}\n\n"
        f"GAP ANALYSIS RESULT:\n{json.dumps(gap_result, indent=2, default=str)}\n\n"
        "Respond with JSON only."
    )
