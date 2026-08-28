"""LLM-as-a-judge prompt templates for contract pipeline evaluation."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_EVALUATION_CRITERIA: list[str] = [
    "completeness",
    "consistency",
    "risk_identification_quality",
    "reasoning_depth",
    "source_citation_accuracy",
]

_CRITERION_GUIDANCE: dict[str, str] = {
    "completeness": (
        "Did the risk analyst surface major clause categories? "
        "Did compliance cover data protection and required clauses? "
        "Did terms comparison cover key commercial terms?"
    ),
    "consistency": (
        "Do risk, compliance, and terms outputs describe the same contract? "
        "Do severity ratings align across overlapping findings?"
    ),
    "risk_identification_quality": (
        "Are flagged risks material and well-scoped? "
        "Are false positives or missed high-severity issues evident?"
    ),
    "reasoning_depth": (
        "Do reasoning_trace fields explain WHY a clause is risky step by step? "
        "Are compliance explanations and impact assessments substantive?"
    ),
    "source_citation_accuracy": (
        "Do source_section and clause_reference values point to real sections? "
        "Are quoted contract snippets consistent with cited locations?"
    ),
}


def get_contract_evaluator_prompt(criteria: list[str] | None = None) -> str:
    """Return the system prompt for the Contract Output Evaluator agent.

    Args:
        criteria: Evaluation criteria names. Defaults to the contract five-point list.

    Returns:
        System prompt string instructing JSON-only judgment.
    """
    resolved = list(criteria) if criteria else list(DEFAULT_EVALUATION_CRITERIA)
    criterion_lines: list[str] = []
    for index, name in enumerate(resolved, start=1):
        guidance = _CRITERION_GUIDANCE.get(name, f"Evaluate the pipeline outputs for {name}.")
        criterion_lines.append(f"{index}. {name} — {guidance}")
    criteria_block = "\n".join(criterion_lines)

    return f"""You are an output evaluator for a multi-agent contract risk-review pipeline.
You judge the combined work of three specialist agents using LLM-as-a-judge reasoning.

Evaluate the pipeline outputs against these criteria:
{criteria_block}

For each criterion, think through the evidence in the specialist outputs before scoring.
Include per-criteria scores in criterion_scores. Scores must be between 0.0 and 1.0.
overall_confidence is your calibrated aggregate judgment.

Set recommendation to "approve" when the outputs are coherent and complete enough
to proceed. Set recommendation to "flag_for_review" when confidence is low, criteria
fail, or contradictions require a human reviewer.

Output ONLY valid JSON with this schema:
{{
  "overall_confidence": 0.0,
  "criterion_scores": {{"criterion_name": 0.0}},
  "findings": ["string"],
  "contradictions": ["string"],
  "summary": "string",
  "recommendation": "approve | flag_for_review"
}}
"""


def get_contract_evaluator_user_prompt(
    risk_result: dict[str, Any],
    compliance_result: dict[str, Any],
    comparison_result: dict[str, Any],
) -> str:
    """Build the user prompt from all three specialist agent outputs.

    Args:
        risk_result: Clause risk analyst output dict.
        compliance_result: Compliance checker output dict.
        comparison_result: Terms comparator output dict.

    Returns:
        User prompt string for the evaluator LLM call.
    """
    return (
        "Evaluate the following specialist-agent outputs.\n"
        "Use explicit reasoning for each criterion before scoring.\n\n"
        f"CLAUSE RISK RESULT:\n{json.dumps(risk_result, indent=2, default=str)}\n\n"
        f"COMPLIANCE RESULT:\n{json.dumps(compliance_result, indent=2, default=str)}\n\n"
        f"TERMS COMPARISON RESULT:\n{json.dumps(comparison_result, indent=2, default=str)}\n\n"
        "Respond with JSON only."
    )
