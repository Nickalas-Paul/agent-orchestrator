"""Contract output evaluator — LLM-as-a-judge for contract review quality."""

from __future__ import annotations

from typing import Any

from packages.core.evaluation import BaseOutputEvaluator
from packages.core.types.schemas import AgentResponse
from packages.core.utils import clamp_confidence
from packages.domain_contract.prompts.evaluator_prompt import (
    DEFAULT_EVALUATION_CRITERIA,
    get_contract_evaluator_prompt,
    get_contract_evaluator_user_prompt,
)


class ContractOutputEvaluator(BaseOutputEvaluator):
    """Evaluates contract review quality across risk, compliance, and comparison results."""

    AGENT_NAME = "contract_evaluator"
    DEFAULT_CRITERIA = list(DEFAULT_EVALUATION_CRITERIA)

    def _build_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """Build contract evaluator system and user prompts."""
        risk_result = kwargs.get("risk_result", {})
        compliance_result = kwargs.get("compliance_result", {})
        comparison_result = kwargs.get("comparison_result", {})

        system_prompt = get_contract_evaluator_prompt(self._criteria)
        user_prompt = get_contract_evaluator_user_prompt(
            risk_result=risk_result,
            compliance_result=compliance_result,
            comparison_result=comparison_result,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _post_parse(
        self, result: dict[str, Any], confidence: float
    ) -> tuple[dict[str, Any], float]:
        """Clamp per-criterion scores if present."""
        criterion_scores_raw = result.get("criterion_scores", {})
        criterion_scores: dict[str, float] = {}
        if isinstance(criterion_scores_raw, dict):
            for key, value in criterion_scores_raw.items():
                try:
                    criterion_scores[str(key)] = clamp_confidence(float(value))
                except (TypeError, ValueError):
                    continue
        result["criterion_scores"] = criterion_scores
        return result, confidence

    async def evaluate_outputs(
        self,
        risk_result: dict[str, Any],
        compliance_result: dict[str, Any],
        comparison_result: dict[str, Any],
        session_id: str = "",
    ) -> AgentResponse:
        """Evaluate contract specialist outputs. Preserves original method signature."""
        return await self.evaluate(
            risk_result=risk_result,
            compliance_result=compliance_result,
            comparison_result=comparison_result,
            session_id=session_id,
        )
