"""RFP output evaluator — LLM-as-a-judge for RFP analysis quality."""

from __future__ import annotations

from typing import Any

from packages.core.evaluation import BaseOutputEvaluator
from packages.core.types.schemas import AgentResponse
from packages.domain_rfp.prompts.evaluator_prompt import (
    DEFAULT_EVALUATION_CRITERIA,
    get_evaluator_prompt,
    get_evaluator_user_prompt,
)


class OutputEvaluator(BaseOutputEvaluator):
    """Evaluates RFP analysis quality across extraction, mapping, and gap results."""

    AGENT_NAME = "evaluator"
    DEFAULT_CRITERIA = list(DEFAULT_EVALUATION_CRITERIA)

    def _build_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """Build RFP evaluator system and user prompts."""
        extraction_result = kwargs.get("extraction_result", {})
        mapping_result = kwargs.get("mapping_result", {})
        gap_result = kwargs.get("gap_result", {})

        system_prompt = get_evaluator_prompt(self._criteria)
        user_prompt = get_evaluator_user_prompt(
            extraction_result=extraction_result,
            mapping_result=mapping_result,
            gap_result=gap_result,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def evaluate_outputs(
        self,
        extraction_result: dict[str, Any],
        mapping_result: dict[str, Any],
        gap_result: dict[str, Any],
        session_id: str = "",
    ) -> AgentResponse:
        """Evaluate RFP specialist outputs. Preserves original method signature."""
        return await self.evaluate(
            extraction_result=extraction_result,
            mapping_result=mapping_result,
            gap_result=gap_result,
            session_id=session_id,
        )
