"""Vendor output evaluator — LLM-as-a-judge for vendor evaluation quality."""

from __future__ import annotations

from typing import Any

from packages.core.evaluation import BaseOutputEvaluator
from packages.core.types.schemas import AgentResponse
from packages.domain_vendor.prompts.evaluator_prompt import (
    DEFAULT_EVALUATION_CRITERIA,
    get_vendor_evaluator_prompt,
    get_vendor_evaluator_user_prompt,
)


class VendorOutputEvaluator(BaseOutputEvaluator):
    """Evaluates vendor analysis quality across capability, pricing, and market results."""

    AGENT_NAME = "vendor_evaluator"
    DEFAULT_CRITERIA = list(DEFAULT_EVALUATION_CRITERIA)

    def _build_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """Build vendor evaluator system and user prompts."""
        capability_result = kwargs.get("capability_result", {})
        pricing_result = kwargs.get("pricing_result", {})
        market_result = kwargs.get("market_result", {})

        system_prompt = get_vendor_evaluator_prompt(self._criteria)
        user_prompt = get_vendor_evaluator_user_prompt(
            capability_result=capability_result,
            pricing_result=pricing_result,
            market_result=market_result,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def evaluate_outputs(
        self,
        capability_result: dict[str, Any],
        pricing_result: dict[str, Any],
        market_result: dict[str, Any],
        session_id: str = "",
    ) -> AgentResponse:
        """Evaluate vendor specialist outputs. Preserves original method signature."""
        return await self.evaluate(
            capability_result=capability_result,
            pricing_result=pricing_result,
            market_result=market_result,
            session_id=session_id,
        )
