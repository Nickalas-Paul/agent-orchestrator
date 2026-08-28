"""Prompt templates for contract specialist agents."""

from packages.domain_contract.prompts.analyst_prompt import (
    get_analyst_prompt,
    get_analyst_user_prompt,
)
from packages.domain_contract.prompts.checker_prompt import (
    get_checker_prompt,
    get_checker_user_prompt,
)
from packages.domain_contract.prompts.comparator_prompt import (
    DEFAULT_STANDARD_TERMS,
    get_comparator_prompt,
    get_comparator_user_prompt,
)
from packages.domain_contract.prompts.evaluator_prompt import (
    DEFAULT_EVALUATION_CRITERIA,
    get_contract_evaluator_prompt,
    get_contract_evaluator_user_prompt,
)

__all__ = [
    "DEFAULT_EVALUATION_CRITERIA",
    "DEFAULT_STANDARD_TERMS",
    "get_analyst_prompt",
    "get_analyst_user_prompt",
    "get_checker_prompt",
    "get_checker_user_prompt",
    "get_comparator_prompt",
    "get_comparator_user_prompt",
    "get_contract_evaluator_prompt",
    "get_contract_evaluator_user_prompt",
]
