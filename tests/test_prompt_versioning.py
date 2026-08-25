"""Unit tests for in-memory PromptRegistry versioning."""

from __future__ import annotations

from packages.core.prompts.registry import PromptRegistry


def test_register_prompt_creates_version_1() -> None:
    registry = PromptRegistry()
    prompt = registry.register_prompt(
        "rfp_extractor_system",
        "You are an extractor.",
        "initial",
    )
    assert prompt.prompt_id == "rfp_extractor_system"
    assert prompt.version == 1
    assert prompt.is_active is True
    assert prompt.template_text == "You are an extractor."
    assert prompt.description == "initial"


def test_register_same_text_is_idempotent() -> None:
    registry = PromptRegistry()
    first = registry.register_prompt("rfp_mapper_system", "Map requirements.")
    second = registry.register_prompt("rfp_mapper_system", "Map requirements.")
    assert first.version == 1
    assert second.version == 1
    assert registry.get_prompt_version("rfp_mapper_system", 2) is None
    assert len(registry.list_prompts()) == 1


def test_register_different_text_increments_version() -> None:
    registry = PromptRegistry()
    first = registry.register_prompt("rfp_analyzer_system", "Analyze gaps v1.")
    second = registry.register_prompt("rfp_analyzer_system", "Analyze gaps v2.")
    assert first.version == 1
    assert second.version == 2
    assert second.is_active is True


def test_get_active_prompt_returns_latest() -> None:
    registry = PromptRegistry()
    registry.register_prompt("rfp_evaluator_system", "Evaluate v1.")
    registry.register_prompt("rfp_evaluator_system", "Evaluate v2.")
    active = registry.get_active_prompt("rfp_evaluator_system")
    assert active is not None
    assert active.version == 2
    assert active.template_text == "Evaluate v2."


def test_get_active_prompt_returns_none_for_unknown() -> None:
    registry = PromptRegistry()
    assert registry.get_active_prompt("does_not_exist") is None


def test_get_prompt_version_retrieves_specific_version() -> None:
    registry = PromptRegistry()
    registry.register_prompt("rfp_extractor_system", "Extractor v1.")
    registry.register_prompt("rfp_extractor_system", "Extractor v2.")
    version_one = registry.get_prompt_version("rfp_extractor_system", 1)
    assert version_one is not None
    assert version_one.version == 1
    assert version_one.template_text == "Extractor v1."


def test_list_prompts_returns_all_active() -> None:
    registry = PromptRegistry()
    registry.register_prompt("rfp_extractor_system", "Extract.")
    registry.register_prompt("rfp_mapper_system", "Map.")
    registry.register_prompt("rfp_mapper_system", "Map better.")
    active = {prompt.prompt_id: prompt.version for prompt in registry.list_prompts()}
    assert active == {"rfp_extractor_system": 1, "rfp_mapper_system": 2}
