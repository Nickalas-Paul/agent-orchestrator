"""Tests for DAG execution and orchestrator engine."""

from __future__ import annotations

import json

import pytest

from packages.core.cloud.base import ModelProvider, ModelResponse
from packages.core.metrics.tracker import MetricsTracker
from packages.core.orchestrator.dag import DAGExecutor, DAGValidationError
from packages.core.orchestrator.engine import OrchestratorEngine
from packages.core.orchestrator.models import AgentCapability
from packages.core.types.schemas import (
    AgentResponse,
    ExecutionPlan,
    TaskDefinition,
    TaskStatus,
    TokenUsage,
)


class MockModelProvider(ModelProvider):
    """Stub provider that returns a preconfigured content string."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[str] = []

    async def invoke(
        self,
        prompt: str,
        model_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
        *,
        agent_name: str = "unknown",
        session_id: str = "unknown",
    ) -> ModelResponse:
        self.calls.append(prompt)
        return ModelResponse(
            content=self.content,
            model_id=model_id or "mock-model",
            token_usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                model_id=model_id or "mock-model",
                estimated_cost_usd=0.0,
            ),
            latency_ms=1,
        )


def _plan_with_tasks(tasks: list[TaskDefinition]) -> ExecutionPlan:
    return ExecutionPlan(
        session_id="session-1",
        original_request="do work",
        tasks=tasks,
    )


def test_dag_validation_catches_cycles() -> None:
    plan = _plan_with_tasks(
        [
            TaskDefinition(
                task_id="a",
                description="A",
                agent_type="agent",
                dependencies=["b"],
            ),
            TaskDefinition(
                task_id="b",
                description="B",
                agent_type="agent",
                dependencies=["a"],
            ),
        ]
    )
    executor = DAGExecutor(plan)
    with pytest.raises(DAGValidationError, match="cycle"):
        executor.validate()


@pytest.mark.asyncio
async def test_dag_executes_tasks_in_dependency_order() -> None:
    plan = _plan_with_tasks(
        [
            TaskDefinition(
                task_id="t1",
                description="first",
                agent_type="agent",
                dependencies=[],
            ),
            TaskDefinition(
                task_id="t2",
                description="second",
                agent_type="agent",
                dependencies=["t1"],
            ),
            TaskDefinition(
                task_id="t3",
                description="third",
                agent_type="agent",
                dependencies=["t2"],
            ),
        ]
    )
    order: list[str] = []

    async def runner(task: TaskDefinition) -> AgentResponse:
        order.append(task.task_id)
        return AgentResponse(
            task_id=task.task_id,
            agent_name=task.agent_type,
            output={"ok": True},
            confidence_score=1.0,
        )

    responses = await DAGExecutor(plan).execute(runner)
    assert order == ["t1", "t2", "t3"]
    assert [r.task_id for r in responses] == ["t1", "t2", "t3"]
    assert all(t.status == TaskStatus.COMPLETED for t in plan.tasks)


@pytest.mark.asyncio
async def test_orchestrator_creates_valid_plan_from_mock_llm_response() -> None:
    payload = {
        "reasoning": "Split into research then draft",
        "tasks": [
            {
                "task_id": "research",
                "description": "Gather requirements",
                "agent_type": "research_agent",
                "dependencies": [],
                "priority": 2,
                "parameters": {"topic": "rfp"},
            },
            {
                "task_id": "draft",
                "description": "Draft response",
                "agent_type": "writer_agent",
                "dependencies": ["research"],
                "priority": 1,
                "parameters": {},
            },
        ],
    }
    provider = MockModelProvider(json.dumps(payload))
    engine = OrchestratorEngine(provider=provider, metrics=MetricsTracker())
    engine.register_agent(
        AgentCapability(
            agent_name="research_agent",
            description="Researches topics",
            supported_task_types=["research"],
        )
    )
    engine.register_agent(
        AgentCapability(
            agent_name="writer_agent",
            description="Writes drafts",
            supported_task_types=["write"],
        )
    )

    plan = await engine.decompose("Build an RFP response", session_id="s-1")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].task_id == "research"
    assert plan.tasks[1].dependencies == ["research"]
    assert plan.session_id == "s-1"
    DAGExecutor(plan).validate()


@pytest.mark.asyncio
async def test_orchestrator_fallback_single_task_when_parse_fails() -> None:
    provider = MockModelProvider("this is not json at all")
    engine = OrchestratorEngine(provider=provider, metrics=MetricsTracker())
    engine.register_agent(
        AgentCapability(
            agent_name="placeholder_agent",
            description="Fallback worker",
            supported_task_types=["general"],
        )
    )

    plan = await engine.decompose("Do the whole thing", session_id="s-2")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Do the whole thing"
    assert plan.tasks[0].agent_type == "placeholder_agent"
    assert plan.tasks[0].dependencies == []
