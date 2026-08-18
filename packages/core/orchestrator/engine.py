"""Main orchestrator engine: decomposition and DAG routing."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from packages.core.cloud.base import ModelProvider
from packages.core.logging.logger import get_logger
from packages.core.metrics.tracker import MetricsTracker
from packages.core.orchestrator.dag import DAGExecutor, TaskRunner
from packages.core.orchestrator.models import AgentCapability, AgentRegistry
from packages.core.types.schemas import (
    AgentResponse,
    ExecutionPlan,
    TaskDefinition,
    TaskStatus,
)

logger = get_logger("orchestrator.engine")


DECOMPOSITION_SYSTEM_PROMPT = """You are an orchestration planner for a multi-agent system.
Decompose the user request into subtasks that can be assigned to available agents.

Available agents:
{agent_catalog}

Return ONLY valid JSON with this schema:
{{
  "reasoning": "brief explanation of the plan",
  "tasks": [
    {{
      "task_id": "task-1",
      "description": "what to do",
      "agent_type": "exact agent_name from the catalog",
      "dependencies": [],
      "priority": 1,
      "parameters": {{}}
    }}
  ]
}}

Rules:
- Use only agent_type values from the available agents list.
- dependencies must reference other task_id values in the same plan.
- Prefer the smallest set of tasks that fully covers the request.
"""


class OrchestratorEngine:
    """Coordinates task decomposition and DAG-based agent routing."""

    def __init__(
        self,
        provider: ModelProvider,
        metrics: MetricsTracker | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            provider: Model provider used for decomposition.
            metrics: Optional metrics tracker.
        """
        self._provider = provider
        self._metrics = metrics or MetricsTracker()
        self._registry = AgentRegistry()

    def register_agent(self, capability: AgentCapability) -> None:
        """Add an agent capability to the registry.

        Args:
            capability: Agent capability descriptor.
        """
        self._registry.register(capability)
        logger.info(
            "agent_registered",
            agent_name=capability.agent_name,
            supported_task_types=capability.supported_task_types,
        )

    async def decompose(self, request: str, session_id: str) -> ExecutionPlan:
        """Decompose a user request into an ``ExecutionPlan``.

        If the LLM response cannot be parsed, falls back to a single-task plan.

        Args:
            request: Original user request.
            session_id: Session correlation id.

        Returns:
            Validated execution plan (may be a fallback single task).
        """
        agent_catalog = self._format_agent_catalog()
        system_prompt = DECOMPOSITION_SYSTEM_PROMPT.format(agent_catalog=agent_catalog)
        prompt = (
            f"{system_prompt}\n\n"
            f"User request:\n{request}\n\n"
            "Respond with JSON only."
        )

        response = await self._provider.invoke(
            prompt=prompt,
            agent_name="orchestrator",
            session_id=session_id,
        )
        plan = self._parse_plan(response.content, request=request, session_id=session_id)
        logger.info(
            "task_decomposed",
            session_id=session_id,
            plan_id=plan.plan_id,
            task_count=len(plan.tasks),
        )
        return plan

    async def execute(
        self,
        plan: ExecutionPlan,
        task_runner: TaskRunner,
    ) -> list[AgentResponse]:
        """Execute an execution plan with the provided task runner.

        Args:
            plan: Plan to execute.
            task_runner: Async callback for individual tasks.

        Returns:
            List of agent responses in execution order.
        """
        executor = DAGExecutor(plan)
        return await executor.execute(task_runner)

    async def run(
        self,
        request: str,
        session_id: str,
        task_runner: TaskRunner,
    ) -> list[AgentResponse]:
        """Decompose then execute a request end-to-end.

        Args:
            request: Original user request.
            session_id: Session correlation id.
            task_runner: Async callback for individual tasks.

        Returns:
            Agent responses from plan execution.
        """
        started = time.perf_counter()
        logger.info("request_received", session_id=session_id, request=request)
        plan = await self.decompose(request, session_id=session_id)
        logger.info(
            "execution_start",
            session_id=session_id,
            plan_id=plan.plan_id,
            task_count=len(plan.tasks),
        )
        results = await self.execute(plan, task_runner)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        summary = self._metrics.get_session_summary(session_id)
        logger.info(
            "execution_end",
            session_id=session_id,
            plan_id=plan.plan_id,
            result_count=len(results),
            total_time_ms=elapsed_ms,
            total_cost_usd=summary.get("estimated_cost_usd", 0.0),
            total_tokens=summary.get("total_tokens", 0),
        )
        return results

    def _format_agent_catalog(self) -> str:
        """Render registered agents into a prompt-friendly catalog."""
        agents = self._registry.list_agents()
        if not agents:
            return (
                "- placeholder_agent: Generic placeholder agent for undecomposed work "
                "(supported_task_types: general)"
            )
        lines: list[str] = []
        for agent in agents:
            types = ", ".join(agent.supported_task_types) or "general"
            lines.append(
                f"- {agent.agent_name}: {agent.description} "
                f"(supported_task_types: {types})"
            )
        return "\n".join(lines)

    def _parse_plan(
        self,
        content: str,
        *,
        request: str,
        session_id: str,
    ) -> ExecutionPlan:
        """Parse LLM JSON into an ExecutionPlan, with single-task fallback."""
        try:
            payload = self._extract_json_object(content)
            raw_tasks = payload.get("tasks", [])
            if not isinstance(raw_tasks, list) or not raw_tasks:
                raise ValueError("No tasks found in decomposition payload")

            tasks: list[TaskDefinition] = []
            for item in raw_tasks:
                if not isinstance(item, dict):
                    raise ValueError("Task entry is not an object")
                tasks.append(
                    TaskDefinition(
                        task_id=str(item.get("task_id") or item.get("id") or ""),
                        description=str(item.get("description", "")).strip() or request,
                        agent_type=str(
                            item.get("agent_type")
                            or item.get("agent_name")
                            or "placeholder_agent"
                        ),
                        dependencies=[
                            str(dep) for dep in item.get("dependencies", []) or []
                        ],
                        priority=int(item.get("priority", 0) or 0),
                        parameters=dict(item.get("parameters") or {}),
                        status=TaskStatus.PENDING,
                    )
                )

            # Ensure every task has an id.
            for index, task in enumerate(tasks, start=1):
                if not task.task_id:
                    task.task_id = f"task-{index}"

            plan = ExecutionPlan(
                session_id=session_id,
                original_request=request,
                tasks=tasks,
            )
            # Validate early so broken LLM plans fall through to fallback.
            DAGExecutor(plan).validate()
            return plan
        except Exception as exc:  # noqa: BLE001 - intentional fallback path
            logger.warning(
                "decomposition_parse_failed",
                session_id=session_id,
                error=str(exc),
            )
            return self._fallback_plan(request=request, session_id=session_id)

    def _fallback_plan(self, request: str, session_id: str) -> ExecutionPlan:
        """Build a single-task plan when structured decomposition fails."""
        default_agent = "placeholder_agent"
        agents = self._registry.list_agents()
        if agents:
            default_agent = agents[0].agent_name

        task = TaskDefinition(
            task_id="task-1",
            description=request,
            agent_type=default_agent,
            dependencies=[],
            priority=1,
            parameters={},
            status=TaskStatus.PENDING,
        )
        return ExecutionPlan(
            session_id=session_id,
            original_request=request,
            tasks=[task],
        )

    def _extract_json_object(self, content: str) -> dict[str, Any]:
        """Extract and parse the first JSON object from model content."""
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model response")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON was not an object")
        return data
