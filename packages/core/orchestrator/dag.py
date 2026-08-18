"""DAG definition, validation, and execution for orchestrator plans."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from packages.core.logging.logger import get_logger
from packages.core.types.schemas import (
    AgentResponse,
    ExecutionPlan,
    TaskDefinition,
    TaskStatus,
)

logger = get_logger("orchestrator.dag")

TaskRunner = Callable[[TaskDefinition], Awaitable[AgentResponse]]


class DAGValidationError(ValueError):
    """Raised when an execution plan DAG is invalid."""


class DAGExecutor:
    """Validates and executes an ``ExecutionPlan`` respecting dependencies."""

    def __init__(self, plan: ExecutionPlan) -> None:
        """Create an executor for the given plan.

        Args:
            plan: Execution plan containing tasks and dependency edges.
        """
        self.plan = plan
        self._tasks_by_id: dict[str, TaskDefinition] = {
            task.task_id: task for task in plan.tasks
        }

    def validate(self) -> None:
        """Validate that the DAG has no cycles and all deps reference known tasks.

        Raises:
            DAGValidationError: If the plan is invalid.
        """
        if not self.plan.tasks:
            raise DAGValidationError("Execution plan contains no tasks")

        task_ids = set(self._tasks_by_id)
        if len(task_ids) != len(self.plan.tasks):
            raise DAGValidationError("Execution plan contains duplicate task_id values")

        for task in self.plan.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise DAGValidationError(
                        f"Task '{task.task_id}' depends on unknown task_id '{dep}'"
                    )
                if dep == task.task_id:
                    raise DAGValidationError(
                        f"Task '{task.task_id}' cannot depend on itself"
                    )

        if self._has_cycle():
            raise DAGValidationError("Execution plan DAG contains a cycle")

    def topological_levels(self) -> list[list[TaskDefinition]]:
        """Return tasks grouped into dependency levels (ready for parallelization).

        Returns:
            Ordered levels where each level's tasks have satisfied dependencies.
        """
        self.validate()
        remaining = {task.task_id: set(task.dependencies) for task in self.plan.tasks}
        completed: set[str] = set()
        levels: list[list[TaskDefinition]] = []

        while remaining:
            ready_ids = [tid for tid, deps in remaining.items() if deps <= completed]
            if not ready_ids:
                raise DAGValidationError("Unable to resolve execution order (cycle?)")

            level = [self._tasks_by_id[tid] for tid in ready_ids]
            # Stable-ish ordering: priority desc, then task_id.
            level.sort(key=lambda t: (-t.priority, t.task_id))
            levels.append(level)

            for tid in ready_ids:
                del remaining[tid]
                completed.add(tid)

        return levels

    async def execute(self, task_runner: TaskRunner) -> list[AgentResponse]:
        """Execute tasks in dependency order using ``task_runner``.

        Tasks within a level are executed sequentially for now, but grouped so
        async parallel execution can replace the inner loop later.

        Args:
            task_runner: Async callback that runs a single task.

        Returns:
            Agent responses in execution order.
        """
        levels = self.topological_levels()
        responses: list[AgentResponse] = []

        for level_index, level in enumerate(levels):
            logger.info(
                "dag_level_start",
                plan_id=self.plan.plan_id,
                level_index=level_index,
                task_count=len(level),
            )
            # Sequential within level today; structure preserves future fan-out.
            for task in level:
                task.status = TaskStatus.RUNNING
                logger.info(
                    "task_started",
                    plan_id=self.plan.plan_id,
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                )
                try:
                    response = await task_runner(task)
                    task.status = TaskStatus.COMPLETED
                    responses.append(response)
                    logger.info(
                        "task_completed",
                        plan_id=self.plan.plan_id,
                        task_id=task.task_id,
                        agent_name=response.agent_name,
                    )
                except Exception as exc:  # noqa: BLE001 - surface failure, continue policy later
                    task.status = TaskStatus.FAILED
                    logger.error(
                        "task_failed",
                        plan_id=self.plan.plan_id,
                        task_id=task.task_id,
                        error=str(exc),
                    )
                    raise

        return responses

    def _has_cycle(self) -> bool:
        """Detect cycles via DFS coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks_by_id}

        def visit(node: str) -> bool:
            color[node] = GRAY
            for dep in self._tasks_by_id[node].dependencies:
                # Edge direction: dependency -> dependent for topo; for cycle
                # detection we walk from task to its dependencies.
                if color[dep] == GRAY:
                    return True
                if color[dep] == WHITE and visit(dep):
                    return True
            color[node] = BLACK
            return False

        return any(color[node] == WHITE and visit(node) for node in self._tasks_by_id)
