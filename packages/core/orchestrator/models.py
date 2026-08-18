"""Orchestrator-specific Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from packages.core.types.schemas import TaskDefinition


class AgentCapability(BaseModel):
    """Describes what an agent can do and which task types it supports."""

    agent_name: str
    description: str
    supported_task_types: list[str] = Field(default_factory=list)


class AgentRegistry(BaseModel):
    """In-memory registry of available agent capabilities."""

    registered_agents: dict[str, AgentCapability] = Field(default_factory=dict)

    def register(self, capability: AgentCapability) -> None:
        """Register or replace an agent capability."""
        self.registered_agents[capability.agent_name] = capability

    def list_agents(self) -> list[AgentCapability]:
        """Return all registered agent capabilities."""
        return list(self.registered_agents.values())


class DecompositionResult(BaseModel):
    """Structured result of request decomposition before plan materialization."""

    original_request: str
    reasoning: str = ""
    tasks: list[TaskDefinition] = Field(default_factory=list)
