"""Dynamic workflow capability: orchestrate sub-agents from a sandboxed Python script."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_monty import ResourceLimits

from pydantic_ai_harness.dynamic_workflow._toolset import DynamicWorkflowToolset, WorkflowAgent


@dataclass(kw_only=True)
class DynamicWorkflow(AbstractCapability[AgentDepsT]):
    """Capability that lets the model orchestrate named sub-agents from a Python script.

    Instead of delegating to one sub-agent per tool call, the model writes a single
    Python script (run in a Monty sandbox) that calls each sub-agent as an async
    function and composes the results — fan out in parallel with `asyncio.gather`,
    chain one agent's output into the next, vote across several, or loop until done.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import DynamicWorkflow

    reviewer = Agent('openai:gpt-5', name='reviewer', instructions='Review code for bugs.')
    summarizer = Agent('openai:gpt-5', name='summarizer', instructions='Summarize findings.')

    orchestrator = Agent(
        'openai:gpt-5',
        capabilities=[
            DynamicWorkflow(
                agents=[WorkflowAgent(agent=reviewer), WorkflowAgent(agent=summarizer)],
            )
        ],
    )
    ```

    Sub-agents run as isolated runs (their own message history). The parent's `deps`
    are forwarded, and by default the parent's `usage` is shared so a parent
    `usage_limits` bounds the whole agent tree. An exact `max_agent_calls` ceiling is
    enforced host-side, and workflows do not nest — a sub-agent invoked from a workflow
    cannot start its own.

    Set `defer_loading=True` (with a stable `id`) to keep the orchestration tool and
    its sub-agent catalog out of the prompt until the model loads the capability —
    paying near-zero tokens on turns that don't need it.
    """

    agents: Sequence[WorkflowAgent[AgentDepsT]]
    """Sub-agents the orchestration script can call as async functions.

    Each `WorkflowAgent` bundles the agent with its sandbox function name (a valid
    Python identifier, unique across the workflow) and an optional catalog description.

    Pass a mutable `list` to reveal sub-agents at runtime: appending a `WorkflowAgent` mid-run
    (the host holds the list, typically via `deps`) makes it callable on the next step. The
    newcomer is announced to the model via an enqueued message, while the `run_workflow`
    description stays frozen at the agents present when the run started — so the prompt-cache
    prefix never changes. Requires a `PendingMessageDrainCapability` in the run (auto-injected
    by Pydantic AI) so the announcement drains into the conversation.
    """

    tool_name: str = 'run_workflow'
    """Name of the orchestration tool exposed to the model."""

    max_agent_calls: int = 50
    """Maximum total sub-agent runs per agent run (an exact, host-enforced ceiling).

    Unlike a parent `usage_limits`, this holds exactly even under concurrent fan-out.
    """

    max_retries: int = 3
    """Maximum retries for the orchestration tool (syntax/runtime errors count as retries)."""

    forward_usage: bool = True
    """Share the parent run's `usage` with sub-agents so `usage_limits` bounds the tree.

    Note: a shared `usage_limits` is best-effort under concurrent fan-out (sub-agents
    can pass the limit check before any of them increments it); use `max_agent_calls`
    for an exact ceiling.
    """

    resource_limits: ResourceLimits | None = None
    """Monty sandbox limits guarding the orchestration script's own CPU/memory.

    These bound the script itself (e.g. a runaway `while` loop), not sub-agent latency:
    `max_duration_secs` counts only the sandbox's CPU time, not time awaiting sub-agents.
    `None` applies a safe backstop (30s CPU, 256 MB); pass `{}` to disable all limits.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Not spec-serializable: `agents` holds live Agent objects, not YAML-expressible config.
        return None

    def get_toolset(self) -> DynamicWorkflowToolset[AgentDepsT] | None:
        """Provide the orchestration toolset to the agent."""
        return DynamicWorkflowToolset(
            agents=self.agents,
            tool_name=self.tool_name,
            max_agent_calls=self.max_agent_calls,
            max_retries=self.max_retries,
            forward_usage=self.forward_usage,
            resource_limits=self.resource_limits,
            toolset_id=self.id,
        )
