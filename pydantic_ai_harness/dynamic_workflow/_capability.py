"""Dynamic workflow capability: orchestrate sub-agents from a sandboxed Python script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.dynamic_workflow._toolset import (
    DynamicWorkflowToolset,
    WorkflowAgent,
    WorkflowResourceLimits,
)


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
    are forwarded, and by default the parent's `usage` accumulator is shared so the whole
    tree's token and request spend is tallied in one place. For a hard cap on sub-agent
    runs, use the exact, host-enforced `max_agent_calls` ceiling — a shared `usage_limits`
    is best-effort (see `forward_usage`). Workflows do not nest — a sub-agent invoked from
    a workflow cannot start its own.

    Set `defer_loading=True` (with a stable `id`) to keep the orchestration tool and
    its sub-agent catalog out of the prompt until the model loads the capability —
    paying near-zero tokens on turns that don't need it.
    """

    agents: list[WorkflowAgent[AgentDepsT]]
    """Sub-agents the orchestration script can call as async functions.

    Each `WorkflowAgent` bundles the agent with its sandbox function name (a valid
    Python identifier, unique across the workflow) and an optional catalog description.

    A `list` (not a read-only `Sequence`) because the host can keep a reference to it and append
    a `WorkflowAgent` mid-run to reveal it: it becomes callable on the next step, announced to the
    model via an enqueued message, while the `run_workflow` description stays frozen at the agents
    present when the run started — so the prompt-cache prefix never changes. Requires a
    `PendingMessageDrainCapability` in the run (auto-injected by Pydantic AI) so the announcement
    drains into the conversation. Reveal is **append-only**: a revealed sub-agent cannot be removed
    or hidden again for the rest of the run — plan the catalog as monotonically growing.
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
    """Share the parent run's `usage` accumulator with sub-agents, so the whole tree's token
    and request spend is tallied in one place.

    This does **not** forward the parent's configured `usage_limits` into sub-agent runs
    (`RunContext` does not expose the limit value): set `sub_agent_usage_limits` to bound
    sub-agents, and the parent re-checks its own `usage_limits` at its own request boundaries.
    Use `max_agent_calls` for an exact ceiling on sub-agent runs.
    """

    sub_agent_usage_limits: UsageLimits | None = None
    """`UsageLimits` applied to every sub-agent run, replacing pydantic-ai's default.

    Each sub-agent run is sequential, so its own limits are enforced exactly. With
    `forward_usage=False`, a per-sub-agent `total_tokens_limit` of `T` together with
    `max_agent_calls` of `N` bound the whole agent tree to at most `N * T` tokens — a hard
    ceiling. With `forward_usage=True` the limit is checked against the shared counter instead,
    a tree-wide cap that is best-effort under concurrent fan-out. `None` keeps the default
    (`request_limit=50`, no token limit).
    """

    resource_limits: WorkflowResourceLimits | Literal['unlimited'] | None = None
    """Sandbox limits guarding the orchestration script's own memory/allocations.

    `None` applies a safe backstop (256 MB, 50M allocations) with no wall-clock cap; `'unlimited'`
    removes all limits; a `WorkflowResourceLimits` mapping is merged onto the backstop, so a partial
    dict overrides only the caps it names and leaves the others at their backstop value.

    There is intentionally no default `max_duration_secs`: the sandbox's duration timer counts total
    wall-clock — including time awaiting sub-agents fanned out with `asyncio.gather` — so a default
    cap would abort ordinary parallel workflows, not just a runaway. Set one explicitly to bound a
    whole orchestration's runtime (also the only guard against a pure-CPU `while True`).
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
            sub_agent_usage_limits=self.sub_agent_usage_limits,
            resource_limits=self.resource_limits,
            toolset_id=self.id,
        )
