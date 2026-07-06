"""Dynamic workflow capability: orchestrate sub-agents from a sandboxed Python script."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.experimental.dynamic_workflow._toolset import (
    DynamicWorkflowToolset,
    WorkflowAgent,
    WorkflowResourceLimits,
    index_workflow_agents,
    validate_workflow_agent,
)


@dataclass(kw_only=True)
class DynamicWorkflow(AbstractCapability[AgentDepsT]):
    """Capability that lets the model orchestrate named sub-agents from a Python script.

    Instead of delegating to one sub-agent per tool call, the model writes a single
    Python script (run in a Monty sandbox) that calls each sub-agent as an async
    function and composes the results -- fan out in parallel with `asyncio.gather`,
    chain one agent's output into the next, vote across several, or loop until done.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.experimental.dynamic_workflow import DynamicWorkflow

    reviewer = Agent('openai:gpt-5', name='reviewer', description='Review code for bugs.')
    summarizer = Agent('openai:gpt-5', name='summarizer', description='Summarize findings.')

    orchestrator = Agent(
        'openai:gpt-5',
        capabilities=[DynamicWorkflow(agents=[reviewer, summarizer])],
    )
    ```

    Sub-agents run as isolated runs (their own message history). The parent's `deps`
    are forwarded, and by default the parent's `usage` accumulator is shared so the whole
    tree's token and request spend is tallied in one place. For a hard cap on sub-agent
    runs, use the exact, host-enforced `max_agent_calls` ceiling -- a shared `usage_limits`
    is best-effort (see `forward_usage`). Workflows do not nest -- a sub-agent invoked from
    a workflow cannot start its own.

    Set `defer_loading=True` (with a stable `id`) to keep the orchestration tool and
    its sub-agent catalog out of the prompt until the model loads the capability --
    paying near-zero tokens on turns that don't need it.
    """

    agents: Sequence[AbstractAgent[AgentDepsT, Any] | WorkflowAgent[AgentDepsT]]
    """Sub-agents the orchestration script can call as async functions.

    This sequence is read at construction only. A raw agent entry is shorthand for
    `WorkflowAgent(agent)`, using the agent's own `name` and `description`. Use a
    `WorkflowAgent` entry when this workflow needs a per-use-site override.

    Later mutation of the passed sequence has no effect on the catalog. Use `reveal()`
    to add a sub-agent after construction.

    A revealed sub-agent becomes callable on the next step, announced to the model via an enqueued
    message, while the `run_workflow` description stays frozen at the agents present when the run
    started -- so the prompt-cache prefix never changes. Requires a `PendingMessageDrainCapability`
    in the run (auto-injected by Pydantic AI) so the announcement drains into the conversation.
    Reveal is append-only: a revealed sub-agent cannot be removed or hidden again for the rest of
    the run -- plan the catalog as monotonically growing.
    """

    _catalog: list[WorkflowAgent[AgentDepsT]] = field(init=False, repr=False)
    """Normalized catalog passed by reference to toolsets."""

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

    Requests within one sub-agent run happen one at a time, so request limits are enforced per
    run. With `forward_usage=False`, a per-sub-agent `total_tokens_limit` of `T` together with
    `max_agent_calls` of `N` bounds the whole agent tree to roughly `N * T` tokens; each run can
    overshoot its token limit by the final response, because core checks token limits after a
    response arrives. With `forward_usage=True` the limit is checked against the shared counter
    instead, a tree-wide cap that is best-effort under concurrent fan-out. `None` keeps the
    default (`request_limit=50`, no token limit).
    """

    resource_limits: WorkflowResourceLimits | Literal['unlimited'] | None = None
    """Sandbox limits guarding the orchestration script's own memory/allocations.

    `None` applies a safe backstop (256 MB, 50M allocations) with no wall-clock cap; `'unlimited'`
    removes all limits; a `WorkflowResourceLimits` mapping is merged onto the backstop, so a partial
    dict overrides only the caps it names and leaves the others at their backstop value.

    There is intentionally no default `max_duration_secs`: the sandbox's duration timer counts total
    wall-clock -- including time awaiting sub-agents fanned out with `asyncio.gather` -- so a default
    cap would abort ordinary parallel workflows, not just a runaway. Set one explicitly to bound a
    whole orchestration's runtime (also the only guard against a pure-CPU `while True`).
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Not spec-serializable: `agents` holds live Agent objects, not YAML-expressible config.
        return None

    def __post_init__(self) -> None:
        catalog = [self._normalize_workflow_agent(entry) for entry in self.agents]
        index_workflow_agents(catalog)
        self._catalog = catalog

    def _normalize_workflow_agent(
        self, entry: AbstractAgent[AgentDepsT, Any] | WorkflowAgent[AgentDepsT]
    ) -> WorkflowAgent[AgentDepsT]:
        """Normalize a public catalog entry to the internal wrapper form."""
        if isinstance(entry, WorkflowAgent):
            return entry
        return WorkflowAgent(agent=entry)

    def reveal(self, agent: AbstractAgent[AgentDepsT, Any] | WorkflowAgent[AgentDepsT]) -> None:
        """Reveal a sub-agent on the next model step.

        This is the supported runtime API for revealing sub-agents after a run has started.

        The revealed sub-agent is announced to the model on the next step and becomes callable
        then. The `run_workflow` tool description stays frozen at the agents present when the run
        started. Reveal is append-only: a revealed sub-agent cannot be removed or hidden again for
        the rest of the run. The sub-agent's resolved name must be a valid, unique sandbox
        function name; invalid entries raise `UserError` at the call site. If one
        `DynamicWorkflow` instance is shared across concurrent runs, `reveal()` reveals to all
        in-flight runs and joins the baseline catalog for runs that start afterwards.
        """
        entry = self._normalize_workflow_agent(agent)
        existing_names: set[str] = set()
        for catalog_entry in self._catalog:
            existing_names.add(validate_workflow_agent(catalog_entry, existing_names))
        validate_workflow_agent(entry, existing_names)
        self._catalog.append(entry)

    def get_toolset(self) -> DynamicWorkflowToolset[AgentDepsT]:
        """Provide the orchestration toolset to the agent."""
        return DynamicWorkflowToolset(
            # Toolsets keep this same list object; `reveal()` appends to it so in-flight
            # toolsets can fold in the new sub-agent on the next step.
            agents=self._catalog,
            tool_name=self.tool_name,
            max_agent_calls=self.max_agent_calls,
            max_retries=self.max_retries,
            forward_usage=self.forward_usage,
            sub_agent_usage_limits=self.sub_agent_usage_limits,
            resource_limits=self.resource_limits,
            toolset_id=self.id,
        )
