"""Toolset for the `DynamicWorkflow` capability.

Exposes a single `run_workflow` tool: the model writes a Python orchestration
script (run in a Monty sandbox) that calls named sub-agents as async functions
and composes their results — fan-out, chaining, voting, loops — in one step.
"""

from __future__ import annotations

import contextvars
import keyword
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Generic, Literal

from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_ai.usage import UsageLimits
from pydantic_core import to_jsonable_python
from typing_extensions import TypedDict

try:
    from pydantic_monty import MontyRepl, MontyRuntimeError, MontySyntaxError, ResourceLimits
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for DynamicWorkflow. Install it with: pip install "pydantic-ai-harness[code-mode]"'
    ) from _import_error

from pydantic_ai_harness._monty_exec import MontyExecutor, PrintCapture

# Set while a workflow script is executing, so a sub-agent that itself tries to run a workflow can
# be refused — workflows do not nest. asyncio copies the context into each task `asyncio.gather`
# schedules, so concurrently-dispatched sub-agents inherit this flag (the capability is asyncio-only).
_in_workflow: contextvars.ContextVar[bool] = contextvars.ContextVar('pydantic_ai_harness_in_workflow', default=False)


class WorkflowResourceLimits(TypedDict, total=False):
    """Caps on the orchestration script's own sandbox resources (not sub-agent latency).

    A harness-owned view of the sandbox limits the capability supports, so the public API does
    not depend on the underlying sandbox's own types. Every field is optional; an omitted field
    keeps its backstop value.
    """

    max_duration_secs: float
    """Maximum total wall-clock seconds for the script — **including** time spent awaiting
    sub-agents dispatched concurrently with `asyncio.gather` (the sandbox's duration timer accrues
    across that suspension; it only excludes the wait for sub-agents awaited one at a time). There
    is no default cap, because one would also kill ordinary parallel fan-out, not just a runaway.
    Set this only to put a hard ceiling on a whole orchestration's runtime; it is also the only
    guard against a pure-CPU `while True` loop, which otherwise blocks the event loop."""

    max_memory: int
    """Maximum sandbox memory, in bytes."""

    max_allocations: int
    """Maximum number of sandbox allocations."""


def _default_resource_limits() -> ResourceLimits:
    """Backstop limits for the sandbox itself.

    Caps memory and allocations to stop an accidental runaway that builds unbounded data.
    Deliberately omits `max_duration_secs`: in the sandbox that limit counts total wall-clock,
    including time suspended awaiting concurrently-gathered sub-agents, so a default cap would
    abort ordinary parallel fan-out. Set `resource_limits` explicitly to add a wall-clock ceiling.
    """
    return {
        'max_memory': 256 * 1024 * 1024,
        'max_allocations': 50_000_000,
    }


def _resolve_resource_limits(limits: WorkflowResourceLimits | Literal['unlimited'] | None) -> ResourceLimits:
    """Resolve the public `resource_limits` value to the limits handed to the sandbox.

    - `None` → the safe backstop (`_default_resource_limits`).
    - `'unlimited'` → no limits at all (an empty mapping).
    - a `WorkflowResourceLimits` mapping → merged *onto* the backstop, so a partial dict overrides
      only the caps it names and the others keep their backstop value (a `{'max_memory': ...}`
      never silently drops the duration backstop).
    """
    if limits is None:
        return _default_resource_limits()
    if isinstance(limits, str):  # 'unlimited'
        return {}
    return {**_default_resource_limits(), **limits}


class _WorkflowArguments(TypedDict):
    code: Annotated[str, Field(description='The Python orchestration script to execute in the sandbox.')]


_WORKFLOW_ARGS_ADAPTER = TypeAdapter(_WorkflowArguments)
_WORKFLOW_ARGS_JSON_SCHEMA = _WORKFLOW_ARGS_ADAPTER.json_schema()
_WORKFLOW_ARGS_VALIDATOR: SchemaValidatorProt = _WORKFLOW_ARGS_ADAPTER.validator  # pyright: ignore[reportAssignmentType]

_WORKFLOW_BASE_DESCRIPTION = """\
Write and run a Python orchestration script in a sandbox to coordinate multiple sub-agents.

Use this to break a task across specialized sub-agents and combine their results in a single step —
fan work out in parallel, chain one agent's output into the next, vote across several, or loop until
done — instead of delegating to one sub-agent at a time.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes** and **no third-party libraries**.
- **Useful standard-library modules**: `asyncio`, `math`, `json`, `re`, `typing`. Import what you use
  at the top of the script. Other modules are unavailable or stubbed — don't rely on them.
- **No wall-clock or timing primitives** (`asyncio.sleep`, `datetime.now()`, the `time` module).

Each sub-agent below is an async function. Call it with the `task` keyword argument — write
`reviewer(task="...")`, not `reviewer("...")`; all parameters are keyword-only. A sub-agent returns
that agent's output: a string by default, or — if it has a structured `output_type` — a dict, whose
fields you read by subscript (`r["field"]`), not attribute (`r.field`). Run several at once with
`asyncio.gather` rather than awaiting each sequentially:

```python
import asyncio
reviews = await asyncio.gather(reviewer(task="check auth"), reviewer(task="check parsing"))
```

`asyncio.gather` does **not** support `return_exceptions=True`, and a sub-agent that raises cannot be
caught inside the script: one failure aborts the whole script and you retry it. Design the script so
sub-agents don't depend on catching each other's errors.

The last expression's value is captured as the result — you do **not** need to `print()` it, and
printing produces a string representation, not structured data. Use `print()` only for debug logging.
If `print()` was also called, the result is returned as `{"output": "<printed text>", "result": <last
expression>}`.\
"""


def _is_valid_sandbox_name(name: str) -> bool:
    """Whether `name` can be exposed as a sandbox function: a non-keyword Python identifier.

    `str.isidentifier()` alone is not enough — Python keywords (`for`, `class`, `async`, ...) are
    valid identifiers but cannot be used as function names, so the model could never call them.
    Callers guard the empty/`None` case before this is reached.
    """
    return name.isidentifier() and not keyword.iskeyword(name)


# Every sub-agent is exposed with the same fixed signature — `(*, task: str) -> Any` — so build it once
# and render each catalog entry through core's `FunctionSignature` (the renderer code_mode and
# Pydantic AI already use). This keeps the catalog format consistent across capabilities, forces
# keyword-only `task` to match `dispatch` (which reads `kwargs['task']`), and renders docstrings
# safely — a hand-rolled f-string breaks on a newline or a quote inside a description.
_SUB_AGENT_PARAMS_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {'task': {'type': 'string'}},
    'required': ['task'],
}
_SUB_AGENT_SIGNATURE = FunctionSignature.from_schema(name='_', parameters_schema=_SUB_AGENT_PARAMS_SCHEMA)


def _render_agent_block(name: str, description: str | None) -> str:
    """Render one sub-agent as the async function signature shown to the model."""
    return _SUB_AGENT_SIGNATURE.render('...', name=name, description=description, is_async=True)


def _render_catalog(catalog: Mapping[str, str | None]) -> str:
    """Render the available sub-agents as async function signatures for the tool description."""
    blocks = [_render_agent_block(name, description) for name, description in catalog.items()]
    listing = '```python\n' + '\n\n'.join(blocks) + '\n```'
    return f'{_WORKFLOW_BASE_DESCRIPTION}\n\nAvailable sub-agents:\n\n{listing}'


def _render_reveal(name: str, description: str | None, tool_name: str) -> str:
    """Announcement enqueued when a sub-agent is revealed mid-run.

    Delivered as a conversation message (not folded into the cached tool description), so the
    prompt-cache prefix stays stable while the model still learns the agent is now callable.
    """
    block = _render_agent_block(name, description)
    return f'A new sub-agent is now available to call from inside the `{tool_name}` script:\n\n```python\n{block}\n```'


def _workflow_result(result: Any, printed: str) -> Any:
    """Shape the tool return: the script's result, its captured `print()` output, or both."""
    if not printed:
        return result if result is not None else {}
    if result is None:
        return {'output': printed}
    return {'output': printed, 'result': result}


@dataclass(frozen=True, kw_only=True)
class WorkflowAgent(Generic[AgentDepsT]):
    """One sub-agent exposed to the orchestration script as an async function.

    Bundles the agent with its sandbox function name and catalog description so the
    three travel together — there is no second collection to keep in sync.
    """

    agent: AbstractAgent[AgentDepsT, Any]
    """The sub-agent to run when the script calls this function."""

    name: str | None = None
    """Sandbox function name; must be a valid Python identifier and unique across the
    workflow. Falls back to the agent's `name`."""

    description: str | None = None
    """Description shown to the model in the sub-agent catalog, rendered as the sandbox
    function's docstring. When omitted, the model sees only the bare signature — set this
    to tell the model what the sub-agent does and what to pass as `task`."""

    @property
    def resolved_name(self) -> str | None:
        """The sandbox function name: the explicit `name`, else the agent's `name`."""
        return self.name or self.agent.name


@dataclass(kw_only=True)
class DynamicWorkflowToolset(AbstractToolset[AgentDepsT]):
    """Single-tool toolset that runs sub-agent orchestration scripts in a Monty sandbox."""

    agents: list[WorkflowAgent[AgentDepsT]]
    """Sub-agents callable from the orchestration script, each as an async function.

    A `list` (not a read-only `Sequence`) because the host can append a `WorkflowAgent` mid-run
    to reveal it: it becomes callable on the next step, announced to the model via an enqueued
    message, while the tool description stays frozen at the agents present when the run started
    (so the prompt-cache prefix never changes). Reveal is append-only — a revealed sub-agent
    cannot be removed or hidden again within the run.
    """

    tool_name: str = 'run_workflow'
    """Name of the tool exposed to the model."""

    max_agent_calls: int = 50
    """Maximum total sub-agent runs per agent run (an exact, host-enforced ceiling)."""

    max_retries: int = 3
    """Maximum retries for the `run_workflow` tool (syntax/runtime errors count as retries)."""

    forward_usage: bool = True
    """Share the parent run's `usage` accumulator with sub-agents, so the whole tree's token and
    request spend is tallied in one place. Note: the parent's configured `usage_limits` value is
    not forwarded into sub-agent runs (`RunContext` does not expose it); set `sub_agent_usage_limits`
    to bound sub-agents, and use `max_agent_calls` for an exact ceiling on sub-agent runs."""

    sub_agent_usage_limits: UsageLimits | None = None
    """`UsageLimits` applied to every sub-agent run, replacing pydantic-ai's default.

    Each sub-agent run is sequential, so its own limits are enforced exactly. With
    `forward_usage=False`, a per-sub-agent `total_tokens_limit` of `T` together with
    `max_agent_calls` of `N` bound the whole tree to at most `N * T` tokens — a hard ceiling. With
    `forward_usage=True` the limit is checked against the shared counter instead, giving a tree-wide
    cap that is best-effort under concurrent fan-out (sub-agents can pass the check before any of
    them increments it). `None` keeps the default (`request_limit=50`, no token limit)."""

    resource_limits: WorkflowResourceLimits | Literal['unlimited'] | None = None
    """Sandbox limits guarding the orchestration script's own memory/allocations (not sub-agents).

    `None` applies a safe backstop (256 MB, 50M allocations) with no wall-clock cap; `'unlimited'`
    removes all limits; a `WorkflowResourceLimits` mapping is merged onto the backstop, so a partial
    dict overrides only the caps it names and the others keep their backstop value. See
    `WorkflowResourceLimits.max_duration_secs` for why there is no default duration cap.
    """

    toolset_id: str | None = None
    """Stable toolset id (used for durable execution); defaults to the tool name."""

    # Per-run count of sub-agent calls. Reset on `for_run`, preserved across `for_run_step`.
    _call_count: int = field(default=0, init=False, repr=False)

    # Sub-agents indexed by resolved sandbox name; seeded from `agents` in `__post_init__` and
    # extended in place as runtime appends to `agents` are revealed (`_reveal_pending`).
    _by_name: dict[str, WorkflowAgent[AgentDepsT]] = field(init=False, repr=False)

    # Catalog rendered into the (cached) tool description, frozen at run start. Built from the
    # agents present when the run began and never widened by reveals, so the description — and
    # thus the prompt-cache prefix — never changes mid-run.
    _baseline_catalog: dict[str, str | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.agents:
            raise UserError('DynamicWorkflow requires at least one sub-agent in `agents`.')
        self._by_name = {}
        for entry in self.agents:
            name = entry.resolved_name
            if not name:
                raise UserError(
                    'DynamicWorkflow sub-agent has no `name` and its agent has no `name`; '
                    'set `WorkflowAgent(name=...)` so it can be exposed as a sandbox function.'
                )
            if not _is_valid_sandbox_name(name):
                raise UserError(
                    f'DynamicWorkflow sub-agent name {name!r} cannot be exposed as a sandbox function: '
                    'it must be a Python identifier that is not a reserved keyword. Rename it.'
                )
            if name in self._by_name:
                raise UserError(f'DynamicWorkflow has two sub-agents named {name!r}; names must be unique.')
            self._by_name[name] = entry
        self._baseline_catalog = {name: entry.description for name, entry in self._by_name.items()}

    @property
    def id(self) -> str | None:
        return self.toolset_id or self.tool_name

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Fresh instance per run so the sub-agent-call budget is per-run."""
        return replace(self)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Preserve the per-run budget across steps."""
        return self

    def _reveal_pending(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reveal sub-agents appended to `agents` since the run started.

        Diffs the live `agents` list against the names already known (`_by_name`, which holds the
        baseline plus anything revealed so far) and folds each newcomer in — so `dispatch` resolves
        it — enqueuing an announcement for the model. The description renders from the frozen
        `_baseline_catalog`, never `_by_name`, so the cached prompt prefix is unaffected. A newcomer
        whose name is invalid or already taken is skipped; the announcement makes a successful
        reveal visible, so a silently-dropped one shows up as "the agent never appeared".

        Synchronous and await-free between snapshotting `agents` and mutating `_by_name`, so a
        concurrently-running `dispatch` never observes a half-revealed agent — the same
        await-free-critical-section reasoning that keeps `max_agent_calls` exact under fan-out.
        """
        for entry in tuple(self.agents):
            name = entry.resolved_name
            if not name or not _is_valid_sandbox_name(name) or name in self._by_name:
                continue
            self._by_name[name] = entry
            ctx.enqueue(_render_reveal(name, entry.description, self.tool_name))

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        self._reveal_pending(ctx)
        description = _render_catalog(self._baseline_catalog)
        return {
            self.tool_name: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=self.tool_name,
                    description=description,
                    parameters_json_schema=_WORKFLOW_ARGS_JSON_SCHEMA,
                    metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                    sequential=True,
                ),
                max_retries=self.max_retries,
                args_validator=_WORKFLOW_ARGS_VALIDATOR,
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if _in_workflow.get():
            raise ModelRetry(
                'Workflows do not nest: this sub-agent was invoked from a workflow and cannot start '
                'its own. Return your result to the orchestrating workflow instead.'
            )

        code = tool_args['code']
        budget_exhausted = False

        async def dispatch(agent_name: str, kwargs: dict[str, Any]) -> Any:
            nonlocal budget_exhausted
            if 'task' not in kwargs:
                raise TypeError(f'{agent_name}() missing required keyword argument: task')
            task = kwargs['task']
            # Budget check + increment must stay suspension-free: there must be no `await`
            # between them. asyncio only switches tasks at suspension points, so an await-free
            # check-then-increment is atomic across the concurrently-gathered dispatches, which
            # is what makes `max_agent_calls` an exact ceiling under fan-out. Insert an `await`
            # here (e.g. an async permission check) and the count can race past the limit; you
            # would then need an explicit reservation instead.
            if self._call_count >= self.max_agent_calls:
                budget_exhausted = True
                raise RuntimeError(f'sub-agent call budget ({self.max_agent_calls}) exhausted')
            self._call_count += 1
            try:
                result = await self._by_name[agent_name].agent.run(
                    task,
                    deps=ctx.deps,
                    usage=ctx.usage if self.forward_usage else None,
                    usage_limits=self.sub_agent_usage_limits,
                )
            except Exception as exc:
                # Don't leak host internals (file paths, deps/agent reprs) to the model;
                # surface the failing agent and error type only.
                raise RuntimeError(f'sub-agent {agent_name!r} raised {type(exc).__name__}') from exc
            return to_jsonable_python(result.output)

        limits = _resolve_resource_limits(self.resource_limits)
        capture = PrintCapture()
        in_workflow_token = _in_workflow.set(True)
        try:
            repl = MontyRepl(limits=limits)
            monty_state = repl.feed_start(code, print_callback=capture)
            # `_by_name` is not mutated during a run (reveals land in `get_tools`), so it is a
            # stable name registry for the whole script. Sub-agents always run concurrently (the
            # executor's defaults); durable ordering (global_sequential) lands with durability.
            completed = await MontyExecutor(dispatch=dispatch, valid_names=self._by_name).run(monty_state)
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in workflow:\n{capture.prepend_to(e.display())}') from e
        except MontyRuntimeError as e:
            if budget_exhausted:
                # Retrying can't help — the per-run budget is spent. Return a terminal result
                # so the model concludes instead of burning retries into a hard failure.
                return {
                    'error': (
                        f'This run exhausted its sub-agent call budget ({self.max_agent_calls}). '
                        'Conclude using the results already gathered; further sub-agent calls in '
                        'this run will be refused.'
                    )
                }
            raise ModelRetry(f'Runtime error in workflow:\n{capture.prepend_to(e.display())}') from e
        finally:
            _in_workflow.reset(in_workflow_token)

        return _workflow_result(completed.output, capture.joined)
