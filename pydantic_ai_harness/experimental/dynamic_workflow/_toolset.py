"""Toolset for the `DynamicWorkflow` capability.

Exposes a single `run_workflow` tool: the model writes a Python orchestration
script (run in a Monty sandbox) that calls named sub-agents as async functions
and composes their results -- fan-out, chaining, voting, loops -- in one step.
"""

from __future__ import annotations

import contextvars
import copy
import json
import keyword
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Generic, Literal

from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_ai.usage import UsageLimits
from pydantic_core import to_jsonable_python
from typing_extensions import Self, TypedDict

try:
    from pydantic_monty import MontyRepl, MontyRuntimeError, MontySyntaxError, ResourceLimits
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for DynamicWorkflow. '
        'Install it with: uv add "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

from pydantic_ai_harness._monty_exec import MontyExecutor, PrintCapture, is_sandbox_panic

# Set while a workflow script is executing, so a sub-agent that itself tries to run a workflow can
# be refused -- workflows do not nest. asyncio copies the context into each task `asyncio.gather`
# schedules, so concurrently-dispatched sub-agents inherit this flag (the capability is asyncio-only).
_in_workflow: contextvars.ContextVar[bool] = contextvars.ContextVar('pydantic_ai_harness_in_workflow', default=False)


class WorkflowResourceLimits(TypedDict, total=False):
    """Caps on the orchestration script's own sandbox resources (not sub-agent latency).

    A harness-owned view of the sandbox limits the capability supports, so the public API does
    not depend on the underlying sandbox's own types. Every field is optional; an omitted field
    keeps its backstop value.
    """

    max_duration_secs: float
    """Maximum total wall-clock seconds for the script -- **including** time spent awaiting
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
    """Backstop sandbox limits; no duration cap -- see `WorkflowResourceLimits.max_duration_secs`."""
    return {
        'max_memory': 256 * 1024 * 1024,
        'max_allocations': 50_000_000,
    }


# The keys `WorkflowResourceLimits` accepts. A `total=False` TypedDict does not validate keys at
# runtime, so a typo (e.g. `max_durations_secs`) would otherwise merge through and be silently
# dropped -- quietly disabling the only guard against a pure-CPU `while True`. We reject unknowns.
_RESOURCE_LIMIT_KEYS = frozenset(WorkflowResourceLimits.__annotations__)
_MODEL_SAFE_EXCEPTION_MESSAGE_TYPES = (UsageLimitExceeded,)
_MAX_COMPLETED_DISPATCHES = 20
_MAX_TASK_PREVIEW_CHARS = 120
_MAX_RESULT_PREVIEW_CHARS = 300
_TRUNCATED_MARKER = ' ... [truncated]'


def _resolve_resource_limits(limits: WorkflowResourceLimits | Literal['unlimited'] | None) -> ResourceLimits:
    """Resolve the public `resource_limits` value to the limits handed to the sandbox.

    A partial mapping merges *onto* the backstop rather than replacing it, so `{'max_memory': ...}`
    never silently drops the allocations backstop. Full semantics: `DynamicWorkflow.resource_limits`.
    """
    if limits is None:
        return _default_resource_limits()
    if limits == 'unlimited':
        return {}
    unknown = set(limits) - _RESOURCE_LIMIT_KEYS
    if unknown:
        raise UserError(
            f'Unknown `resource_limits` key(s): {sorted(unknown)}. Valid keys are {sorted(_RESOURCE_LIMIT_KEYS)}.'
        )
    return {**_default_resource_limits(), **limits}


class _WorkflowArguments(TypedDict):
    code: Annotated[str, Field(description='The Python orchestration script to execute in the sandbox.')]


_WORKFLOW_ARGS_ADAPTER = TypeAdapter(_WorkflowArguments)
_WORKFLOW_ARGS_JSON_SCHEMA = _WORKFLOW_ARGS_ADAPTER.json_schema()
_WORKFLOW_ARGS_VALIDATOR: SchemaValidatorProt = _WORKFLOW_ARGS_ADAPTER.validator  # pyright: ignore[reportAssignmentType]

_WORKFLOW_BASE_DESCRIPTION = """\
Write and run a Python orchestration script in a sandbox to coordinate multiple sub-agents.

Use this to break a task across specialized sub-agents and combine their results in a single step --
fan work out in parallel, chain one agent's output into the next, vote across several, or loop until
done -- instead of delegating to one sub-agent at a time.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes** and **no third-party libraries**.
- **Useful standard-library modules**: `asyncio`, `math`, `json`, `re`, `typing`. Import what you use
  at the top of the script. Other modules are unavailable or stubbed -- don't rely on them.
- **No wall-clock or timing primitives** (`asyncio.sleep`, `datetime.now()`, the `time` module).

Each sub-agent below is an async function. Call it with the `task` keyword argument -- write
`reviewer(task="...")`, not `reviewer("...")`; all parameters are keyword-only. A sub-agent returns
that agent's output: a string by default, or -- if it has a structured `output_type` -- a dict, whose
fields you read by subscript (`r["field"]`), not attribute (`r.field`). Each sub-agent call is an
independent run with no memory of earlier calls; include all needed context in `task`. Run several
at once with `asyncio.gather` rather than awaiting each sequentially:

```python
import asyncio
reviews = await asyncio.gather(reviewer(task="check auth"), reviewer(task="check parsing"))
```

`asyncio.gather` does **not** support `return_exceptions=True`, and a sub-agent that raises cannot be
caught inside the script: one failure aborts the whole script and you retry it. Design the script so
sub-agents don't depend on catching each other's errors.

The last expression's value is captured as the result -- you do **not** need to `print()` it, and
printing produces a string representation, not structured data. Use `print()` only for debug logging.
Return shapes: no print returns the last expression value (or `{}` if it is `None`); print plus a
non-`None` value returns `{"output": "<printed text>", "result": <last expression>}`; print plus
`None` returns `{"output": "<printed text>"}`. If a script fails after some sub-agent calls complete,
those completed results are reported back so a retry can reuse them.\
"""


def _is_valid_sandbox_name(name: str) -> bool:
    """Whether `name` can be exposed as a sandbox function: a non-keyword Python identifier.

    `str.isidentifier()` alone is not enough -- Python keywords (`for`, `class`, `async`, ...) are
    valid identifiers but cannot be used as function names, so the model could never call them.
    Callers guard the empty/`None` case before this is reached.
    """
    return name.isidentifier() and not keyword.iskeyword(name)


# Every sub-agent is exposed with the same fixed parameters -- `(*, task: str)` -- and a per-agent
# return schema. Render each catalog entry through core's `FunctionSignature` (the renderer
# code_mode and Pydantic AI already use). This keeps the catalog format consistent across
# capabilities, forces keyword-only `task` to match `dispatch` (which reads `kwargs['task']`), and
# renders docstrings safely -- a hand-rolled f-string breaks on a newline or a quote inside a
# description.
_SUB_AGENT_PARAMS_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {'task': {'type': 'string'}},
    'required': ['task'],
}
_NO_CONFLICTING_TYPE_NAMES: frozenset[str] = frozenset()


def _agent_signature(name: str, agent: AbstractAgent[Any, Any]) -> FunctionSignature:
    """Build the sandbox function signature for one sub-agent."""
    try:
        return_schema = agent.output_json_schema()
    except Exception:
        return_schema = None
    return FunctionSignature.from_schema(
        name=name,
        parameters_schema=_SUB_AGENT_PARAMS_SCHEMA,
        return_schema=return_schema,
    )


def _render_agent_block(
    signature: FunctionSignature,
    description: str | None,
    *,
    conflicting_type_names: frozenset[str] = _NO_CONFLICTING_TYPE_NAMES,
) -> str:
    """Render one sub-agent as the async function signature shown to the model."""
    return signature.render(
        '...',
        description=description,
        is_async=True,
        conflicting_type_names=conflicting_type_names,
    )


def _render_catalog(catalog: Mapping[str, WorkflowAgent[AgentDepsT]], *, max_agent_calls: int) -> str:
    """Render the available sub-agents as async function signatures for the tool description."""
    signatures = {name: _agent_signature(name, entry.agent) for name, entry in catalog.items()}
    signature_list = list(signatures.values())
    conflicting = FunctionSignature.get_conflicting_type_names(signature_list)
    type_blocks = FunctionSignature.render_type_definitions(signature_list, conflicting)
    function_blocks = [
        _render_agent_block(signatures[name], entry.description, conflicting_type_names=conflicting)
        for name, entry in catalog.items()
    ]
    listing = '```python\n' + '\n\n'.join([*type_blocks, *function_blocks]) + '\n```'
    budget = f'This run can make at most {max_agent_calls} sub-agent calls in total; plan fan-out width accordingly.'
    return f'{_WORKFLOW_BASE_DESCRIPTION}\n\n{budget}\n\nAvailable sub-agents:\n\n{listing}'


def _render_reveal(
    name: str,
    catalog: Mapping[str, WorkflowAgent[AgentDepsT]],
    tool_name: str,
) -> str:
    """Announcement enqueued when a sub-agent is revealed mid-run.

    Delivered as a conversation message (not folded into the cached tool description), so the
    prompt-cache prefix stays stable while the model still learns the agent is now callable.
    """
    signatures = {agent_name: _agent_signature(agent_name, entry.agent) for agent_name, entry in catalog.items()}
    signature = signatures[name]
    conflicting = FunctionSignature.get_conflicting_type_names(list(signatures.values()))
    type_blocks = FunctionSignature.render_type_definitions([signature], conflicting)
    function_block = _render_agent_block(signature, catalog[name].description, conflicting_type_names=conflicting)
    block = '\n\n'.join([*type_blocks, function_block])
    return f'A new sub-agent is now available to call from inside the `{tool_name}` script:\n\n```python\n{block}\n```'


def _workflow_result(result: object, printed: str) -> object:
    """Shape the tool return: the script's result, its captured `print()` output, or both."""
    if not printed:
        return result if result is not None else {}
    if result is None:
        return {'output': printed}
    return {'output': printed, 'result': result}


@dataclass(frozen=True)
class _CompletedDispatch:
    """One sub-agent result completed by the failed workflow script."""

    agent_name: str
    task: str
    result: object


def _truncate_preview(value: str, max_chars: int) -> str:
    """Trim a preview with an explicit marker so the model can see it is incomplete."""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - len(_TRUNCATED_MARKER)] + _TRUNCATED_MARKER


def _json_preview(value: object, max_chars: int) -> str:
    """Render a compact JSON preview of a completed sub-agent result."""
    rendered = json.dumps(value, ensure_ascii=True, separators=(',', ':'))
    return _truncate_preview(rendered, max_chars)


def _completed_dispatch_lines(completed: list[_CompletedDispatch]) -> list[str]:
    """Format the most recent completed sub-agent results for model-facing salvage."""
    shown = completed[-_MAX_COMPLETED_DISPATCHES:]
    lines = [
        f'{entry.agent_name}(task={json.dumps(_truncate_preview(entry.task, _MAX_TASK_PREVIEW_CHARS), ensure_ascii=True)})'
        f' -> {_json_preview(entry.result, _MAX_RESULT_PREVIEW_CHARS)}'
        for entry in shown
    ]
    omitted = len(completed) - len(shown)
    if omitted:
        lines.insert(0, f'... {omitted} earlier completed result(s) omitted ...')
    return lines


def _completed_retry_section(completed: list[_CompletedDispatch]) -> str:
    """Build the optional retry-message section listing salvageable completed results."""
    lines = _completed_dispatch_lines(completed)
    if not lines:
        return ''
    listing = '\n'.join(f'- {line}' for line in lines)
    return (
        '\n\nCompleted sub-agent results from the failed script '
        '(reuse these values instead of re-calling them; their budget was already spent):\n'
        f'{listing}'
    )


def _budget_terminal_result(
    *,
    max_agent_calls: int,
    last_error: str,
    completed_dispatches: list[_CompletedDispatch],
) -> dict[str, object]:
    """Build the terminal result returned after the exact sub-agent-call budget is exhausted."""
    return {
        'error': (
            f'This run exhausted its sub-agent call budget ({max_agent_calls}). '
            'Conclude using the results already gathered; further sub-agent calls in '
            'this run will be refused.'
        ),
        'last_error': last_error,
        'completed': _completed_dispatch_lines(completed_dispatches),
    }


@dataclass(frozen=True, kw_only=True)
class WorkflowAgent(Generic[AgentDepsT]):
    """One sub-agent exposed to the orchestration script as an async function.

    Bundles the agent with its sandbox function name and catalog description so the
    three travel together -- there is no second collection to keep in sync.
    """

    agent: AbstractAgent[AgentDepsT, Any]
    """The sub-agent to run when the script calls this function."""

    name: str | None = None
    """Sandbox function name; must be a valid Python identifier and unique across the
    workflow. Falls back to the agent's `name`."""

    description: str | None = None
    """Description shown to the model in the sub-agent catalog, rendered as the sandbox
    function's docstring. When omitted, the model sees only the bare signature -- set this
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

    `DynamicWorkflow.reveal()` is the public entry point for adding one mid-run. This remains a
    mutable list because append is the underlying reveal mechanism and direct list append still
    works for existing callers; see `DynamicWorkflow.agents` for the reveal contract."""

    tool_name: str = 'run_workflow'
    """Name of the tool exposed to the model."""

    max_agent_calls: int = 50
    """Maximum total sub-agent runs per agent run (an exact, host-enforced ceiling)."""

    max_retries: int = 3
    """Maximum retries for the `run_workflow` tool (syntax/runtime errors count as retries)."""

    forward_usage: bool = True
    """Share the parent run's `usage` accumulator with sub-agents. See
    `DynamicWorkflow.forward_usage` for what is and is not forwarded."""

    sub_agent_usage_limits: UsageLimits | None = None
    """`UsageLimits` applied to every sub-agent run, replacing pydantic-ai's default.
    See `DynamicWorkflow.sub_agent_usage_limits` for the budgeting semantics."""

    resource_limits: WorkflowResourceLimits | Literal['unlimited'] | None = None
    """Sandbox limits guarding the orchestration script's own memory/allocations (not sub-agents).
    See `DynamicWorkflow.resource_limits` for the `None`/`'unlimited'`/partial-dict semantics."""

    toolset_id: str | None = None
    """Stable toolset id; defaults to the tool name."""

    # Per-run count of sub-agent calls; reset on `for_run`.
    _call_count: int = field(default=0, init=False, repr=False)

    # Sub-agents indexed by resolved sandbox name; seeded from `agents` in `__post_init__` and
    # extended in place as runtime appends to `agents` are revealed (`_reveal_pending`).
    _by_name: dict[str, WorkflowAgent[AgentDepsT]] = field(init=False, repr=False)

    # Tool description, frozen at run start. Rendered from the agents present when the run began
    # and never re-rendered after a reveal, so the description -- and thus the prompt-cache
    # prefix -- never changes mid-run.
    _description: str = field(init=False, repr=False)

    # `id()` of reveal entries already warned about (invalid or colliding name), so a persistent
    # bad append warns once rather than every step. Rebuilt per run by `for_run`'s shallow clone.
    _reveal_warned: set[int] = field(default_factory=set[int], init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.agents:
            raise UserError('DynamicWorkflow requires at least one sub-agent in `agents`.')
        if self.max_agent_calls < 1:
            raise UserError('DynamicWorkflow `max_agent_calls` must be at least 1.')
        _resolve_resource_limits(self.resource_limits)  # validate keys now, not at the first tool call
        self._rebuild(strict=True)

    def _rebuild(self, *, strict: bool) -> set[int]:
        """Rebuild the name index and the frozen tool description from the current `agents`.

        With `strict=True` an unusable entry (no name, invalid name, duplicate) raises `UserError`;
        with `strict=False` it is skipped with a warning and its `id()` is returned so the caller
        can suppress a second warning from `_reveal_pending`. The lenient mode exists for `for_run`:
        a mid-run append that `_reveal_pending` tolerated must not hard-fail every later run.
        """
        by_name: dict[str, WorkflowAgent[AgentDepsT]] = {}
        skipped: set[int] = set()
        for entry in self.agents:
            name = entry.resolved_name
            if not name:
                problem = (
                    'DynamicWorkflow sub-agent has no `name` and its agent has no `name`; '
                    'set `WorkflowAgent(name=...)` so it can be exposed as a sandbox function.'
                )
            elif not _is_valid_sandbox_name(name):
                problem = (
                    f'DynamicWorkflow sub-agent name {name!r} cannot be exposed as a sandbox function: '
                    'it must be a Python identifier that is not a reserved keyword. Rename it.'
                )
            elif name in by_name:
                problem = f'DynamicWorkflow has two sub-agents named {name!r}; names must be unique.'
            else:
                by_name[name] = entry
                continue
            if strict:
                raise UserError(problem)
            skipped.add(id(entry))
            warnings.warn(f'{problem} It is not callable in this run.', stacklevel=3)
        self._by_name = by_name
        self._description = _render_catalog(by_name, max_agent_calls=self.max_agent_calls)
        return skipped

    @property
    def id(self) -> str | None:
        return self.toolset_id or self.tool_name

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Self:
        """Fresh instance per run so the sub-agent-call budget is per-run.

        `dataclasses.replace` would re-run `__post_init__`, whose strict validation raises on an
        unusable entry that a mid-run append left in `agents` -- turning one bad reveal into a hard
        failure of every later run. Clone shallowly instead (keeping `agents` shared, so appends
        stay visible to this run) and rebuild the per-run state leniently.
        """
        clone = copy.copy(self)
        clone._call_count = 0
        # Seed the warned set with the entries skipped here so `_reveal_pending` doesn't warn again.
        clone._reveal_warned = clone._rebuild(strict=False)
        return clone

    def _reveal_pending(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reveal sub-agents appended to `agents` since the run started.

        Diffs the live `agents` list against the names already known (`_by_name`, which holds the
        baseline plus anything revealed so far) and folds each newcomer in -- so `dispatch` resolves
        it -- enqueuing an announcement for the model. The frozen `_description` is untouched, so
        the cached prompt prefix is unaffected. Re-seeing an already-known entry (a baseline agent,
        or one revealed on an earlier step) is a no-op, identified by object identity so it is never
        mistaken for a name collision.

        A newcomer whose name is invalid, or already taken by a *different* sub-agent, cannot be
        revealed: the original stays in place (reveal never silently swaps an agent out) and a
        warning is emitted once so the dropped reveal is not silent.

        Synchronous and await-free between snapshotting `agents` and mutating `_by_name`, so a
        concurrently-running `dispatch` never observes a half-revealed agent -- the same
        await-free-critical-section reasoning that keeps `max_agent_calls` exact under fan-out.
        """
        for entry in tuple(self.agents):
            name = entry.resolved_name
            existing = self._by_name.get(name) if name else None
            if existing is entry:
                continue
            if name and _is_valid_sandbox_name(name) and existing is None:
                self._by_name[name] = entry
                try:
                    ctx.enqueue(_render_reveal(name, self._by_name, self.tool_name))
                except UserError as exc:
                    warnings.warn(
                        f'DynamicWorkflow revealed sub-agent {name!r}, but could not enqueue its announcement: '
                        f'{exc}. It is callable in this run, but the model will not see the reveal message.',
                        stacklevel=2,
                    )
                continue
            if id(entry) not in self._reveal_warned:
                self._reveal_warned.add(id(entry))
                reason = (
                    'the name is already used by another sub-agent'
                    if existing is not None
                    else 'a sandbox function name must be a non-keyword Python identifier'
                )
                warnings.warn(
                    f'DynamicWorkflow could not reveal a sub-agent named {name!r}: {reason}. '
                    'It is not callable; any existing sub-agent of that name is unchanged.',
                    stacklevel=2,
                )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        self._reveal_pending(ctx)
        return {
            self.tool_name: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=self.tool_name,
                    description=self._description,
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
            return {
                'error': (
                    'Workflows do not nest: this sub-agent was invoked from a workflow and cannot start '
                    'its own. Return your result to the orchestrating workflow instead.'
                )
            }

        code = tool_args['code']
        budget_exhausted = False
        completed_dispatches: list[_CompletedDispatch] = []

        async def dispatch(agent_name: str, kwargs: dict[str, Any]) -> Any:
            nonlocal budget_exhausted
            # The sandbox signature is `(*, task: str)`, but Monty does not validate kwargs against
            # it, so check here: a dropped extra kwarg or a non-string `task` would otherwise run the
            # sub-agent on silently-wrong input. Each raises before the budget is touched.
            if 'task' not in kwargs:
                raise TypeError(f'{agent_name}() missing required keyword argument: task')
            extra = sorted(set(kwargs) - {'task'})
            if extra:
                raise TypeError(f'{agent_name}() got unexpected keyword argument(s): {", ".join(extra)}; only task')
            task = kwargs['task']
            if not isinstance(task, str):
                raise TypeError(f'{agent_name}() task must be a string, got {type(task).__name__}')
            # Budget check + increment must stay suspension-free: there must be no `await`
            # between them. asyncio only switches tasks at suspension points, so an await-free
            # check-then-increment is atomic across the concurrently-gathered dispatches, which
            # is what makes `max_agent_calls` an exact ceiling under fan-out. Insert an `await`
            # here (e.g. an async permission check) and the count can race past the limit; you
            # would then need an explicit reservation instead.
            #
            # This exists precisely because `usage_limits` cannot give an exact ceiling here:
            # core's own limit check is split from its increment by the model-request `await`
            # (a TOCTOU race -- N gathered sub-agents all pass the check before any increments;
            # measured ~20x overshoot), and `RunContext` exposes `usage` but not `usage_limits`,
            # so the parent's configured limit can't be forwarded to sub-agents at all.
            # TODO: file upstream on pydantic-ai -- (a) expose `usage_limits` on `RunContext`,
            # (b) atomic reserve-then-request in the run loop. Until then, tree-wide token caps
            # stay best-effort (see `forward_usage` / `sub_agent_usage_limits` docstrings).
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
                output = to_jsonable_python(result.output)
                completed_dispatches.append(_CompletedDispatch(agent_name=agent_name, task=task, result=output))
                return output
            except Exception as exc:
                # Don't leak host internals (file paths, deps/agent reprs) to the model;
                # surface the failing agent and error type by default.
                message = f'sub-agent {agent_name!r} raised {type(exc).__name__}'
                if isinstance(exc, _MODEL_SAFE_EXCEPTION_MESSAGE_TYPES):
                    message = f'{message}: {exc}'
                raise RuntimeError(message) from exc

        limits = _resolve_resource_limits(self.resource_limits)
        capture = PrintCapture()
        in_workflow_token = _in_workflow.set(True)
        try:
            repl = MontyRepl(limits=limits)
            monty_state = repl.feed_start(code, print_callback=capture)
            # `_by_name` is not mutated while a script executes (reveals land in `get_tools`,
            # which does not interleave with `call_tool`), so it is a stable name registry for
            # the whole script. Sub-agents always run concurrently (the executor's defaults);
            # durable ordering (global_sequential) lands with durability.
            completed = await MontyExecutor(dispatch=dispatch, valid_names=self._by_name).run(monty_state)
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in workflow:\n{capture.prepend_to(e.display())}') from e
        except MontyRuntimeError as e:
            if budget_exhausted:
                # On this capability's deferred-future path, host-raised exceptions cannot be
                # caught inside the sandbox (Monty's inline resume path can catch them). The
                # flag proves this script hit the budget; under gather, the displayed error may
                # be another independently surfaced failure from the same batch.
                return _budget_terminal_result(
                    max_agent_calls=self.max_agent_calls,
                    last_error=capture.prepend_to(e.display()),
                    completed_dispatches=completed_dispatches,
                )
            raise ModelRetry(
                f'Runtime error in workflow:\n{capture.prepend_to(e.display())}'
                f'{_completed_retry_section(completed_dispatches)}'
            ) from e
        except BaseException as e:
            # Convert a model-provokable sandbox panic to a retry (see `is_sandbox_panic`);
            # anything else (CancelledError, ...) re-raises unchanged.
            if not is_sandbox_panic(e):
                raise
            if budget_exhausted:
                return _budget_terminal_result(
                    max_agent_calls=self.max_agent_calls,
                    last_error='The workflow script aborted inside the sandbox after exhausting the sub-agent budget.',
                    completed_dispatches=completed_dispatches,
                )
            raise ModelRetry(
                'The workflow script aborted inside the sandbox. This can happen when the same '
                'sub-agent call is awaited more than once in one asyncio.gather -- give each gathered '
                'call its own invocation. Revise the script and try again.'
                f'{_completed_retry_section(completed_dispatches)}'
            ) from e
        finally:
            _in_workflow.reset(in_workflow_token)

        return _workflow_result(completed.output, capture.joined)
