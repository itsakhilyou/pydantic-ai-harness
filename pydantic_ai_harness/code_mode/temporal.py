"""Temporal support for Code Mode: Monty runs in activities, the workflow orchestrates.

Temporal's workflow sandbox forbids the environment and filesystem access that
`Monty()` needs to locate and spawn its subprocess workers, and a workflow task
that blocks on subprocess I/O trips the SDK's deadlock detector. So under
Temporal the sandbox execution moves to the activity side, using Monty's
serializable suspended state:

- A `start` activity feeds the snippet into a fresh Monty session (restoring
  dumped REPL state, if any) and drives it until it needs tool results. At that
  point it serializes the suspended interpreter with `snapshot.dump()` and
  returns the pending calls plus the dump.
- The workflow dispatches each pending call through the regular toolset path,
  so every nested tool call remains its own recorded, retryable activity.
- A `resume` activity restores the dump with `load_snapshot()` in a fresh
  session (typically on a different worker process), delivers the results, and
  drives on to the next suspension or completion.

The workflow itself never touches Monty: it only awaits activities, and on
history replay every sandbox segment is served from recorded activity results
rather than re-executed.

Usage:

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalAgent
from pydantic_ai_harness.code_mode.temporal import CodeModePlugin, TemporalCodeMode

code_mode = TemporalCodeMode()
agent = Agent('openai:gpt-5', name='my_agent', capabilities=[code_mode])
temporal_agent = TemporalAgent(agent)

worker = Worker(
    client,
    task_queue='my-queue',
    workflows=[MyWorkflow],
    plugins=[AgentPlugin(temporal_agent), CodeModePlugin(code_mode)],
)
```

Caveats compared to running Code Mode locally:

- Interpreter dumps and tool results travel through activity payloads, which
  Temporal caps at 2MB by default. A REPL holding very large values in
  variables can exceed that.
- `mount` entries with `mode='overlay'` lose writes at each suspension: the
  restored overlay starts empty, so overlay writes do not survive across a
  tool call in the same snippet. Read-only and read-write mounts must point at
  paths available on every activity worker.
- `os_access` handlers run inside activities; like any activity side effect,
  they may run again if an activity is retried.
- Exceptions raised by tools are rebuilt from their type name and message
  before re-entering the sandbox, so `except SomeCustomError` in sandbox code
  only matches built-in exception types.
- A Monty worker crash (or `request_timeout` kill) resets the REPL session and
  surfaces as a `ModelRetry`, where the local path lets `MontyCrashedError`
  propagate and fail the run. Retrying with a fresh session is the useful
  behavior in a durable workflow; the local path may adopt it later.
- The sandbox activities do not heartbeat: a worker crash mid-hop is detected
  when `start_to_close_timeout` expires, so detection latency is bounded by
  the activity timeout.
"""

from __future__ import annotations

import asyncio
import binascii
import builtins
from base64 import b64decode, b64encode
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal, NoReturn

try:
    from temporalio import activity, workflow
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ApplicationError
    from temporalio.plugin import SimplePlugin
    from temporalio.workflow import ActivityConfig
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'temporalio is required for TemporalCodeMode. Install it with: '
        'pip install "pydantic-ai-harness[temporal,code-mode]"'
    ) from _import_error

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_core import to_json
from pydantic_monty import (
    AsyncFunctionSnapshot,
    AsyncMonty,
    AsyncMontySession,
    AsyncNameLookupSnapshot,
    AsyncSnapshot,
    CollectString,
    ExternalResult,
    ExternalSettledResult,
    MontyComplete,
    MontyCrashedError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
)
from typing_extensions import Self

from pydantic_ai_harness._monty_exec import DispatchFn, MontyOS, PrintCapture, is_sandbox_panic
from pydantic_ai_harness.code_mode._capability import CodeMode
from pydantic_ai_harness.code_mode._toolset import CodeModeMount, CodeModeToolset

__all__ = [
    'AdvanceResult',
    'CodeModePlugin',
    'ResumeParams',
    'SandboxCall',
    'SettledCall',
    'StartParams',
    'TemporalCodeMode',
    'TemporalCodeModeToolset',
]


@dataclass
class StartParams:
    """Input to the `start` activity: one `run_code` snippet and its session context.

    `repl_state` is a base64-encoded idle session dump from a previous
    `run_code` call, or `None` for a fresh REPL. Monty dumps are binary, and
    Temporal's pydantic payload converter JSON-serializes `bytes` as UTF-8, so
    dumps travel base64-encoded and are only decoded inside the activities.
    """

    code: str
    repl_state: str | None
    type_check: bool
    type_check_stubs: str | None
    valid_names: list[str]
    sequential_names: list[str]


@dataclass
class SandboxCall:
    """A tool call the suspended sandbox is waiting on, as reported by the activity."""

    call_id: int
    name: str
    kwargs: dict[str, Any]


@dataclass
class SettledCall:
    """The workflow-side outcome of one `SandboxCall`.

    `exception_type` discriminates: `None` means `value` is the tool's return
    value (which may itself be `None`); otherwise the call failed and the
    exception is carried by type name and message.
    """

    call_id: int
    value: Any = None
    exception_type: str | None = None
    exception_message: str = ''


@dataclass
class ResumeParams:
    """Input to the `resume` activity: the suspended interpreter and settled results.

    `results` carries every not-yet-consumed result settled so far for this
    snippet, not just the latest batch: results a previous hop did not deliver
    (because the sandbox had not awaited them yet) must be available to later
    hops, and the activity has no state of its own between hops. The workflow
    prunes results a hop reports as consumed (`AdvanceResult.consumed_call_ids`).
    """

    snapshot: str
    results: list[SettledCall]
    valid_names: list[str]
    sequential_names: list[str]


@dataclass
class AdvanceResult:
    """Outcome of one activity hop: the sandbox completed, suspended, or failed.

    - `complete`: `output` holds the snippet's value and `repl_state` the idle
      session dump (base64) for the next `run_code` call.
    - `pending`: `snapshot` holds the suspended interpreter dump (base64) and
      `calls` the tool calls to settle, in dispatch order.
    - `error`: `error_kind` says how the snippet failed. `syntax`/`typing`/
      `runtime` are code errors the model can revise; `crash` and
      `crash-timeout` mean the worker died mid-execution and the REPL state is
      lost.
    """

    status: Literal['complete', 'pending', 'error']
    printed: str = ''
    output: Any = None
    repl_state: str | None = None
    snapshot: str | None = None
    calls: list[SandboxCall] = field(default_factory=list[SandboxCall])
    consumed_call_ids: list[int] = field(default_factory=list[int])
    error_kind: Literal['syntax', 'typing', 'runtime', 'crash', 'crash-timeout'] | None = None
    error_display: str = ''


_DEFAULT_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=120),
    retry_policy=RetryPolicy(maximum_attempts=3),
)


# Sentinel key marking a base64-wrapped `bytes` value on the wire. Temporal's
# pydantic payload converter serializes untyped values with `pydantic_core.to_json`,
# which encodes `bytes` as UTF-8 and fails on binary data (e.g. the raw bytes inside
# a serialized `BinaryContent` tool result), so every value that crosses an activity
# boundary in an `Any`-typed field is wrapped/unwrapped with `_wire_encode`/`_wire_decode`.
_WIRE_BYTES_KEY = '__pydantic_ai_harness_bytes_b64__'


def _wire_encode(value: Any) -> Any:
    """Recursively wrap `bytes` values so they survive JSON payload serialization."""
    if isinstance(value, bytes):
        return {_WIRE_BYTES_KEY: b64encode(value).decode('ascii')}
    if isinstance(value, dict):
        return {key: _wire_encode(item) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, (list, tuple)):
        return [_wire_encode(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _wire_decode(value: Any) -> Any:
    """Inverse of `_wire_encode`: restore wrapped `bytes` values.

    The sentinel key is reserved: a dict whose only entry is the sentinel key
    with a valid-base64 string value decodes to `bytes` even if it was never
    produced by `_wire_encode`. A sentinel-shaped dict whose value is not valid
    base64 passes through unchanged rather than raising -- this runs in
    workflow code, where an exception would fail the workflow task.
    """
    if isinstance(value, dict):
        encoded = value.get(_WIRE_BYTES_KEY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if len(value) == 1 and isinstance(encoded, str):  # pyright: ignore[reportUnknownArgumentType]
            try:
                return b64decode(encoded, validate=True)
            except binascii.Error:
                return dict(value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        return {key: _wire_decode(item) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, (list, tuple)):
        return [_wire_decode(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _rebuild_exception(type_name: str, message: str) -> BaseException:
    """Reconstruct a tool exception from its serialized type name and message.

    Built-in exception types are instantiated directly so sandbox `except`
    clauses still match them. Anything else (including exceptions whose
    constructor needs more than a message, like `UnicodeDecodeError`) becomes a
    dynamically named `Exception` subclass, which preserves the type name the
    model sees in tracebacks without claiming catchability.
    """
    builtin = getattr(builtins, type_name, None)
    if isinstance(builtin, type) and issubclass(builtin, Exception):
        try:
            return builtin(message)
        except TypeError:
            pass
    return type(type_name, (Exception,), {})(message)


def _to_external(settled: SettledCall) -> ExternalSettledResult:
    """Convert a workflow-settled call into the payload Monty's `resume` expects."""
    if settled.exception_type is None:
        return {'return_value': _wire_decode(settled.value)}
    return {'exception': _rebuild_exception(settled.exception_type, settled.exception_message)}


async def _drive(
    snapshot: AsyncSnapshot,
    session: AsyncMontySession,
    collector: CollectString,
    *,
    valid_names: set[str],
    sequential_names: set[str],
    settled: dict[int, ExternalSettledResult],
    os_access: MontyOS | None,
) -> AdvanceResult:
    """Drive a Monty session until it completes or needs tool results from the workflow.

    Mirrors `MontyExecutor`'s handling of name lookups, unknown functions, and
    positional arguments, but where the executor dispatches tool calls, this
    defers them and suspends: external calls are answered from `settled` when
    the workflow already provided a result, and otherwise collected until a
    snapshot actually blocks on one, at which point the suspended interpreter
    is dumped and every unsettled call is handed back to the workflow.
    """
    # Unsettled calls captured this hop, in deferral order. All of them are
    # returned at the next suspension (not just the ones the blocking snapshot
    # waits on): a dump does not re-announce earlier FunctionSnapshots, so a
    # call whose info was not handed to the workflow now would be undeliverable
    # in a later hop. This makes dispatch eager for deferred-but-not-yet-awaited
    # calls, matching the local executor's parallel mode.
    deferred: dict[int, SandboxCall] = {}
    # Call ids whose results were delivered into the sandbox during this hop.
    # Reported back so the workflow can prune its settled-results list instead
    # of resending consumed results on every subsequent hop.
    consumed: list[int] = []
    while not isinstance(snapshot, MontyComplete):
        if isinstance(snapshot, AsyncNameLookupSnapshot):
            # Leave the name undefined so the sandbox raises NameError, as in
            # the local executor.
            snapshot = await snapshot.resume(os=os_access)
        elif isinstance(snapshot, AsyncFunctionSnapshot):
            call_id = snapshot.call_id
            if call_id in settled:
                # A restored suspension for a sequential (`def`) tool: the
                # workflow already settled it, deliver the result inline.
                consumed.append(call_id)
                snapshot = await snapshot.resume(settled.pop(call_id), os=os_access)
                continue
            name = snapshot.function_name
            if name not in valid_names:
                # Unknown functions and OS calls without an `os_access` handler
                # both surface here; match the local executor's NameError.
                snapshot = await snapshot.resume({'exception': NameError(f'Unknown function: {name}')}, os=os_access)
            elif snapshot.args:
                snapshot = await snapshot.resume(
                    {'exception': TypeError(f'{name}() does not accept positional arguments; use keyword arguments')},
                    os=os_access,
                )
            elif name in sequential_names:
                # Rendered as `def` (sync): the sandbox needs the result inline,
                # so suspend here. Earlier-deferred calls go first in the list,
                # preserving the local executor's barrier ordering.
                calls = [
                    *deferred.values(),
                    SandboxCall(call_id=call_id, name=name, kwargs=_wire_encode(dict(snapshot.kwargs))),
                ]
                return AdvanceResult(
                    status='pending',
                    printed=collector.output,
                    snapshot=b64encode(snapshot.dump()).decode('ascii'),
                    calls=calls,
                    consumed_call_ids=consumed,
                )
            else:
                deferred[call_id] = SandboxCall(call_id=call_id, name=name, kwargs=_wire_encode(dict(snapshot.kwargs)))
                defer: ExternalResult = {'future': ...}
                snapshot = await snapshot.resume(defer, os=os_access)
        else:
            # `AsyncFutureSnapshot` -- every sandbox task is blocked on external futures.
            deliverable = {cid: settled.pop(cid) for cid in snapshot.pending_call_ids if cid in settled}
            if deliverable:
                consumed.extend(deliverable)
                snapshot = await snapshot.resume(results=deliverable, os=os_access)
            else:
                # Every awaited future is unsettled; hand the interpreter and
                # the outstanding calls back to the workflow.
                return AdvanceResult(
                    status='pending',
                    printed=collector.output,
                    snapshot=b64encode(snapshot.dump()).decode('ascii'),
                    calls=list(deferred.values()),
                    consumed_call_ids=consumed,
                )
    return AdvanceResult(
        status='complete',
        printed=collector.output,
        output=_wire_encode(snapshot.output),
        repl_state=b64encode(await session.dump()).decode('ascii'),
        consumed_call_ids=consumed,
    )


def _error_result(
    kind: Literal['syntax', 'typing', 'runtime'], display: str, collector: CollectString
) -> AdvanceResult:
    return AdvanceResult(status='error', printed=collector.output, error_kind=kind, error_display=display)


def _payload_safe(result: AdvanceResult, collector: CollectString) -> AdvanceResult:
    """Backstop against values the activity payload converter cannot serialize.

    `_wire_encode` handles `bytes` values, but the sandbox can still produce
    values `pydantic_core.to_json` rejects (e.g. a dict with a non-UTF-8 `bytes`
    key). Returning such a result would fail the activity's own result
    conversion, burning its retry attempts on deterministic re-execution;
    dry-run the serialization here and report a code error the model can revise
    instead.
    """
    try:
        to_json(result)
    except Exception as e:
        return _error_result(
            'runtime',
            f'The code produced a value that cannot be returned from the sandbox: {e}. '
            'Return JSON-compatible data (strings, numbers, lists, dicts with string keys).',
            collector,
        )
    return result


async def _run_hop(
    *,
    os_access: MontyOS | None,
    mount: CodeModeMount | None,
    request_timeout: float | None,
    start: StartParams | None = None,
    resume: ResumeParams | None = None,
) -> AdvanceResult:
    """Run one activity hop: feed a new snippet (`start`) or restore a dump (`resume`).

    A fresh worker pool is spawned per hop. Monty workers start in a few
    milliseconds, which is negligible next to the activity round trip itself,
    and per-hop pools keep the activity worker free of process-wide state.
    """
    collector = CollectString()
    async with AsyncMonty(request_timeout=request_timeout) as pool:
        params = start if start is not None else resume
        assert params is not None, 'either start or resume params are required'
        type_check = start.type_check if start is not None else False
        type_check_stubs = start.type_check_stubs if start is not None else None
        async with pool.checkout(type_check=type_check, type_check_stubs=type_check_stubs) as session:
            try:
                settled: dict[int, ExternalSettledResult] = {}
                if start is not None:
                    if start.repl_state is not None:
                        await session.load(b64decode(start.repl_state))
                    snapshot = await session.feed_start(
                        start.code,
                        print_callback=collector,
                        os=os_access,
                        mount=mount,
                        skip_type_check=not start.type_check,
                    )
                else:
                    assert resume is not None
                    settled = {s.call_id: _to_external(s) for s in resume.results}
                    snapshot = await session.load_snapshot(
                        b64decode(resume.snapshot), mount=mount, print_callback=collector, os=os_access
                    )
                result = await _drive(
                    snapshot,
                    session,
                    collector,
                    valid_names=set(params.valid_names),
                    sequential_names=set(params.sequential_names),
                    settled=settled,
                    os_access=os_access,
                )
                return _payload_safe(result, collector)
            except MontySyntaxError as e:
                return _error_result('syntax', e.display(), collector)
            except MontyTypingError as e:
                return _error_result('typing', e.display(), collector)
            except MontyRuntimeError as e:
                return _error_result('runtime', e.display(), collector)
            except MontyCrashedError as e:
                kind: Literal['crash', 'crash-timeout'] = 'crash-timeout' if e.timed_out else 'crash'
                return AdvanceResult(status='error', printed=collector.output, error_kind=kind, error_display=str(e))
            except BaseException as e:
                # A Rust-side panic surfacing host-side; anything else
                # (CancelledError, ...) re-raises unchanged.
                if not is_sandbox_panic(e):
                    raise
                return AdvanceResult(status='error', printed=collector.output, error_kind='crash', error_display=str(e))


# Prefixes for code errors the model can revise in place; crash kinds are
# handled separately because they also reset the REPL session.
_CODE_ERROR_PREFIXES: dict[str, str] = {
    'syntax': 'Syntax error in code:\n',
    'typing': 'Type error in code:\n',
    'runtime': 'Runtime error:\n',
}
_CRASH_MESSAGES: dict[str, str] = {
    'crash': 'The code aborted inside the sandbox and the session was reset. Revise the code and try again.',
    'crash-timeout': (
        'The code exceeded the sandbox execution time limit and the session was reset. '
        'Break the work into smaller run_code calls and try again.'
    ),
}


def _exception_wire_form(exc: Exception) -> tuple[str, str]:
    """Extract the `(type name, message)` the sandbox should see for a failed dispatch.

    Under Temporal, a tool exception that pydantic-ai does not serialize across
    the activity boundary (anything but `ModelRetry`/`ApprovalRequired`/
    `CallDeferred`) reaches the workflow as an `ActivityError` whose message is
    the fixed string "Activity task failed"; the original exception's class
    name and message live on the `ApplicationError` in its cause chain. Unwrap
    it so sandbox `except` clauses and model-visible errors keep working like
    they do on the local path.
    """
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, ApplicationError) and cause.type:
            return cause.type, cause.message
        cause = cause.__cause__
    return type(exc).__name__, str(exc)


async def _settle_call(dispatch: DispatchFn, call: SandboxCall) -> SettledCall:
    """Dispatch one pending sandbox call and serialize its outcome for the resume activity."""
    try:
        value = await dispatch(call.name, _wire_decode(call.kwargs))
    except Exception as exc:
        exception_type, exception_message = _exception_wire_form(exc)
        return SettledCall(call_id=call.call_id, exception_type=exception_type, exception_message=exception_message)
    return SettledCall(call_id=call.call_id, value=_wire_encode(value))


@dataclass(kw_only=True)
class TemporalCodeModeToolset(CodeModeToolset[AgentDepsT]):
    """Code mode toolset that runs the Monty sandbox in Temporal activities.

    Inside a workflow, `_run_sandboxed` ping-pongs the suspended interpreter
    through the `start`/`resume` activities while nested tool calls dispatch
    through the regular (temporalized) toolset path. Outside a workflow it
    behaves exactly like `CodeModeToolset`.
    """

    start_activity: Callable[[StartParams], Coroutine[Any, Any, AdvanceResult]]
    resume_activity: Callable[[ResumeParams], Coroutine[Any, Any, AdvanceResult]]
    activity_config: ActivityConfig | None = None
    max_hops: int = 50

    async def __aenter__(self) -> Self:
        """Enter the wrapped toolset; inside a workflow, no Monty pool is created."""
        if not workflow.in_workflow():
            return await super().__aenter__()
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Exit symmetrically with `__aenter__`: only tear down a pool if one was created."""
        if self._monty_pool is None:
            return await self.wrapped.__aexit__(*args)
        return await super().__aexit__(*args)

    async def _run_sandboxed(
        self,
        code: str,
        *,
        dispatch: DispatchFn,
        callable_defs: dict[str, ToolDefinition],
        sequential_names: set[str],
        global_sequential: bool,
        type_check: bool,
    ) -> tuple[Any, str]:
        if not workflow.in_workflow():
            return await super()._run_sandboxed(
                code,
                dispatch=dispatch,
                callable_defs=callable_defs,
                sequential_names=sequential_names,
                global_sequential=global_sequential,
                type_check=type_check,
            )

        valid_names = sorted(callable_defs)
        seq_names = sorted(sequential_names)
        capture = PrintCapture()
        # Settled-but-not-yet-consumed results, resent on each resume hop; see
        # `ResumeParams.results`.
        settled: list[SettledCall] = []

        result = await self._execute_hop(
            self.start_activity,
            StartParams(
                code=code,
                repl_state=b64encode(self._repl_state).decode('ascii') if self._repl_state is not None else None,
                type_check=type_check,
                type_check_stubs=self._build_type_check_stubs(callable_defs) if type_check else None,
                valid_names=valid_names,
                sequential_names=seq_names,
            ),
            summary='run_code sandbox: start',
        )
        hops = 0
        while True:
            capture('stdout', result.printed)
            consumed = set(result.consumed_call_ids)
            settled = [s for s in settled if s.call_id not in consumed]
            if result.status == 'complete':
                assert result.repl_state is not None, 'complete result must carry the session dump'
                self._repl_state = b64decode(result.repl_state)
                return _wire_decode(result.output), capture.joined
            if result.status == 'error':
                self._raise_code_error(result, capture)
            hops += 1
            if hops > self.max_hops:
                # Each round is at least one recorded activity, so an unbounded
                # tool-call loop in model code would grow the workflow history
                # without limit; abandon the snippet before dispatching another
                # round and let the model try a smaller one. REPL state stays at
                # its pre-snippet value, as on any other code error.
                raise ModelRetry(
                    f'The code needed more than {self.max_hops} rounds of tool calls in one run_code '
                    'call and was abandoned. Batch tool calls (e.g. with asyncio.gather) or split '
                    'the work into smaller run_code calls.'
                )
            # Temporal does not force a tool execution mode (the run-scoped
            # context var stays at its 'parallel' default), so match the local
            # executor: dispatch concurrently -- nested activities run in
            # parallel, and Temporal's deterministic task scheduling keeps
            # gather replay-safe -- unless the run opted into sequential mode.
            if global_sequential:
                for call in result.calls:
                    settled.append(await _settle_call(dispatch, call))
            else:
                settled.extend(await asyncio.gather(*(_settle_call(dispatch, call) for call in result.calls)))
            assert result.snapshot is not None, 'pending result must carry a snapshot'
            result = await self._execute_hop(
                self.resume_activity,
                ResumeParams(
                    snapshot=result.snapshot,
                    results=list(settled),
                    valid_names=valid_names,
                    sequential_names=seq_names,
                ),
                summary='run_code sandbox: resume',
            )

    async def _execute_hop(
        self,
        hop_activity: Callable[..., Coroutine[Any, Any, AdvanceResult]],
        params: StartParams | ResumeParams,
        *,
        summary: str,
    ) -> AdvanceResult:
        # User config merges over the defaults, so a partial override (e.g. just
        # `retry_policy`) keeps the default start-to-close timeout.
        config: ActivityConfig = {'summary': summary, **_DEFAULT_ACTIVITY_CONFIG, **(self.activity_config or {})}
        return await workflow.execute_activity(activity=hop_activity, args=[params], **config)

    def _raise_code_error(self, result: AdvanceResult, capture: PrintCapture) -> NoReturn:
        """Map an activity-reported sandbox failure to the same `ModelRetry` the local path raises."""
        assert result.error_kind is not None, 'error result must carry an error kind'
        prefix = _CODE_ERROR_PREFIXES.get(result.error_kind)
        if prefix is not None:
            raise ModelRetry(f'{prefix}{capture.prepend_to(result.error_display)}')
        # The worker died mid-execution (crash or watchdog timeout), so the
        # REPL's accumulated state is gone; reset like the local panic path so
        # the retry starts from a fresh, type-checked session.
        self._repl_state = None
        raise ModelRetry(_CRASH_MESSAGES[result.error_kind])


@dataclass
class TemporalCodeMode(CodeMode[AgentDepsT]):
    """Code mode capability for agents wrapped in `TemporalAgent`.

    Drop-in replacement for `CodeMode` that executes the Monty sandbox inside
    Temporal activities instead of the workflow sandbox (which cannot spawn
    Monty's subprocess workers). Nested tool calls still run as their own
    activities, so per-tool durability and history are unchanged; the sandbox
    compute segments between them become recorded activities as well, which
    means replay never re-executes model-written code.

    The sandbox activities must be registered on the worker alongside the
    agent's own activities, e.g. via `CodeModePlugin`:

    ```python
    code_mode = TemporalCodeMode()
    agent = Agent('openai:gpt-5', name='my_agent', capabilities=[code_mode])
    temporal_agent = TemporalAgent(agent)
    worker = Worker(..., plugins=[AgentPlugin(temporal_agent), CodeModePlugin(code_mode)])
    ```

    Outside a workflow (e.g. `TemporalAgent.run()` called directly), it behaves
    exactly like `CodeMode`.
    """

    activity_name: str = field(default='code_mode', kw_only=True)
    """Distinguishes this instance's activity names (`code_mode__<name>__start`/`__resume`).

    One instance can safely serve several agents on the same worker. Register
    two instances on one worker only if they have different `activity_name`s.
    """

    request_timeout: float | None = field(default=60.0, kw_only=True)
    """Per-hop wall-clock limit (seconds) enforced by the Monty pool's watchdog.

    A sandbox compute segment that exceeds it is killed and surfaces to the
    model as a retryable "code took too long" error instead of an activity
    timeout loop, which matters because model-written code can loop forever.
    It bounds the compute between tool calls, not the number of tool calls --
    `max_hops` bounds those. Must be lower than the activity
    `start_to_close_timeout`. `None` disables the watchdog and leaves runaway
    code to the activity timeout and its retry policy.
    """

    activity_config: ActivityConfig | None = field(default=None, kw_only=True)
    """Temporal activity options for the sandbox `start`/`resume` activities.

    Merged over the defaults (120s start-to-close timeout, comfortably above
    `request_timeout`, and 3 attempts), so a partial config overrides only the
    keys it sets. This is separate from the `TemporalAgent` activity config,
    which covers model requests and tool calls.
    """

    max_hops: int = field(default=50, kw_only=True)
    """Maximum rounds of tool calls per `run_code` call before the snippet is abandoned.

    Every round is at least one recorded activity, so a tool-call loop in
    model-written code (`request_timeout` only bounds compute between calls)
    would otherwise grow the workflow history without limit. Exceeding the cap
    raises a `ModelRetry` telling the model to batch or split the work.
    """

    _start_activity: Callable[[StartParams], Coroutine[Any, Any, AdvanceResult]] = field(init=False, repr=False)
    _resume_activity: Callable[[ResumeParams], Coroutine[Any, Any, AdvanceResult]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the activity definitions bound to this instance's sandbox configuration."""

        async def code_mode_start(params: StartParams) -> AdvanceResult:
            return await _run_hop(
                os_access=self.os_access, mount=self.mount, request_timeout=self.request_timeout, start=params
            )

        async def code_mode_resume(params: ResumeParams) -> AdvanceResult:
            return await _run_hop(
                os_access=self.os_access, mount=self.mount, request_timeout=self.request_timeout, resume=params
            )

        self._start_activity = activity.defn(name=f'code_mode__{self.activity_name}__start')(code_mode_start)
        self._resume_activity = activity.defn(name=f'code_mode__{self.activity_name}__resume')(code_mode_resume)

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        """The activity definitions to register on the worker (see `CodeModePlugin`)."""
        return [self._start_activity, self._resume_activity]

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset with the Temporal-aware code mode toolset."""
        return TemporalCodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            dynamic_catalog=self.dynamic_catalog,
            os_access=self.os_access,
            mount=self.mount,
            start_activity=self._start_activity,
            resume_activity=self._resume_activity,
            activity_config=self.activity_config,
            max_hops=self.max_hops,
        )


class CodeModePlugin(SimplePlugin):
    """Temporal worker plugin that registers a `TemporalCodeMode`'s sandbox activities.

    Add it to the worker's plugins next to the agent's `AgentPlugin`.
    """

    def __init__(self, code_mode: TemporalCodeMode[Any]):
        """Create a worker plugin exposing `code_mode`'s start/resume activities."""
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            name='CodeModePlugin',
            activities=code_mode.temporal_activities,
        )
