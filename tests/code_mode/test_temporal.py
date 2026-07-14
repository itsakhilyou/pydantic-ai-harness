"""Temporal integration tests for CodeMode.

Under Temporal, `TemporalCodeMode` runs the Monty sandbox inside activities:
a `start` activity drives a snippet until it needs tool results and returns
the suspended interpreter as a serialized dump, the workflow dispatches the
pending tool calls through the regular (temporalized) toolset path, and a
`resume` activity restores the dump and continues. The workflow never spawns
Monty subprocess workers, so the workflow sandbox restrictions and deadlock
detector are never in play.

The `TestSandboxActivities` unit tests drive the activity functions directly
(they are plain async callables); the workflow tests start a local Temporal
dev server via `WorkflowEnvironment.start_local()` -- the Temporal SDK
downloads and runs `temporalite` automatically.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest

try:
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalAgent,
    )
    from temporalio import workflow
    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import (
        RestrictedWorkflowAccessError,
        SandboxedWorkflowRunner,
        SandboxRestrictions,
    )
    from temporalio.workflow import ActivityConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.code_mode.temporal import (
    CodeModePlugin,
    ResumeParams,
    SettledCall,
    StartParams,
    TemporalCodeMode,
)

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7244  # avoid conflict with other test suites
TASK_QUEUE = 'pydantic-ai-harness-code-mode-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


def _workflow_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            # Coverage imports parser modules lazily while tracing workflow code.
            'coverage',
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=[
            '--dynamic-config-value',
            'frontend.enableServerVersionCheck=false',
        ],
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin()],
    )


# ---------------------------------------------------------------------------
# Tools and agents (module-level -- Temporal requirement)
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def boom(x: int) -> int:
    """Always fails with a retryable error."""
    raise ModelRetry('boom hint: try something else')


def bad_value(x: int) -> int:
    """Always fails with a plain exception (not serialized by pydantic-ai's activity wrapper)."""
    raise ValueError('bad value: nope')


PNG_HEADER = b'\x89PNG\r\n\x1a\n'
BLOB = PNG_HEADER + bytes(range(256))


def get_images() -> list[BinaryContent]:
    """Return binary image content."""
    return [BinaryContent(data=BLOB, media_type='image/png')]


_captured_tool_defs: list[list[ToolDefinition]] = []


# FunctionModel that emits a run_code tool call for the given code snippet.
def _code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that generates a run_code call on the first request, then returns the result as text."""
    _captured_tool_defs.append(info.function_tools)

    # Check if we already got a tool result back.
    for msg in messages:
        if isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])

    # First call -- emit run_code.
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='run_code',
                args={'code': 'result = await add(a=3, b=4)\nresult'},
                tool_call_id='test_tc_1',
            )
        ]
    )


code_mode = TemporalCodeMode()

code_mode_agent = Agent(
    FunctionModel(_code_mode_model),
    name='code_mode_temporal_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[code_mode],
)

temporal_code_mode_agent = TemporalAgent(
    code_mode_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


def _repl_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that runs two run_code snippets in sequence to exercise REPL persistence."""
    returns = [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    if len(returns) == 0:
        code = (
            "import asyncio\nx, y = await asyncio.gather(add(a=1, b=2), add(a=10, b=20))\nprint('parts', x, y)\nx + y"
        )
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='repl_tc_1')])
    if len(returns) == 1:
        return ModelResponse(
            parts=[ToolCallPart(tool_name='run_code', args={'code': 'x + y + 100'}, tool_call_id='repl_tc_2')]
        )
    return ModelResponse(parts=[TextPart(content=f'first: {returns[0].content} second: {returns[1].content}')])


repl_agent = Agent(
    FunctionModel(_repl_model),
    name='code_mode_temporal_repl_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[code_mode],
)

temporal_repl_agent = TemporalAgent(repl_agent, activity_config=BASE_ACTIVITY_CONFIG)


def _error_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that sends broken code, then code whose tool call fails, then reports both.

    The first snippet has a syntax error, so `run_code` raises `ModelRetry` and
    the model sees a retry prompt. The second snippet calls a tool that raises,
    and catches the rebuilt exception inside the sandbox.
    """
    retries = [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, RetryPromptPart)
    ]
    returns = [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    if not retries:
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': '1 +'}, tool_call_id='err_tc_1')])
    if not returns:
        # `boom` raises ModelRetry (serialized across the activity boundary by
        # pydantic-ai); `bad_value` raises ValueError (reaches the workflow as an
        # ActivityError wrapper, unwrapped back to its original type name), so the
        # sandbox can catch it as ValueError specifically.
        code = (
            'try:\n'
            '    await boom(x=1)\n'
            'except Exception as e:\n'
            "    a = f'{e}'\n"
            'try:\n'
            '    await bad_value(x=1)\n'
            'except ValueError as e:\n'
            "    b = f'{e}'\n"
            "f'{a} | {b}'"
        )
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='err_tc_2')])
    retry_content = retries[0].content
    assert isinstance(retry_content, str)
    return ModelResponse(parts=[TextPart(content=f'retry said: {retry_content} | result: {returns[0].content}')])


error_agent = Agent(
    FunctionModel(_error_model),
    name='code_mode_temporal_error_agent',
    toolsets=[FunctionToolset(tools=[add, boom, bad_value], id='math')],
    capabilities=[code_mode],
)

temporal_error_agent = TemporalAgent(error_agent, activity_config=BASE_ACTIVITY_CONFIG)


def _binary_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model whose snippet pulls binary tool content into the sandbox and inspects it."""
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                    return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])
    code = "imgs = await get_images()\ndata = imgs[0]['data']\n{'head': data[0], 'n': len(data)}"
    return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='bin_tc_1')])


binary_agent = Agent(
    FunctionModel(_binary_model),
    name='code_mode_temporal_binary_agent',
    toolsets=[FunctionToolset(tools=[get_images], id='media')],
    capabilities=[code_mode],
)

temporal_binary_agent = TemporalAgent(binary_agent, activity_config=BASE_ACTIVITY_CONFIG)


def _timeout_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model whose first snippet loops forever, exercising the watchdog-timeout retry path."""
    retries = [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, RetryPromptPart)
    ]
    if not retries:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='run_code', args={'code': 'x = 0\nwhile True:\n    x = x + 1'}, tool_call_id='to_tc_1'
                )
            ]
        )
    retry_content = retries[0].content
    assert isinstance(retry_content, str)
    return ModelResponse(parts=[TextPart(content=f'timeout retry: {retry_content}')])


# A tight watchdog and explicit activity config: runaway sandbox code must be
# killed by the Monty pool (surfacing as a model retry), not by the activity
# start-to-close timeout.
timeout_code_mode = TemporalCodeMode(
    activity_name='timeout_probe',
    request_timeout=1.0,
    activity_config=ActivityConfig(
        start_to_close_timeout=timedelta(seconds=30),
        retry_policy=RetryPolicy(maximum_attempts=1),
    ),
)

timeout_agent = Agent(
    FunctionModel(_timeout_model),
    name='code_mode_temporal_timeout_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[timeout_code_mode],
)

temporal_timeout_agent = TemporalAgent(timeout_agent, activity_config=BASE_ACTIVITY_CONFIG)


def _hops_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model whose first snippet needs two rounds of tool calls, against a one-round cap.

    After the cap aborts it, the second snippet needs exactly one round, which the
    cap must allow (including processing that round's completion).
    """
    retries = [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, RetryPromptPart)
    ]
    returns = [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    if not retries:
        code = 'a = await add(a=1, b=1)\nb = await add(a=a, b=1)\nb'
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='hop_tc_1')])
    if not returns:
        code = 'a = await add(a=2, b=3)\na'
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='hop_tc_2')])
    retry_content = retries[0].content
    assert isinstance(retry_content, str)
    return ModelResponse(parts=[TextPart(content=f'hops retry: {retry_content} | ok: {returns[0].content}')])


hops_code_mode = TemporalCodeMode(activity_name='hops_probe', max_hops=1)

hops_agent = Agent(
    FunctionModel(_hops_model),
    name='code_mode_temporal_hops_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[hops_code_mode],
)

temporal_hops_agent = TemporalAgent(hops_agent, activity_config=BASE_ACTIVITY_CONFIG)

# Plain CodeMode must be rejected inside a workflow with a pointer to TemporalCodeMode.
plain_code_mode_agent = Agent(
    FunctionModel(_code_mode_model),
    name='plain_code_mode_temporal_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode()],
)

temporal_plain_agent = TemporalAgent(plain_code_mode_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class CodeModeWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        result = await temporal_code_mode_agent.run(prompt)
        return {
            'output': str(result.output),
            'messages': result.all_messages_json().decode(),
        }


@workflow.defn
class ReplWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_repl_agent.run(prompt)
        return str(result.output)


@workflow.defn
class SequentialReplWorkflow:
    """Same as ReplWorkflow, but with the run opted into sequential tool execution."""

    @workflow.run
    async def run(self, prompt: str) -> str:
        with ToolManager.parallel_execution_mode('sequential'):
            result = await temporal_repl_agent.run(prompt)
        return str(result.output)


@workflow.defn
class BinaryWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_binary_agent.run(prompt)
        return str(result.output)


@workflow.defn
class ErrorWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_error_agent.run(prompt)
        return str(result.output)


@workflow.defn
class TimeoutWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_timeout_agent.run(prompt)
        return str(result.output)


@workflow.defn
class HopsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_hops_agent.run(prompt)
        return str(result.output)


@workflow.defn
class PlainCodeModeWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_plain_agent.run(prompt)
        return str(result.output)  # pragma: no cover


@workflow.defn
class SandboxRestrictionWorkflow:
    """Probe that the workflow sandbox still blocks Python subprocess calls."""

    @workflow.run
    async def run(self) -> str:
        try:
            subprocess.run([sys.executable, '-c', 'pass'], check=True)
        except RestrictedWorkflowAccessError as e:
            return e.qualified_name
        return 'subprocess was allowed'  # pragma: no cover


ALL_WORKFLOWS = [
    CodeModeWorkflow,
    ReplWorkflow,
    SequentialReplWorkflow,
    BinaryWorkflow,
    ErrorWorkflow,
    TimeoutWorkflow,
    HopsWorkflow,
    PlainCodeModeWorkflow,
    SandboxRestrictionWorkflow,
]

ALL_PLUGINS = [
    AgentPlugin(temporal_code_mode_agent),
    AgentPlugin(temporal_repl_agent),
    AgentPlugin(temporal_binary_agent),
    AgentPlugin(temporal_error_agent),
    AgentPlugin(temporal_timeout_agent),
    AgentPlugin(temporal_hops_agent),
    AgentPlugin(temporal_plain_agent),
    CodeModePlugin(code_mode),
    CodeModePlugin(timeout_code_mode),
    CodeModePlugin(hops_code_mode),
]


# ---------------------------------------------------------------------------
# Workflow integration tests
# ---------------------------------------------------------------------------


async def test_code_mode_runs_in_temporal_workflow(client: Client) -> None:
    """CodeMode works inside a Temporal workflow without whitelisting Monty.

    The sandbox execution runs in `start`/`resume` activities; the nested
    `add` call is dispatched from the workflow as its own activity; and the
    recorded history replays cleanly without re-executing any Monty code.
    """
    _captured_tool_defs.clear()
    workflow_id = 'test_code_mode_temporal_1'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        result = await client.execute_workflow(
            CodeModeWorkflow.run,
            args=['Calculate 3 + 4'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        sandbox_result = await client.execute_workflow(
            SandboxRestrictionWorkflow.run,
            id='test_code_mode_temporal_sandbox_restrictions',
            task_queue=TASK_QUEUE,
        )

    assert result['output'] == 'done: 7'
    assert sandbox_result == 'subprocess.run.__call__'

    messages = json.loads(result['messages'])
    assert len(messages) == 4

    # 1. User prompt
    assert messages[0]['kind'] == 'request'
    assert messages[0]['parts'][0]['part_kind'] == 'user-prompt'
    assert messages[0]['parts'][0]['content'] == 'Calculate 3 + 4'

    # 2. Model response -- run_code tool call
    assert messages[1]['kind'] == 'response'
    tc = messages[1]['parts'][0]
    assert tc['part_kind'] == 'tool-call'
    assert tc['tool_name'] == 'run_code'
    assert tc['args'] == {'code': 'result = await add(a=3, b=4)\nresult'}
    assert tc['tool_call_id'] == 'test_tc_1'

    # 3. Tool return with nested tool call metadata
    assert messages[2]['kind'] == 'request'
    tr = messages[2]['parts'][0]
    assert tr['part_kind'] == 'tool-return'
    assert tr['tool_name'] == 'run_code'
    assert tr['content'] == 7
    assert tr['tool_call_id'] == 'test_tc_1'

    # Verify nested tool call/return metadata
    metadata = tr['metadata']
    assert metadata is not None
    assert metadata['code_mode'] is True
    nested_calls = metadata['tool_calls']
    nested_returns = metadata['tool_returns']
    assert len(nested_calls) == 1
    assert len(nested_returns) == 1

    nested_call = next(iter(nested_calls.values()))
    assert nested_call['tool_name'] == 'add'
    assert nested_call['args'] == {'a': 3, 'b': 4}

    nested_return = next(iter(nested_returns.values()))
    assert nested_return['tool_name'] == 'add'
    assert nested_return['content'] == 7
    assert nested_return['tool_call_id'] == nested_call['tool_call_id']

    # 4. Final text response
    assert messages[3]['kind'] == 'response'
    assert messages[3]['parts'][0]['part_kind'] == 'text'
    assert messages[3]['parts'][0]['content'] == 'done: 7'

    # 5. Verify tool definitions sent to the model
    assert len(_captured_tool_defs) == 2
    for tool_defs in _captured_tool_defs:
        tool_names = [td.name for td in tool_defs]
        # CodeMode wraps `add` into `run_code` -- the model should only see `run_code`
        assert 'run_code' in tool_names
        assert 'add' not in tool_names

        run_code_td = next(td for td in tool_defs if td.name == 'run_code')
        assert run_code_td.description is not None
        assert 'async def add' in run_code_td.description
        assert run_code_td.parameters_json_schema['properties']['code']['type'] == 'string'

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    replay_result = await Replayer(
        workflows=[CodeModeWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
    ).replay_workflow(history)
    assert replay_result.replay_failure is None


async def test_repl_state_and_parallel_calls(client: Client) -> None:
    """REPL variables survive across run_code calls, and gathered calls settle in one hop."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            ReplWorkflow.run,
            args=['sum things'],
            id='test_code_mode_temporal_repl',
            task_queue=TASK_QUEUE,
        )

    # First snippet: print + gathered adds; second snippet reuses x and y from REPL state.
    assert output == "first: {'output': 'parts 3 30\\n', 'result': 33} second: 133"


async def test_sequential_execution_mode(client: Client) -> None:
    """A run opted into sequential tool execution dispatches pending calls one at a time."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            SequentialReplWorkflow.run,
            args=['sum things'],
            id='test_code_mode_temporal_repl_sequential',
            task_queue=TASK_QUEUE,
        )

    assert output == "first: {'output': 'parts 3 30\\n', 'result': 33} second: 133"


async def test_binary_tool_content_reaches_sandbox(client: Client) -> None:
    """Binary tool results survive the activity payload boundary byte-for-byte.

    Raw `bytes` inside an untyped payload field would fail Temporal's pydantic
    JSON converter (or silently decode to `str` when UTF-8-clean), so values
    crossing the activity boundary are base64-wrapped on the wire.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            BinaryWorkflow.run,
            args=['inspect the image'],
            id='test_code_mode_temporal_binary',
            task_queue=TASK_QUEUE,
        )

    assert output == f"done: {{'head': {BLOB[0]}, 'n': {len(BLOB)}}}"


async def test_code_error_and_tool_exception(client: Client) -> None:
    """Broken code surfaces as a ModelRetry, and a failing tool's exception is catchable in-sandbox.

    The broken snippet lands on a fresh (type-checked) REPL, and Monty's type
    checker parses before execution, so the parse failure is reported as a
    type error -- same as the local execution path.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            ErrorWorkflow.run,
            args=['do something broken'],
            id='test_code_mode_temporal_error',
            task_queue=TASK_QUEUE,
        )

    assert output.startswith('retry said: Type error in code:')
    assert 'invalid-syntax' in output
    # The rebuilt exceptions keep their original type names and messages: the
    # ModelRetry propagates typed through pydantic-ai's activity wrapper, and the
    # ValueError is unwrapped from the ActivityError cause chain.
    assert output.endswith('result: boom hint: try something else | bad value: nope')


async def test_runaway_code_hits_watchdog_not_workflow(client: Client) -> None:
    """An infinite loop is killed by the Monty watchdog and surfaces as a model retry.

    Under the old in-workflow design this would have blocked the workflow
    thread and tripped Temporal's 2-second deadlock detector; here the loop
    burns inside an activity until `request_timeout` kills the worker.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            TimeoutWorkflow.run,
            args=['loop forever'],
            id='test_code_mode_temporal_timeout',
            task_queue=TASK_QUEUE,
        )

    assert output.startswith('timeout retry: The code exceeded the sandbox execution time limit')


async def test_tool_call_rounds_are_capped(client: Client) -> None:
    """A snippet needing more tool-call rounds than `max_hops` is abandoned with a retry.

    Every round is at least one recorded activity, so without the cap a
    tool-call loop in model code would grow the workflow history without limit.
    A snippet needing exactly `max_hops` rounds must still complete, including
    the final round's result.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            HopsWorkflow.run,
            args=['add twice'],
            id='test_code_mode_temporal_hops',
            task_queue=TASK_QUEUE,
        )

    assert output.startswith('hops retry: The code needed more than 1 rounds of tool calls')
    assert output.endswith('| ok: 5')


async def test_plain_code_mode_fails_fast_in_workflow(client: Client) -> None:
    """Plain CodeMode inside a workflow fails immediately with a pointer to TemporalCodeMode.

    Without the guard this would surface as a sandbox restriction error from
    Monty's binary resolution and the workflow task would retry forever.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        plugins=ALL_PLUGINS,
        workflow_runner=_workflow_runner(),
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                PlainCodeModeWorkflow.run,
                args=['Calculate 3 + 4'],
                id='test_code_mode_temporal_plain_guard',
                task_queue=TASK_QUEUE,
            )

    assert 'TemporalCodeMode' in str(exc_info.value.cause)


async def test_temporal_code_mode_outside_workflow_runs_locally() -> None:
    """Outside a workflow, TemporalCodeMode behaves exactly like CodeMode (local Monty pool)."""
    agent = Agent(
        FunctionModel(_code_mode_model),
        name='code_mode_local_fallback_agent',
        toolsets=[FunctionToolset(tools=[add], id='math')],
        capabilities=[TemporalCodeMode(activity_name='local_fallback')],
    )
    result = await agent.run('Calculate 3 + 4')
    assert result.output == 'done: 7'


# ---------------------------------------------------------------------------
# Activity protocol unit tests (no Temporal server)
# ---------------------------------------------------------------------------


def _start_params(code: str, **overrides: Any) -> StartParams:
    params = StartParams(
        code=code,
        repl_state=None,
        type_check=False,
        type_check_stubs=None,
        valid_names=['add', 'ask'],
        sequential_names=['ask'],
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


class TestSandboxActivities:
    """Drives the start/resume activities directly, as the workflow would."""

    capability = TemporalCodeMode(activity_name='unit')

    @property
    def start(self) -> Any:
        return self.capability.temporal_activities[0]

    @property
    def resume(self) -> Any:
        return self.capability.temporal_activities[1]

    async def test_complete_without_tool_calls(self) -> None:
        result = await self.start(_start_params("print('hi')\n1 + 1"))
        assert result.status == 'complete'
        assert result.output == 2
        assert result.printed == 'hi\n'
        assert result.repl_state is not None

    async def test_suspend_dispatch_resume(self) -> None:
        result = await self.start(_start_params('r = await add(a=3, b=4)\nr'))
        assert result.status == 'pending'
        assert [(c.name, c.kwargs) for c in result.calls] == [('add', {'a': 3, 'b': 4})]
        assert result.snapshot is not None

        done = await self.resume(
            ResumeParams(
                snapshot=result.snapshot,
                results=[SettledCall(call_id=result.calls[0].call_id, value=7)],
                valid_names=['add', 'ask'],
                sequential_names=['ask'],
            )
        )
        assert done.status == 'complete'
        assert done.output == 7

    async def test_repl_state_round_trip(self) -> None:
        first = await self.start(_start_params('x = 41'))
        assert first.status == 'complete'
        second = await self.start(_start_params('x + 1', repl_state=first.repl_state))
        assert second.status == 'complete'
        assert second.output == 42

    async def test_sequential_tool_suspends_at_call_site(self) -> None:
        # `add` is async (deferred); `ask` is sequential (rendered as a sync `def`), so
        # its call suspends immediately, carrying the earlier deferred call first to
        # preserve the local executor's barrier ordering.
        code = 'f = add(a=1, b=2)\ns = ask(q="hi")\nr = await f\n[s, r]'
        result = await self.start(_start_params(code))
        assert result.status == 'pending'
        assert [(c.name, c.kwargs) for c in result.calls] == [('add', {'a': 1, 'b': 2}), ('ask', {'q': 'hi'})]
        assert result.snapshot is not None

        settled = [
            SettledCall(call_id=result.calls[0].call_id, value=3),
            SettledCall(call_id=result.calls[1].call_id, value='hello'),
        ]
        done = await self.resume(
            ResumeParams(
                snapshot=result.snapshot, results=settled, valid_names=['add', 'ask'], sequential_names=['ask']
            )
        )
        assert done.status == 'complete'
        assert done.output == ['hello', 3]

    async def test_unknown_function_raises_name_error(self) -> None:
        result = await self.start(_start_params('await missing(a=1)', valid_names=['add']))
        assert result.status == 'error'
        assert result.error_kind == 'runtime'
        assert 'Unknown function: missing' in result.error_display

    async def test_positional_arguments_rejected(self) -> None:
        result = await self.start(_start_params('await add(1, 2)'))
        assert result.status == 'error'
        assert result.error_kind == 'runtime'
        assert 'does not accept positional arguments' in result.error_display

    async def test_undefined_name_raises_name_error(self) -> None:
        result = await self.start(_start_params('undefined_thing'))
        assert result.status == 'error'
        assert result.error_kind == 'runtime'
        assert 'NameError' in result.error_display

    async def test_syntax_error(self) -> None:
        result = await self.start(_start_params('1 +'))
        assert result.status == 'error'
        assert result.error_kind == 'syntax'

    async def test_typing_error_with_stubs(self) -> None:
        stubs = 'async def add(*, a: int, b: int) -> int:\n    raise NotImplementedError()'
        result = await self.start(_start_params('await add(a="nope", b=4)', type_check=True, type_check_stubs=stubs))
        assert result.status == 'error'
        assert result.error_kind == 'typing'

    async def test_runtime_error_keeps_printed_output(self) -> None:
        result = await self.start(_start_params("print('before')\n1 / 0"))
        assert result.status == 'error'
        assert result.error_kind == 'runtime'
        assert result.printed == 'before\n'
        assert 'ZeroDivisionError' in result.error_display

    async def test_builtin_exception_is_catchable_by_type(self) -> None:
        result = await self.start(
            _start_params('try:\n    r = await add(a=1, b=2)\nexcept KeyError as e:\n    r = 0\nr')
        )
        assert result.status == 'pending'
        done = await self.resume(
            ResumeParams(
                snapshot=result.snapshot,
                results=[
                    SettledCall(call_id=result.calls[0].call_id, exception_type='KeyError', exception_message='k')
                ],
                valid_names=['add', 'ask'],
                sequential_names=['ask'],
            )
        )
        assert done.status == 'complete'
        assert done.output == 0

    async def test_custom_exception_keeps_type_name(self) -> None:
        code = 'try:\n    r = await add(a=1, b=2)\nexcept Exception as e:\n    r = f"caught {e}"\nr'
        result = await self.start(_start_params(code))
        assert result.status == 'pending'
        done = await self.resume(
            ResumeParams(
                snapshot=result.snapshot,
                results=[
                    SettledCall(
                        call_id=result.calls[0].call_id,
                        exception_type='SomeCustomToolError',
                        exception_message='custom boom',
                    )
                ],
                valid_names=['add', 'ask'],
                sequential_names=['ask'],
            )
        )
        assert done.status == 'complete'
        assert done.output == 'caught custom boom'

    async def test_exception_type_with_unfriendly_constructor(self) -> None:
        # UnicodeDecodeError's constructor needs five arguments, and `license` is a
        # builtin that is not an exception type at all; both fall back to a
        # dynamically named Exception subclass instead of failing to rebuild.
        for type_name in ('UnicodeDecodeError', 'license'):
            code = 'try:\n    r = await add(a=1, b=2)\nexcept Exception as e:\n    r = f"caught {e}"\nr'
            result = await self.start(_start_params(code))
            assert result.status == 'pending'
            done = await self.resume(
                ResumeParams(
                    snapshot=result.snapshot,
                    results=[
                        SettledCall(call_id=result.calls[0].call_id, exception_type=type_name, exception_message='odd')
                    ],
                    valid_names=['add', 'ask'],
                    sequential_names=['ask'],
                )
            )
            assert done.status == 'complete'
            assert done.output == 'caught odd'

    async def test_watchdog_timeout_reports_crash(self) -> None:
        capability = TemporalCodeMode(activity_name='unit_timeout', request_timeout=0.3)
        start = capability.temporal_activities[0]
        result = await start(_start_params('x = 0\nwhile True:\n    x = x + 1'))
        assert result.status == 'error'
        assert result.error_kind == 'crash-timeout'

    async def test_host_side_panic_reports_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A Rust-side panic surfacing host-side (pyo3 PanicException) must come back as
        # a crash result, not an activity failure. Injected via the drive loop because
        # Monty does not panic on any input we can construct.
        class PanicException(BaseException):
            """Named to match the pyo3 panic class `is_sandbox_panic` recognizes."""

        async def _panic(*args: Any, **kwargs: Any) -> Any:
            raise PanicException('sandbox aborted')

        monkeypatch.setattr('pydantic_ai_harness.code_mode.temporal._drive', _panic)
        result = await self.start(_start_params('1 + 1'))
        assert result.status == 'error'
        assert result.error_kind == 'crash'
        assert 'sandbox aborted' in result.error_display

    async def test_non_panic_base_exception_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The activity's panic guard catches BaseException but must re-raise anything
        # that is not a VM panic (e.g. cancellation).
        class _Boom(BaseException):
            pass

        async def _boom(*args: Any, **kwargs: Any) -> Any:
            raise _Boom('boom')

        monkeypatch.setattr('pydantic_ai_harness.code_mode.temporal._drive', _boom)
        with pytest.raises(_Boom):
            await self.start(_start_params('1 + 1'))

    async def test_reserved_wire_key_is_decoded_or_passed_through(self) -> None:
        # The base64 wire sentinel key is reserved: a tool result shaped exactly like a
        # wrapped bytes value decodes to bytes, and a sentinel-shaped dict whose value is
        # not valid base64 passes through unchanged instead of raising mid-protocol.
        # Observed from inside the sandbox, since the activity result re-encodes bytes.
        for code, payload, expected in (
            # Decoded: the sandbox sees b'ABC', whose len is 3 (the dict's len would be 1).
            ('r = await add(a=1, b=2)\nlen(r)', {'__pydantic_ai_harness_bytes_b64__': 'QUJD'}, 3),
            # Passed through: the sandbox sees the dict unchanged.
            (
                "r = await add(a=1, b=2)\nr['__pydantic_ai_harness_bytes_b64__']",
                {'__pydantic_ai_harness_bytes_b64__': 'not base64!'},
                'not base64!',
            ),
        ):
            result = await self.start(_start_params(code))
            assert result.status == 'pending'
            done = await self.resume(
                ResumeParams(
                    snapshot=result.snapshot,
                    results=[SettledCall(call_id=result.calls[0].call_id, value=payload)],
                    valid_names=['add', 'ask'],
                    sequential_names=['ask'],
                )
            )
            assert done.status == 'complete'
            assert done.output == expected

    async def test_unserializable_output_reports_code_error(self) -> None:
        # A dict with a non-UTF-8 bytes key defeats the payload converter and the bytes
        # wire wrapping (which only covers values). The activity must dry-run the result
        # serialization and report a revisable code error instead of failing the activity.
        result = await self.start(_start_params('{b"\\xff\\xfe": 1}'))
        assert result.status == 'error'
        assert result.error_kind == 'runtime'
        assert 'cannot be returned from the sandbox' in result.error_display
