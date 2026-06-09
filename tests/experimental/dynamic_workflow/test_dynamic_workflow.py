"""Tests for the DynamicWorkflow capability."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_ai_harness.experimental.dynamic_workflow import DynamicWorkflow, WorkflowAgent
from pydantic_ai_harness.experimental.dynamic_workflow._toolset import (  # pyright: ignore[reportPrivateUsage]
    DynamicWorkflowToolset,
    _default_resource_limits,
    _in_workflow,
    _render_catalog,
    _resolve_resource_limits,
)

pytestmark = pytest.mark.anyio

# Builds a `WorkflowAgent` wrapping a trivial `TestModel` sub-agent, with overridable
# output text, sandbox name, and catalog description.
MakeAgent = Callable[..., WorkflowAgent[None]]


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (the shared Monty loop uses asyncio)."""
    return 'asyncio'


def _sub_agent(text: str = 'ok', name: str | None = 'sub') -> Agent[None, str]:
    return Agent(TestModel(custom_output_text=text), name=name)


@pytest.fixture
def make_agent() -> MakeAgent:
    """Factory fixture: build a `WorkflowAgent` whose sandbox name defaults to the agent's `name`."""

    def _make(text: str = 'ok', name: str | None = 'sub', description: str | None = None) -> WorkflowAgent[None]:
        return WorkflowAgent(agent=_sub_agent(text, name), description=description)

    return _make


def _ctx() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1)


def _ctx_with_queue() -> RunContext[None]:
    """A `RunContext` with a live pending-message queue, so `enqueue` (used by reveal) works."""
    return RunContext[None](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1, pending_messages=[]
    )


def _enqueued_text(ctx: RunContext[None]) -> str:
    """Join the user-prompt text of every message enqueued on `ctx` (reveal announcements)."""
    return '\n'.join(
        part.content
        for pending in ctx.pending_messages or []
        for message in pending.messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


async def _run_script(ts: DynamicWorkflowToolset[None], code: str, ctx: RunContext[None] | None = None) -> Any:
    ctx = ctx or _ctx()
    tools = await ts.get_tools(ctx)
    tool = tools[ts.tool_name]
    return await ts.call_tool(ts.tool_name, {'code': code}, ctx, tool)


# --- Construction / wiring -------------------------------------------------


def test_capability_provides_toolset_with_propagated_config(make_agent: MakeAgent) -> None:
    reviewer = make_agent(name='reviewer')
    cap = DynamicWorkflow[None](
        agents=[reviewer],
        tool_name='orchestrate',
        max_agent_calls=7,
        max_retries=1,
        forward_usage=False,
        id='wf',
    )
    toolset = cap.get_toolset()
    assert isinstance(toolset, DynamicWorkflowToolset)
    assert toolset.agents == [reviewer]
    assert toolset.tool_name == 'orchestrate'
    assert toolset.max_agent_calls == 7
    assert toolset.max_retries == 1
    assert toolset.forward_usage is False
    # The toolset id derives from the capability id for durable execution.
    assert toolset.id == 'wf'


def test_toolset_id_defaults_to_tool_name(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    assert ts.id == 'run_workflow'


def test_defer_loading_flag(make_agent: MakeAgent) -> None:
    cap = DynamicWorkflow[None](agents=[make_agent()], id='wf', defer_loading=True)
    assert cap.defer_loading is True
    assert cap.id == 'wf'


def test_not_spec_serializable() -> None:
    # `agents` holds live Agent objects, so the capability opts out of spec construction.
    assert DynamicWorkflow.get_serialization_name() is None


# --- Name resolution -------------------------------------------------------


def test_sandbox_name_falls_back_to_agent_name() -> None:
    ts = DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=_sub_agent(name='reviewer'))])
    assert set(ts._by_name) == {'reviewer'}  # pyright: ignore[reportPrivateUsage]


def test_explicit_name_overrides_agent_name() -> None:
    # The sandbox handle can differ from the agent's own `name`.
    ts = DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=_sub_agent(name='reviewer'), name='check')])
    assert set(ts._by_name) == {'check'}  # pyright: ignore[reportPrivateUsage]


def test_workflow_agent_resolved_name() -> None:
    assert WorkflowAgent(agent=_sub_agent(name='reviewer')).resolved_name == 'reviewer'
    assert WorkflowAgent(agent=_sub_agent(name='reviewer'), name='check').resolved_name == 'check'
    assert WorkflowAgent(agent=_sub_agent(name=None)).resolved_name is None


# --- Validation ------------------------------------------------------------


def test_invalid_identifier_name_raises() -> None:
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=_sub_agent(), name='bad-name')])


def test_keyword_name_raises() -> None:
    # `'class'.isidentifier()` is True, but a Python keyword can't be a sandbox function name —
    # the model could never call it (`await class(...)` is a syntax error). Reject it up front.
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=_sub_agent(), name='class')])


def test_empty_agents_raises() -> None:
    with pytest.raises(UserError, match='at least one sub-agent'):
        DynamicWorkflowToolset[None](agents=[])


def test_missing_name_raises() -> None:
    # No explicit name and the agent has no `name` either: nothing to expose as a function.
    with pytest.raises(UserError, match='has no `name`'):
        DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=_sub_agent(name=None))])


def test_duplicate_name_raises(make_agent: MakeAgent) -> None:
    with pytest.raises(UserError, match='must be unique'):
        DynamicWorkflowToolset[None](agents=[make_agent(name='dup'), make_agent(name='dup')])


# --- Catalog rendering / discovery surface ---------------------------------


async def test_description_lists_agents_as_functions(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent(name='reviewer')])
    tools = await ts.get_tools(_ctx())
    desc = tools['run_workflow'].tool_def.description
    assert desc is not None
    assert 'async def reviewer(*, task: str) -> Any:' in desc
    # No description -> bare signature, not a useless `"""reviewer"""` docstring echoing the name.
    assert '"""reviewer"""' not in desc


async def test_descriptions_override(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent(name='reviewer', description='Reviews code for bugs.')])
    tools = await ts.get_tools(_ctx())
    desc = tools['run_workflow'].tool_def.description
    assert desc is not None
    assert '"""Reviews code for bugs."""' in desc


async def test_custom_tool_name(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()], tool_name='orchestrate')
    tools = await ts.get_tools(_ctx())
    assert set(tools) == {'orchestrate'}


def test_render_catalog_without_description() -> None:
    out = _render_catalog({'sub': None})
    assert 'async def sub(*, task: str) -> Any:' in out
    assert '    ...' in out


# --- Execution -------------------------------------------------------------


async def test_single_sub_agent_call(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent('looks good', 'reviewer')])
    out = await _run_script(ts, "await reviewer(task='check')")
    assert out == 'looks good'


async def test_parallel_fan_out(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent('r', 'reviewer'), make_agent('s', 'summarizer')])
    code = "import asyncio\nawait asyncio.gather(reviewer(task='a'), reviewer(task='b'), summarizer(task='c'))"
    out = await _run_script(ts, code)
    assert out == ['r', 'r', 's']


async def test_chaining(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent('done', 'sub')])
    code = "a = await sub(task='one')\nb = await sub(task='two: ' + a)\n[a, b]"
    out = await _run_script(ts, code)
    assert out == ['done', 'done']


async def test_structured_output_arrives_as_dict() -> None:
    class Review(BaseModel):
        score: int
        note: str

    reviewer: Agent[None, Review] = Agent(
        TestModel(custom_output_args={'score': 9, 'note': 'great'}), name='reviewer', output_type=Review
    )
    ts = DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=reviewer)])
    out = await _run_script(ts, "r = await reviewer(task='x')\nr['score']")
    assert out == 9


async def test_via_agent_run_end_to_end() -> None:
    observed_returns: list[Any] = []
    seen_tools: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append([td.name for td in info.function_tools])
        ret = next(
            (
                p
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, ToolReturnPart) and p.tool_name == 'run_workflow'
            ),
            None,
        )
        if ret is not None:
            observed_returns.append(ret.content)
            return ModelResponse(parts=[TextPart(f'done: {ret.content}')])
        code = "import asyncio\nresults = await asyncio.gather(reviewer(task='a'), reviewer(task='b'))\nresults"
        return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': code})])

    reviewer = _sub_agent('reviewed', 'reviewer')
    agent: Agent[None, str] = Agent(
        FunctionModel(model_fn), capabilities=[DynamicWorkflow[None](agents=[WorkflowAgent(agent=reviewer)])]
    )
    result = await agent.run('please review')
    # Model is shown only the orchestration tool, not the sub-agents directly.
    assert seen_tools[0] == ['run_workflow']
    assert observed_returns == [['reviewed', 'reviewed']]
    assert result.output == "done: ['reviewed', 'reviewed']"


# --- Budget and guards -----------------------------------------------------


async def test_max_agent_calls_enforced_exactly() -> None:
    runs: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        runs.append('x')
        return ModelResponse(parts=[TextPart('ok')])

    counted: Agent[None, str] = Agent(FunctionModel(model_fn), name='counted')
    ts = DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=counted)], max_agent_calls=2)
    code = 'for i in range(5):\n    await counted(task=str(i))'
    # Budget exhaustion returns a terminal result (not a retry that can never succeed).
    out = await _run_script(ts, code)
    assert isinstance(out, dict)
    assert 'budget' in out['error']
    # Exactly the budget ran before the next call was refused.
    assert len(runs) == 2


async def test_max_agent_calls_exact_under_concurrent_fan_out() -> None:
    runs: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        runs.append('x')
        return ModelResponse(parts=[TextPart('ok')])

    counted: Agent[None, str] = Agent(FunctionModel(model_fn), name='counted')
    ts = DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=counted)], max_agent_calls=3)
    # Fan eight calls out concurrently: the await-free check-then-increment must admit exactly the
    # budget (3), not race past it. A regression inserting an `await` before the increment fails here.
    code = 'import asyncio\nawait asyncio.gather(*[counted(task=str(i)) for i in range(8)])'
    out = await _run_script(ts, code)
    assert isinstance(out, dict)
    assert 'budget' in out['error']
    assert len(runs) == 3


def test_max_agent_calls_must_be_positive(make_agent: MakeAgent) -> None:
    with pytest.raises(UserError, match='max_agent_calls'):
        DynamicWorkflowToolset[None](agents=[make_agent()], max_agent_calls=0)


async def test_nested_workflow_is_refused(make_agent: MakeAgent) -> None:
    # A workflow already in progress (the flag a sub-agent would inherit) refuses to start another.
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    token = _in_workflow.set(True)
    try:
        with pytest.raises(ModelRetry, match='do not nest'):
            await _run_script(ts, "await sub(task='x')")
    finally:
        _in_workflow.reset(token)


async def test_unknown_agent_raises_model_retry(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(ModelRetry, match='Unknown function'):
        await _run_script(ts, "await nonexistent(task='x')")


async def test_missing_task_kwarg(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(ModelRetry, match='missing required keyword argument'):
        await _run_script(ts, 'await sub(wrong=1)')


async def test_extra_kwargs_rejected(make_agent: MakeAgent) -> None:
    # An extra kwarg must not be silently dropped — the model needs to know it was ignored.
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(ModelRetry, match='unexpected keyword argument'):
        await _run_script(ts, "await sub(task='x', foo='y')")


async def test_non_str_task_rejected(make_agent: MakeAgent) -> None:
    # A non-string task (a dict/list would otherwise be silently smeared into message parts).
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(ModelRetry, match='task must be a string'):
        await _run_script(ts, "await sub(task={'k': 'v'})")


async def test_forward_usage_true_shares_parent_usage(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()], forward_usage=True)
    ctx = _ctx()
    await _run_script(ts, "await sub(task='x')", ctx)
    assert ctx.usage.requests > 0


async def test_forward_usage_false_isolates_usage(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()], forward_usage=False)
    ctx = _ctx()
    await _run_script(ts, "await sub(task='x')", ctx)
    assert ctx.usage.requests == 0


async def test_sub_agent_usage_limits_enforced() -> None:
    sub: Agent[None, str] = Agent(TestModel(), name='limited')

    @sub.tool_plain
    def helper() -> str:
        return 'used'

    # A tool-using TestModel agent needs two model requests (call the tool, then answer), so a
    # request_limit of 1 must trip the second — proving the limit reaches the sub-agent run.
    ts = DynamicWorkflowToolset[None](
        agents=[WorkflowAgent(agent=sub)],
        forward_usage=False,
        sub_agent_usage_limits=UsageLimits(request_limit=1),
    )
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, "await limited(task='x')")
    msg = str(exc_info.value)
    assert 'limited' in msg
    assert 'UsageLimitExceeded' in msg


async def test_sub_agent_usage_limits_generous_allows_run() -> None:
    sub: Agent[None, str] = Agent(TestModel(custom_output_text='done'), name='roomy')

    @sub.tool_plain
    def helper() -> str:
        return 'used'

    # The same two-request agent runs fine when the limit is high enough, so the cap is the
    # configured value rather than a blanket rejection.
    ts = DynamicWorkflowToolset[None](
        agents=[WorkflowAgent(agent=sub)],
        forward_usage=False,
        sub_agent_usage_limits=UsageLimits(request_limit=5),
    )
    out = await _run_script(ts, "await roomy(task='x')")
    assert out == 'done'


# --- Errors / output shapes ------------------------------------------------


async def test_syntax_error(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(ModelRetry, match='Syntax error'):
        await _run_script(ts, 'this is not valid python !!!')


async def test_runtime_error(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(ModelRetry, match='Runtime error'):
        await _run_script(ts, 'x = 1 / 0')


async def test_duplicate_future_in_gather_is_retryable(make_agent: MakeAgent) -> None:
    # Awaiting the same sub-agent call twice in one gather makes the Monty VM panic; that panic
    # must surface as a retry, not tear down the whole agent run.
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    code = 'import asyncio\nf = sub(task="x")\nawait asyncio.gather(f, f)'
    with pytest.raises(ModelRetry, match='aborted inside the sandbox'):
        await _run_script(ts, code)


async def test_non_panic_base_exception_propagates(make_agent: MakeAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    # The panic guard catches BaseException but must re-raise anything that is not a VM panic.
    class _Boom(BaseException):
        pass

    async def _boom(self: Any, state: Any) -> Any:
        raise _Boom('boom')

    monkeypatch.setattr('pydantic_ai_harness._monty_exec.MontyExecutor.run', _boom)
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(_Boom):
        await _run_script(ts, "await sub(task='x')")


async def test_print_only_returns_output_dict(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    out = await _run_script(ts, "print('hello')")
    assert out == {'output': 'hello\n'}


async def test_print_with_result_returns_both(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    out = await _run_script(ts, "print('log')\n42")
    assert out == {'output': 'log\n', 'result': 42}


async def test_no_result_returns_empty_dict(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    out = await _run_script(ts, 'x = 1')
    assert out == {}


async def test_runtime_error_includes_prints(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    with pytest.raises(ModelRetry, match='stdout before error'):
        await _run_script(ts, "print('before crash')\n1 / 0")


async def test_sub_agent_error_does_not_leak_host_internals() -> None:
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('SECRET_TOKEN_sk_abc123 at /Users/victim/secret.py')

    bad: Agent[None, str] = Agent(FunctionModel(boom), name='bad')
    ts = DynamicWorkflowToolset[None](agents=[WorkflowAgent(agent=bad)])
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, "await bad(task='x')")
    msg = str(exc_info.value)
    assert 'SECRET_TOKEN' not in msg
    assert '/Users/victim' not in msg
    assert 'bad' in msg  # the failing agent is named


# --- Sandbox resource limits -----------------------------------------------


async def test_runaway_loop_stopped_by_duration_cap(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()], resource_limits={'max_duration_secs': 0.2})
    with pytest.raises(ModelRetry, match='Runtime error'):
        await _run_script(ts, 'while True:\n    x = 1')


def test_resolve_resource_limits_none_is_backstop() -> None:
    assert _resolve_resource_limits(None) == _default_resource_limits()


def test_resolve_resource_limits_unlimited_removes_all() -> None:
    assert _resolve_resource_limits('unlimited') == {}


def test_resolve_resource_limits_partial_merges_onto_backstop() -> None:
    # A partial dict overrides only the cap it names; the others keep their backstop value
    # (so it can't silently drop the duration backstop).
    resolved = _resolve_resource_limits({'max_memory': 1})
    # Equality with the full backstop (plus the override) proves the duration/allocation
    # backstops survive a partial dict rather than being dropped.
    assert resolved == {**_default_resource_limits(), 'max_memory': 1}


async def test_unlimited_runs_without_a_backstop(make_agent: MakeAgent) -> None:
    # `'unlimited'` resolves to no limits; a trivial script still completes normally.
    ts = DynamicWorkflowToolset[None](agents=[make_agent()], resource_limits='unlimited')
    out = await _run_script(ts, '1 + 1')
    assert out == 2


def test_resolve_resource_limits_rejects_unknown_keys() -> None:
    # A typo'd key (e.g. plural `max_durations_secs`) must not be silently dropped — that would
    # quietly disable the duration cap it was meant to set.
    with pytest.raises(UserError, match='Unknown `resource_limits` key'):
        _resolve_resource_limits({'max_durations_secs': 5})  # pyright: ignore[reportArgumentType]


def test_unknown_resource_limit_key_raises_at_construction(make_agent: MakeAgent) -> None:
    with pytest.raises(UserError, match='Unknown `resource_limits` key'):
        DynamicWorkflowToolset[None](
            agents=[make_agent()],
            resource_limits={'max_durations_secs': 5},  # pyright: ignore[reportArgumentType]
        )


# --- Lifecycle -------------------------------------------------------------


async def test_for_run_resets_budget(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()], max_agent_calls=1)
    await _run_script(ts, "await sub(task='x')")
    assert ts._call_count == 1  # pyright: ignore[reportPrivateUsage]
    fresh = await ts.for_run(_ctx())
    assert isinstance(fresh, DynamicWorkflowToolset)
    assert fresh._call_count == 0  # pyright: ignore[reportPrivateUsage]


async def test_for_run_step_preserves_self(make_agent: MakeAgent) -> None:
    ts = DynamicWorkflowToolset[None](agents=[make_agent()])
    assert await ts.for_run_step(_ctx()) is ts


# --- Runtime reveal --------------------------------------------------------


async def test_runtime_reveal_announces_and_makes_callable(make_agent: MakeAgent) -> None:
    # Appending to the live `agents` list reveals a sub-agent mid-run.
    agents = [make_agent('base-out', 'base')]
    ts = DynamicWorkflowToolset[None](agents=agents)
    ctx = _ctx_with_queue()

    await ts.get_tools(ctx)
    assert set(ts._by_name) == {'base'}  # pyright: ignore[reportPrivateUsage]
    assert _enqueued_text(ctx) == ''  # nothing revealed yet

    agents.append(make_agent('extra-out', 'extra'))
    await ts.get_tools(ctx)
    assert set(ts._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    assert 'async def extra(*, task: str) -> Any:' in _enqueued_text(ctx)

    out = await _run_script(ts, "await extra(task='x')", ctx)
    assert out == 'extra-out'


async def test_reveal_is_idempotent_across_steps(make_agent: MakeAgent) -> None:
    agents = [make_agent('b', 'base')]
    ts = DynamicWorkflowToolset[None](agents=agents)
    ctx = _ctx_with_queue()

    agents.append(make_agent('e', 'extra'))
    await ts.get_tools(ctx)
    await ts.get_tools(ctx)  # re-resolving tools must not re-announce an already-revealed agent
    assert _enqueued_text(ctx).count('async def extra') == 1


async def test_reveal_keeps_tool_description_frozen(make_agent: MakeAgent) -> None:
    # The cached prompt prefix must not change when an agent is revealed.
    agents = [make_agent('b', 'base')]
    ts = DynamicWorkflowToolset[None](agents=agents)
    ctx = _ctx_with_queue()

    before = (await ts.get_tools(ctx))['run_workflow'].tool_def.description
    agents.append(make_agent('e', 'extra'))
    after = (await ts.get_tools(ctx))['run_workflow'].tool_def.description
    assert before == after
    assert after is not None and 'extra' not in after  # the reveal never enters the description


async def test_reveal_skips_invalid_and_duplicate_names(make_agent: MakeAgent) -> None:
    agents = [make_agent('b', 'base')]
    ts = DynamicWorkflowToolset[None](agents=agents)
    ctx = _ctx_with_queue()
    await ts.get_tools(ctx)

    # A nameless agent and one whose name collides with the baseline are both skipped with a
    # warning (not silently), neither becomes callable, and the original is not shadowed.
    agents.append(WorkflowAgent(agent=_sub_agent(name=None)))
    agents.append(make_agent('shadow', 'base'))
    with pytest.warns(UserWarning, match='could not reveal'):
        await ts.get_tools(ctx)
    assert set(ts._by_name) == {'base'}  # pyright: ignore[reportPrivateUsage]
    assert _enqueued_text(ctx) == ''

    # The original baseline agent still runs — it was not shadowed. Re-resolving tools does not
    # re-warn for the same bad entries (warn-once), so this call emits no warning.
    assert await _run_script(ts, "await base(task='x')", ctx) == 'b'


async def test_reveal_end_to_end_via_agent_run() -> None:
    base = _sub_agent('base-done', 'base')
    extra = _sub_agent('extra-done', 'extra')
    # The host keeps a reference to the live catalog (here a closure; in practice often via `deps`).
    agents: list[WorkflowAgent[None]] = [WorkflowAgent(agent=base)]
    saw_announcement: list[bool] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart) and p.tool_name == 'run_workflow'
        ]
        if len(returns) == 0:
            # First step: reveal `extra`, and run `base` now to force a second step.
            agents.append(WorkflowAgent(agent=extra))
            return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': "await base(task='go')"})])
        if len(returns) == 1:
            # Second step: the announcement for `extra` has arrived and it is now callable.
            user_text = '\n'.join(
                p.content
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, UserPromptPart) and isinstance(p.content, str)
            )
            saw_announcement.append('async def extra(*, task: str)' in user_text)
            return ModelResponse(
                parts=[ToolCallPart(tool_name='run_workflow', args={'code': "await extra(task='go')"})]
            )
        return ModelResponse(parts=[TextPart(f'final: {returns[-1].content}')])

    agent: Agent[None, str] = Agent(FunctionModel(model_fn), capabilities=[DynamicWorkflow[None](agents=agents)])
    result = await agent.run('start')
    assert saw_announcement == [True]  # the model saw the reveal announcement, mid-run
    assert result.output == 'final: extra-done'  # and the revealed sub-agent actually ran
