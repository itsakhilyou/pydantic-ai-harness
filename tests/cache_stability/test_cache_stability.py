"""Tests for the observational `CacheStabilityMonitor` capability.

The public behavior is driven through `Agent(..., capabilities=[...])` with a
`FunctionModel` that returns preset `RequestUsage` per step, so each response
carries the `cache_read_tokens` / `cache_write_tokens` the monitor reads. The
repo runs pytest with `filterwarnings=['error']`, so an unexpected
`CacheBustWarning` fails a test on its own; runs that should stay silent assert
that explicitly.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from pydantic_ai_harness.cache_stability import (
    CacheBustWarning,
    CacheStabilityMonitor,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _usage(*, read: int = 0, write: int = 0) -> RequestUsage:
    return RequestUsage(input_tokens=10, output_tokens=5, cache_read_tokens=read, cache_write_tokens=write)


def _agent(usages: list[RequestUsage], monitor: CacheStabilityMonitor[None]) -> Agent[None, str]:
    """Agent whose model emits one preset-usage response per step.

    Every response but the last returns a tool call so the run keeps stepping;
    the last returns text so the run finishes. Each step's `after_model_request`
    sees the matching usage.
    """
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        usage = usages[i]
        if i == len(usages) - 1:
            return ModelResponse(parts=[TextPart('done')], usage=usage)
        return ModelResponse(parts=[ToolCallPart('noop', {})], usage=usage)

    def noop() -> str:
        return 'ok'

    return Agent(FunctionModel(fn), deps_type=type(None), capabilities=[monitor], tools=[noop])


def _agent_from_responses(responses: list[ModelResponse], monitor: CacheStabilityMonitor[None]) -> Agent[None, str]:
    """Agent whose model replays preset `ModelResponse`s, one per step.

    Lets a test control `provider_name` and `timestamp` per response (a model switch, a wall-clock
    gap) -- fields `FunctionModel` leaves untouched -- which the simpler `_agent` helper can't.
    Every response but the last must carry a tool call so the run keeps stepping.
    """
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        return responses[i]

    def noop() -> str:
        return 'ok'

    return Agent(FunctionModel(fn), deps_type=type(None), capabilities=[monitor], tools=[noop])


async def test_collapse_warns() -> None:
    """A large drop in cache_read below the established prefix warns."""
    usages = [_usage(read=0, write=8000), _usage(read=8000, write=200), _usage(read=500)]
    agent = _agent(usages, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning, match='request 3'):
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_stable_prefix_is_silent() -> None:
    """An append-only run whose reads keep pace with the prefix never warns."""
    usages = [_usage(read=0, write=8000), _usage(read=8000, write=200), _usage(read=8200)]
    agent = _agent(usages, CacheStabilityMonitor())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_below_min_prefix_never_warns() -> None:
    """A prefix under `min_prefix_tokens` is too small to judge, so a drop is ignored."""
    usages = [_usage(read=0, write=500), _usage(read=500), _usage(read=10)]
    agent = _agent(usages, CacheStabilityMonitor())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await agent.run('hi')


async def test_tunable_thresholds_catch_smaller_regression() -> None:
    """Lowering the floor and raising the ratio flags a regression the defaults ignore."""
    usages = [_usage(read=0, write=200), _usage(read=150)]
    monitor = CacheStabilityMonitor[None](collapse_ratio=1.0, min_prefix_tokens=100)
    agent = _agent(usages, monitor)
    with pytest.warns(CacheBustWarning):
        await agent.run('hi')


async def test_error_filter_escalates_to_exception() -> None:
    """`filterwarnings('error', ...)` turns a bust into a raised exception (dev/CI enforcement)."""
    usages = [_usage(read=0, write=8000), _usage(read=100)]
    agent = _agent(usages, CacheStabilityMonitor())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        with pytest.raises(CacheBustWarning):
            await agent.run('hi')


async def test_for_run_resets_between_runs() -> None:
    """Reusing one monitor across runs judges each run independently (no leaked high-water mark)."""
    monitor = CacheStabilityMonitor[None]()

    busting = _agent([_usage(read=0, write=8000), _usage(read=100)], monitor)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', CacheBustWarning)
        await busting.run('first')

    # A second run with no caching must not inherit the first run's 8000-token prefix.
    silent = _agent([_usage(read=0, write=0), _usage(read=0, write=0)], monitor)
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        await silent.run('second')


async def test_model_failover_does_not_warn() -> None:
    """A mid-run `FallbackModel` failover reads an empty cache on the new model, which must not warn."""
    a_calls = {'n': 0}

    def model_a(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        a_calls['n'] += 1
        if a_calls['n'] == 1:
            # Establish a large cached prefix on model A, then keep the run stepping.
            return ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000))
        raise ModelAPIError('model-a', 'model A is down')

    def model_b(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # B's cache is empty: it reads back nothing of A's prefix.
        return ModelResponse(parts=[TextPart('done')], usage=_usage(read=0))

    def noop() -> str:
        return 'ok'

    fallback = FallbackModel(
        FunctionModel(model_a, model_name='model-a'),
        FunctionModel(model_b, model_name='model-b'),
    )
    agent = Agent(fallback, deps_type=type(None), capabilities=[CacheStabilityMonitor()], tools=[noop])
    with warnings.catch_warnings():
        warnings.simplefilter('error', CacheBustWarning)
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_switch_back_within_ttl_uses_preserved_mark() -> None:
    """Marks are kept per model, so a collapse after switching back to an earlier model still warns.

    A reset-on-switch design would have discarded model A's mark at the switch to B, so the return
    to A would compare against nothing and stay silent. The warning proves the mark survived.
    """
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='anthropic'),
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), provider_name='openai'),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100), provider_name='anthropic'),
    ]
    agent = _agent_from_responses(responses, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning, match='request 3'):
        result = await agent.run('hi')
    assert result.output == 'done'


async def test_expiry_gap_named_when_beyond_ttl() -> None:
    """A collapse after a gap longer than the assumed TTL names the gap, avoiding mis-attribution."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), timestamp=base),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100), timestamp=base + timedelta(seconds=400)),
    ]
    agent = _agent_from_responses(responses, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning, match='past the assumed') as record:
        await agent.run('hi')
    assert '400s earlier' in str(record[0].message)


async def test_small_gap_keeps_generic_expiry_hedge() -> None:
    """A collapse with a short inter-request gap keeps the generic TTL hedge, not a concrete gap."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    responses = [
        ModelResponse(parts=[ToolCallPart('noop', {})], usage=_usage(read=0, write=8000), timestamp=base),
        ModelResponse(parts=[TextPart('done')], usage=_usage(read=100), timestamp=base + timedelta(seconds=5)),
    ]
    agent = _agent_from_responses(responses, CacheStabilityMonitor())
    with pytest.warns(CacheBustWarning) as record:
        await agent.run('hi')
    message = str(record[0].message)
    assert 'e.g. a gap longer than the cache TTL' in message
    assert 'past the assumed' not in message
