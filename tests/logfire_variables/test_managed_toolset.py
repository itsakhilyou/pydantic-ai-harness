"""Tests for the `ManagedToolset` capability (source package `pydantic_ai_harness.logfire`).

Style mirrors `test_managed_tool.py`: module-level `pytestmark = pytest.mark.anyio`, an
`anyio_backend` fixture, a `FunctionModel` that captures the advertised tools, and a unique
variable name per test. `ManagedToolset` manages a whole group of tools from one variable holding
a `dict[str, ManagedToolOverride]`.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai import Agent, Tool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_harness import ManagedToolOverride, ManagedToolset
from pydantic_ai_harness.logfire import ManagedToolset as ManagedToolsetFromPackage

pytestmark = pytest.mark.anyio

_ToolOverrides = dict[str, ManagedToolOverride]


def get_weather(city: str) -> str:
    return f'sunny in {city}'


def get_forecast(city: str) -> str:
    return f'forecast for {city}'


@pytest.fixture(autouse=True, scope='module')
def _configure_logfire() -> None:
    logfire.configure(send_to_logfire=False, console=False)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _capture_tools(seen: list[ToolDefinition]) -> FunctionModel:
    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(info.function_tools)
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _weather_toolset(variable: str, default: _ToolOverrides | None = None) -> ManagedToolset[None]:
    return ManagedToolset(variable, tools=[Tool(get_weather), Tool(get_forecast)], default=default)


@contextmanager
def _variables_provider_configured(capfire: CaptureLogfire, variables_config: VariablesConfig) -> Generator[None]:
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(config=variables_config),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    try:
        yield
    finally:
        logfire.configure(send_to_logfire=False, console=False)


def advertised(seen: list[ToolDefinition]) -> dict[str, str | None]:
    return {td.name: td.description for td in seen}


# --- Construction / variable naming ---


def test_public_reexport() -> None:
    assert ManagedToolset is ManagedToolsetFromPackage


def test_name_becomes_variable_name() -> None:
    assert ManagedToolset('support')._variable.name == 'toolset__support'


def test_hyphenated_name_is_normalized() -> None:
    assert ManagedToolset('support-tools')._variable.name == 'toolset__support_tools'


def test_prefix_in_name_warns_and_is_stripped() -> None:
    with pytest.warns(UserWarning, match='added automatically'):
        capability = ManagedToolset('toolset__already_prefixed')
    assert capability._variable.name == 'toolset__already_prefixed'


def test_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match='invalid variable name'):
        ManagedToolset('has spaces')


def test_duplicate_name_is_allowed() -> None:
    first = ManagedToolset('shared')
    second = ManagedToolset('shared')
    assert first._variable.name == second._variable.name == 'toolset__shared'


def test_explicit_logfire_instance_is_used() -> None:
    capability = ManagedToolset('with_instance', logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)
    assert capability._variable.name == 'toolset__with_instance'


def test_accepts_prebuilt_variable() -> None:
    var = logfire.var(name='toolset__prebuilt', type=_ToolOverrides, default={})
    capability: ManagedToolset[None] = ManagedToolset(var)
    assert capability._variable is var


def test_logfire_instance_with_prebuilt_variable_warns() -> None:
    var = logfire.var(name='toolset__instance_conflict', type=_ToolOverrides, default={})
    with pytest.warns(UserWarning, match='is ignored when `name` is a `Variable`'):
        ManagedToolset(var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)


# --- Owning tools ---


def test_without_tools_owns_no_toolset() -> None:
    assert ManagedToolset('support').get_toolset() is None


def test_with_tools_owns_a_toolset() -> None:
    assert ManagedToolset('support', tools=[Tool(get_weather)]).get_toolset() is not None


# --- Resolution ---


async def test_default_empty_leaves_tools_unchanged() -> None:
    seen: list[ToolDefinition] = []
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[_weather_toolset('default_set')])

    await agent.run('hi', model=_capture_tools(seen))

    assert advertised(seen) == {'get_weather': None, 'get_forecast': None}


async def test_overrides_multiple_tools_from_one_variable() -> None:
    capability = _weather_toolset('multi_set')
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])
    overrides: _ToolOverrides = {
        'get_weather': ManagedToolOverride(name='weather_lookup', description='Look up weather.'),
        'get_forecast': ManagedToolOverride(description='Multi-day forecast.'),
    }

    seen: list[ToolDefinition] = []
    with capability._variable.override(overrides):
        await agent.run('hi', model=_capture_tools(seen))
        assert advertised(seen) == {'weather_lookup': 'Look up weather.', 'get_forecast': 'Multi-day forecast.'}

        # The renamed tool still routes back to its original implementation.
        result = await agent.run('hi', model=TestModel(call_tools=['weather_lookup']))

    assert result.output == '{"weather_lookup":"sunny in a"}'


async def test_manages_tools_registered_elsewhere() -> None:
    # No `tools=`: the toolset manages whatever the agent already has, by name.
    capability: ManagedToolset[None] = ManagedToolset('external_set')
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])

    @agent.tool_plain
    def search(query: str) -> str:
        return f'results for {query}'  # pragma: no cover -- only its advertised definition is asserted

    seen: list[ToolDefinition] = []
    with capability._variable.override({'search': ManagedToolOverride(description='Web search.')}):
        await agent.run('hi', model=_capture_tools(seen))

    assert seen[0].name == 'search'
    assert seen[0].description == 'Web search.'


async def test_unknown_tool_name_is_ignored() -> None:
    capability = _weather_toolset('unknown_set')
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])

    seen: list[ToolDefinition] = []
    with capability._variable.override({'nonexistent': ManagedToolOverride(description='nope')}):
        await agent.run('hi', model=_capture_tools(seen))

    assert advertised(seen) == {'get_weather': None, 'get_forecast': None}


async def test_collision_across_tools_raises() -> None:
    capability = _weather_toolset('collision_set')
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])

    # Renaming get_weather onto the existing get_forecast name collides.
    with capability._variable.override({'get_weather': ManagedToolOverride(name='get_forecast')}):
        with pytest.raises(UserError, match='duplicate tool name'):
            await agent.run('hi', model=_capture_tools([]))


async def test_swapping_names_is_allowed() -> None:
    capability = _weather_toolset('swap_set')
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])
    overrides: _ToolOverrides = {
        'get_weather': ManagedToolOverride(name='get_forecast'),
        'get_forecast': ManagedToolOverride(name='get_weather'),
    }

    seen: list[ToolDefinition] = []
    with capability._variable.override(overrides):
        await agent.run('hi', model=_capture_tools(seen))
        # Names are swapped, with no spurious collision...
        assert set(advertised(seen)) == {'get_weather', 'get_forecast'}

        # ...and a call to the swapped name routes to the right original implementation.
        result = await agent.run('hi', model=TestModel(call_tools=['get_weather']))

    # `get_weather` now points at the original `get_forecast` function.
    assert result.output == '{"get_weather":"forecast for a"}'


# --- Lifecycle ---


def test_current_overrides_empty_outside_run() -> None:
    assert ManagedToolset('outside_set')._current_overrides() == {}


def test_resolved_is_none_outside_run() -> None:
    assert ManagedToolset('outside_set2').resolved is None


async def test_resolved_property_exposes_active_resolution() -> None:
    capability = _weather_toolset('exposed_set')
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])
    captured: list[str | None] = []

    @agent.tool_plain
    def grab() -> str:
        resolved = capability.resolved
        weather = resolved.value.get('get_weather') if resolved is not None else None
        captured.append(weather.description if weather is not None else None)
        return 'ok'

    with capability._variable.override({'get_weather': ManagedToolOverride(description='live')}):
        await agent.run('hi', model=TestModel(call_tools=['grab']))

    assert captured == ['live']
    assert capability.resolved is None


async def test_resolved_once_per_run() -> None:
    from unittest.mock import patch

    capability = _weather_toolset('once_set')
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hi')

    assert spy.call_count == 1


async def test_callable_targeting_and_attributes() -> None:
    from unittest.mock import patch

    capability: ManagedToolset[None] = ManagedToolset(
        'targeting_set',
        label='production',
        targeting_key=lambda ctx: f'run:{ctx.run_step}',
        attributes=lambda ctx: {'tier': 'enterprise'},
    )
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hi')

    spy.assert_called_once_with(targeting_key='run:0', attributes={'tier': 'enterprise'}, label='production')


async def test_static_targeting_and_attributes() -> None:
    from unittest.mock import patch

    capability: ManagedToolset[None] = ManagedToolset(
        'static_set', targeting_key='tenant-1', attributes={'tier': 'free'}
    )
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hi')

    spy.assert_called_once_with(targeting_key='tenant-1', attributes={'tier': 'free'}, label=None)


async def test_provider_backed_resolution(capfire: CaptureLogfire) -> None:
    config = VariablesConfig(
        variables={
            'toolset__remote_set': VariableConfig(
                name='toolset__remote_set',
                labels={
                    'production': LabeledValue(
                        version=1,
                        serialized_value='{"get_weather": {"name": "weather_v2", "description": "Remote weather."}}',
                    )
                },
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    seen: list[ToolDefinition] = []
    with _variables_provider_configured(capfire, config):
        capability = _weather_toolset('remote_set')
        capability.label = 'production'
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])
        await agent.run('hi', model=_capture_tools(seen))

    assert advertised(seen) == {'weather_v2': 'Remote weather.', 'get_forecast': None}
