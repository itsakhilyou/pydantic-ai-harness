"""Tests for the `ManagedToolDefinitions` capability (source package `pydantic_ai_harness.logfire`).

Shared fixtures (`anyio_backend`, Logfire configuration) live in `conftest.py` and shared helpers
(`capture_tools`, `get_weather`, `get_forecast`, `variables_provider`) in `_helpers.py`. The
variable-naming contract shared with the other managed-variable capabilities is covered in
`test_managed_variable.py`; this module focuses on `ManagedToolDefinitions`' own behavior -- resolving
a list of overrides into the agent's advertised tool definitions and routing renamed calls back to
the original implementations.
"""

from __future__ import annotations

import asyncio

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_harness import ManagedToolDefinitions
from pydantic_ai_harness.logfire._managed_tool_definitions import (
    ToolDefinitionOverride,
    _with_parameter_descriptions,
)

from ._helpers import advertised, capture_tools, get_forecast, get_weather, variables_provider

pytestmark = pytest.mark.anyio

_Overrides = list[ToolDefinitionOverride]


def _weather_agent(*capabilities: ManagedToolDefinitions[object]) -> Agent[object, str]:
    agent: Agent[object, str] = Agent(TestModel(), capabilities=list(capabilities))
    agent.tool_plain(get_weather)
    return agent


def _weather_forecast_agent(*capabilities: ManagedToolDefinitions[object]) -> Agent[object, str]:
    agent: Agent[object, str] = Agent(TestModel(), capabilities=list(capabilities))
    agent.tool_plain(get_weather)
    agent.tool_plain(get_forecast)
    return agent


# --- Construction / variable naming ---


def test_name_becomes_variable_name() -> None:
    assert ManagedToolDefinitions('checkout_assistant')._variable.name == 'tool_definitions__checkout_assistant'


def test_accepts_prebuilt_variable() -> None:
    var = logfire.var(name='tool_definitions__prebuilt', type=_Overrides, default=[])
    capability: ManagedToolDefinitions[None] = ManagedToolDefinitions(var)
    assert capability._variable is var


def test_logfire_instance_with_prebuilt_variable_warns() -> None:
    var = logfire.var(name='tool_definitions__instance_conflict', type=_Overrides, default=[])
    with pytest.warns(UserWarning, match='is ignored when `name` is a `Variable`'):
        ManagedToolDefinitions(var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)


def test_prefix_is_stripped_with_warning() -> None:
    with pytest.warns(UserWarning, match='prefix is added automatically'):
        capability = ManagedToolDefinitions('tool_definitions__foo')
    assert capability._variable.name == 'tool_definitions__foo'


def test_override_name_rejects_empty_string() -> None:
    with pytest.raises(ValidationError, match='at least 1 character'):
        ToolDefinitionOverride(name='')


def test_override_new_name_rejects_empty_string() -> None:
    with pytest.raises(ValidationError, match='at least 1 character'):
        ToolDefinitionOverride(name='get_weather', new_name='')


# --- Resolution into advertised tool definitions ---


async def test_default_leaves_tool_definitions_unchanged() -> None:
    seen: list[ToolDefinition] = []
    agent = _weather_forecast_agent(ManagedToolDefinitions('default_defs'))

    await agent.run('hi', model=capture_tools(seen))

    # No overrides published: the tools are advertised exactly as defined in code.
    assert advertised(seen) == {'get_weather': None, 'get_forecast': None}


async def test_override_description_only() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('desc_defs')
    agent = _weather_agent(capability)

    with capability._variable.override(
        [ToolDefinitionOverride(name='get_weather', description='Look up the weather.')]
    ):
        await agent.run('hi', model=capture_tools(seen))

    assert seen[0].name == 'get_weather'
    assert seen[0].description == 'Look up the weather.'


async def test_override_parameter_descriptions() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('paramdesc_defs')
    agent = _weather_agent(capability)

    override = ToolDefinitionOverride(name='get_weather', parameter_descriptions={'city': 'City name'})
    with capability._variable.override([override]):
        await agent.run('hi', model=capture_tools(seen))
        # Only the parameter's description changes; its type and the schema's structure are untouched.
        city = seen[0].parameters_json_schema['properties']['city']
        assert city['description'] == 'City name'
        assert city['type'] == 'string'
        assert seen[0].parameters_json_schema['required'] == ['city']

        # The argument validator is unaffected, so the tool still executes normally.
        result = await agent.run('hi', model=TestModel(call_tools=['get_weather']))

    assert result.output == '{"get_weather":"sunny in a"}'


async def test_parameter_descriptions_ignores_unknown_params() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('unknown_param_defs')
    agent = _weather_agent(capability)

    await agent.run('hi', model=capture_tools(seen))
    code_schema = seen[0].parameters_json_schema

    seen.clear()
    override = ToolDefinitionOverride(name='get_weather', parameter_descriptions={'nonexistent': 'x'})
    with capability._variable.override([override]):
        await agent.run('hi', model=capture_tools(seen))

    # A description keyed by a parameter that doesn't exist is a no-op; the schema is unchanged.
    assert seen[0].parameters_json_schema == code_schema


async def test_override_leaves_parameter_schema_structure_untouched() -> None:
    # Names, types, and required fields are intentionally not overridable, so the advertised schema
    # can never drift from the tool's argument validator. Only descriptions can be tuned.
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('schema_defs')
    agent = _weather_agent(capability)

    await agent.run('hi', model=capture_tools(seen))
    code_schema = seen[0].parameters_json_schema

    seen.clear()
    override = ToolDefinitionOverride(name='get_weather', new_name='weather_lookup', description='changed')
    with capability._variable.override([override]):
        await agent.run('hi', model=capture_tools(seen))

    assert seen[0].parameters_json_schema == code_schema


async def test_overrides_multiple_tools_from_one_variable() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('multi_defs')
    agent = _weather_forecast_agent(capability)
    overrides: _Overrides = [
        ToolDefinitionOverride(name='get_weather', new_name='weather_lookup', description='Look up weather.'),
        ToolDefinitionOverride(name='get_forecast', description='Multi-day forecast.'),
    ]

    with capability._variable.override(overrides):
        await agent.run('hi', model=capture_tools(seen))

    assert advertised(seen) == {'weather_lookup': 'Look up weather.', 'get_forecast': 'Multi-day forecast.'}


# --- Renames ---


async def test_rename_advertises_and_routes_call_to_original() -> None:
    capability = ManagedToolDefinitions('rename_defs')
    agent = Agent(TestModel(), capabilities=[capability])

    seen_tool_names: list[str] = []

    @agent.tool
    def get_weather(ctx: RunContext[object], city: str) -> str:
        # The renamed tool routes back here, and the run context still carries the original name.
        seen_tool_names.append(ctx.tool_name or '')
        return f'sunny in {city}'

    seen: list[ToolDefinition] = []
    with capability._variable.override([ToolDefinitionOverride(name='get_weather', new_name='lookup_weather')]):
        # The renamed tool is what the model sees...
        await agent.run('hi', model=capture_tools(seen))
        assert [td.name for td in seen] == ['lookup_weather']

        # ...and calling it routes back to the original `get_weather` implementation.
        result = await agent.run('hi', model=TestModel(call_tools=['lookup_weather']))

    assert result.output == '{"lookup_weather":"sunny in a"}'
    # `ctx.tool_name` inside the tool is the original code-side name, not the model-facing rename.
    assert seen_tool_names == ['get_weather']


async def test_rename_collision_degrades_to_original_name_with_warning() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('collision_defs')
    agent = _weather_forecast_agent(capability)

    # Renaming get_weather onto the existing get_forecast name collides; a bad managed value must
    # never break a run, so the rename is dropped (the description patch still applies) + warning.
    override = ToolDefinitionOverride(name='get_weather', new_name='get_forecast', description='patched')
    with capability._variable.override([override]):
        with pytest.warns(UserWarning, match='keeping the original name'):
            await agent.run('hi', model=capture_tools(seen))

    assert advertised(seen) == {'get_weather': 'patched', 'get_forecast': None}


async def test_dropped_rename_does_not_misroute_calls_to_the_colliding_name() -> None:
    """After a dropped rename A->B (B exists), a model call to B must reach the real B, not A."""
    capability = ManagedToolDefinitions('collision_routing_defs')

    called: list[str] = []

    def get_weather(city: str) -> str:  # pragma: no cover - must NOT be reached; the assert proves no misroute
        called.append('get_weather')
        return 'sunny'

    def get_forecast(city: str) -> str:
        called.append('get_forecast')
        return 'rainy tomorrow'

    agent = Agent(TestModel(call_tools=['get_forecast']), tools=[get_weather, get_forecast], capabilities=[capability])

    with capability._variable.override([ToolDefinitionOverride(name='get_weather', new_name='get_forecast')]):
        with pytest.warns(UserWarning, match='keeping the original name'):
            await agent.run('hi')

    assert called == ['get_forecast']


# --- Drift / duplicates ---


async def test_override_for_nonexistent_tool_is_inert() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('nonexistent_defs')
    agent = _weather_agent(capability)

    override = ToolDefinitionOverride(name='removed_tool', new_name='renamed', description='nope')
    with capability._variable.override([override]):
        await agent.run('hi', model=capture_tools(seen))

    # The override targets a tool the agent doesn't have, so every real tool is left untouched.
    assert advertised(seen) == {'get_weather': None}


async def test_duplicate_entries_last_wins_with_warning() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions('duplicate_defs')
    agent = _weather_agent(capability)
    overrides: _Overrides = [
        ToolDefinitionOverride(name='get_weather', description='first'),
        ToolDefinitionOverride(name='get_weather', description='second'),
    ]

    with capability._variable.override(overrides):
        with pytest.warns(UserWarning, match="target 'get_weather'"):
            await agent.run('hi', model=capture_tools(seen))

    # The later entry wins.
    assert seen[0].description == 'second'


# --- Lifecycle ---


def test_current_overrides_empty_outside_run() -> None:
    assert ManagedToolDefinitions('outside_defs')._current_overrides() == {}


def test_resolved_is_none_outside_run() -> None:
    assert ManagedToolDefinitions('outside_defs2').resolved is None


async def test_concurrent_runs_are_isolated() -> None:
    # One shared capability instance, two interleaved runs with different overrides: the per-run
    # context variable must keep each run's resolution separate.
    capability = ManagedToolDefinitions('concurrency_defs')
    agent = _weather_agent(capability)

    async def run_with(description: str, bucket: list[str | None]) -> None:
        with capability._variable.override([ToolDefinitionOverride(name='get_weather', description=description)]):
            await asyncio.sleep(0)  # yield so the two runs interleave inside their override blocks
            seen: list[ToolDefinition] = []
            await agent.run('hi', model=capture_tools(seen))
            bucket.append(seen[0].description)

    first: list[str | None] = []
    second: list[str | None] = []
    await asyncio.gather(run_with('first', first), run_with('second', second))

    assert first == ['first']
    assert second == ['second']


# --- Helpers ---


def test_with_parameter_descriptions_patches_only_named_param() -> None:
    schema = {'type': 'object', 'properties': {'a': {'type': 'string'}, 'b': {'type': 'integer'}}}

    result = _with_parameter_descriptions(schema, {'a': 'desc a'})

    assert result is not schema
    assert result['properties']['a'] == {'type': 'string', 'description': 'desc a'}
    assert result['properties']['b'] == {'type': 'integer'}


def test_with_parameter_descriptions_without_properties_is_noop() -> None:
    schema: dict[str, object] = {'type': 'object'}
    assert _with_parameter_descriptions(schema, {'a': 'b'}) is schema


# --- Provider-backed resolution ---


async def test_provider_backed_resolution_uses_remote_value(capfire: CaptureLogfire) -> None:
    # Exercises the real JSON deserialize path for the list payload, not just the in-memory
    # `.override()` path other tests use.
    config = VariablesConfig(
        variables={
            'tool_definitions__remote_defs': VariableConfig(
                name='tool_definitions__remote_defs',
                labels={
                    'production': LabeledValue(
                        version=1,
                        serialized_value='[{"name": "get_weather", "new_name": "weather_v2", "description": "Get weather"}]',
                    )
                },
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    seen: list[ToolDefinition] = []
    with variables_provider(capfire, config):
        capability = ManagedToolDefinitions('remote_defs', label='production')
        agent = _weather_agent(capability)
        await agent.run('hi', model=capture_tools(seen))
        assert seen[0].name == 'weather_v2'
        assert seen[0].description == 'Get weather'

        # The remote rename still routes the model's call back to the original implementation.
        result = await agent.run('hi', model=TestModel(call_tools=['weather_v2']))

    assert result.output == '{"weather_v2":"sunny in a"}'

    spans = capfire.exporter.exported_spans_as_dict()
    resolution = next(
        s for s in spans if s['attributes'].get('logfire.msg') == 'Resolve variable tool_definitions__remote_defs'
    )
    assert resolution['attributes']['reason'] == 'resolved'
    assert resolution['attributes']['label'] == 'production'
