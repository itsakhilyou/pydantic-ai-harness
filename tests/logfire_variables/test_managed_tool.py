"""Tests for the `ManagedTool` capability (source package `pydantic_ai_harness.logfire`).

Shared fixtures (`anyio_backend`, Logfire configuration) live in `conftest.py` and shared helpers
(`capture_tools`, `get_weather`, `variables_provider`) in `_helpers.py`. The variable-naming
contract shared with the other managed-variable capabilities is covered in `test_managed_variable.py`;
this module focuses on `ManagedTool`'s own behavior -- owning a `Tool`, and resolving overrides into
advertised tool definitions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from pydantic import ValidationError
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_harness import ManagedTool, ManagedToolOverride
from pydantic_ai_harness.logfire import ManagedTool as ManagedToolFromPackage
from pydantic_ai_harness.logfire import ManagedToolOverride as ManagedToolOverrideFromPackage
from pydantic_ai_harness.logfire._managed_tool import _with_parameter_descriptions

from ._helpers import capture_tools, get_weather, variables_provider

pytestmark = pytest.mark.anyio


def _weather_agent(*capabilities: AbstractCapability[None]) -> Agent[None, str]:
    agent = Agent(TestModel(), capabilities=list(capabilities))
    agent.tool_plain(get_weather)
    return agent


# --- Construction / variable naming ---


def test_public_reexport() -> None:
    assert ManagedTool is ManagedToolFromPackage
    assert ManagedToolOverride is ManagedToolOverrideFromPackage


def test_tool_name_becomes_variable_name() -> None:
    capability = ManagedTool('get_weather')
    assert capability._variable.name == 'tool__get_weather'


def test_explicit_variable_name_overrides_tool_name() -> None:
    capability = ManagedTool('get_weather', variable='weather_config')
    assert capability._variable.name == 'tool__weather_config'


def test_accepts_prebuilt_variable() -> None:
    var = logfire.var(name='tool__prebuilt', type=ManagedToolOverride, default=ManagedToolOverride())
    capability = ManagedTool('get_weather', variable=var)
    assert capability._variable is var


def test_logfire_instance_with_prebuilt_variable_warns() -> None:
    var = logfire.var(name='tool__instance_conflict', type=ManagedToolOverride, default=ManagedToolOverride())
    with pytest.warns(UserWarning, match='is ignored when `variable` is a `Variable`') as caught:
        ManagedTool('get_weather', variable=var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)
    # Filename should point at this test module (the user's call site), not the library internals.
    assert caught[0].filename == __file__


# --- Owning a Tool ---


def test_name_arg_does_not_own_a_toolset() -> None:
    # Managing a tool registered elsewhere contributes no toolset of its own.
    assert ManagedTool('get_weather').get_toolset() is None


def test_tool_arg_derives_variable_name_and_owns_a_toolset() -> None:
    capability = ManagedTool(Tool(get_weather))
    # The variable name defaults to the owned tool's name...
    assert capability._variable.name == 'tool__get_weather'
    # ...and the capability provides the tool itself.
    assert capability.get_toolset() is not None


async def test_owned_tool_is_provided_and_managed() -> None:
    capability = ManagedTool(Tool(get_weather), variable='owned_tool')
    # No separately-registered tool: the capability both provides and manages `get_weather`.
    agent: Agent[None, str] = Agent(TestModel(), capabilities=[capability])

    seen: list[ToolDefinition] = []
    with capability._variable.override(ManagedToolOverride(name='weather_lookup', description='Look up weather.')):
        await agent.run('hi', model=capture_tools(seen))
        assert seen[0].name == 'weather_lookup'
        assert seen[0].description == 'Look up weather.'

        result = await agent.run('hi', model=TestModel(call_tools=['weather_lookup']))

    assert result.output == '{"weather_lookup":"sunny in a"}'


# --- Resolution into advertised tool definitions ---


async def test_default_override_leaves_tool_unchanged() -> None:
    seen: list[ToolDefinition] = []
    agent = _weather_agent(ManagedTool('get_weather', variable='default_tool'))

    await agent.run('hi', model=capture_tools(seen))

    assert [td.name for td in seen] == ['get_weather']
    assert seen[0].description is None


async def test_override_description_only() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedTool('get_weather', variable='desc_tool')
    agent = _weather_agent(capability)

    with capability._variable.override(ManagedToolOverride(description='Look up the weather.')):
        await agent.run('hi', model=capture_tools(seen))

    assert seen[0].name == 'get_weather'
    assert seen[0].description == 'Look up the weather.'


async def test_override_parameter_descriptions() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedTool('get_weather', variable='paramdesc_tool')
    agent = _weather_agent(capability)

    with capability._variable.override(ManagedToolOverride(parameter_descriptions={'city': 'City name'})):
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
    capability = ManagedTool('get_weather', variable='unknown_param_tool')
    agent = _weather_agent(capability)

    await agent.run('hi', model=capture_tools(seen))
    code_schema = seen[0].parameters_json_schema

    seen.clear()
    with capability._variable.override(ManagedToolOverride(parameter_descriptions={'nonexistent': 'x'})):
        await agent.run('hi', model=capture_tools(seen))

    # A description keyed by a parameter that doesn't exist is a no-op; the schema is unchanged.
    assert seen[0].parameters_json_schema == code_schema


async def test_override_leaves_parameter_schema_structure_untouched() -> None:
    # Names, types, and required fields are intentionally not overridable, so the advertised schema
    # can never drift from the tool's argument validator. Only descriptions can be tuned.
    seen: list[ToolDefinition] = []
    capability = ManagedTool('get_weather', variable='schema_tool')
    agent = _weather_agent(capability)

    await agent.run('hi', model=capture_tools(seen))
    code_schema = seen[0].parameters_json_schema

    seen.clear()
    with capability._variable.override(ManagedToolOverride(name='weather_lookup', description='changed')):
        await agent.run('hi', model=capture_tools(seen))

    assert seen[0].parameters_json_schema == code_schema


def test_with_parameter_descriptions_patches_only_named_param() -> None:
    schema = {'type': 'object', 'properties': {'a': {'type': 'string'}, 'b': {'type': 'integer'}}}

    result = _with_parameter_descriptions(schema, {'a': 'desc a'})

    assert result is not schema
    assert result['properties']['a'] == {'type': 'string', 'description': 'desc a'}
    assert result['properties']['b'] == {'type': 'integer'}


def test_with_parameter_descriptions_without_properties_is_noop() -> None:
    schema: dict[str, object] = {'type': 'object'}
    assert _with_parameter_descriptions(schema, {'a': 'b'}) is schema


def test_override_name_rejects_empty_string() -> None:
    # Defends against a UI-side rollout of `{"name": ""}` producing an unnamed tool that
    # crashes the provider call with an opaque schema error. `None` (keep the original) is fine.
    with pytest.raises(ValidationError, match='at least 1 character'):
        ManagedToolOverride(name='')


async def test_override_name_advertises_and_routes_call() -> None:
    capability = ManagedTool('get_weather', variable='name_tool')
    agent = _weather_agent(capability)

    seen: list[ToolDefinition] = []
    with capability._variable.override(ManagedToolOverride(name='weather_lookup')):
        # The renamed tool is what the model sees and calls...
        await agent.run('hi', model=capture_tools(seen))
        assert [td.name for td in seen] == ['weather_lookup']

        # ...and calling it routes back to the original `get_weather` implementation.
        result = await agent.run('hi', model=TestModel(call_tools=['weather_lookup']))

    assert result.output == '{"weather_lookup":"sunny in a"}'


async def test_override_name_and_description() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedTool('get_weather', variable='both_tool')
    agent = _weather_agent(capability)
    override = ManagedToolOverride(name='weather_lookup', description='Look up weather.')

    with capability._variable.override(override):
        await agent.run('hi', model=capture_tools(seen))

    assert seen[0].name == 'weather_lookup'
    assert seen[0].description == 'Look up weather.'


async def test_managing_absent_tool_is_noop() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedTool('nonexistent')
    agent = _weather_agent(capability)

    with capability._variable.override(ManagedToolOverride(name='renamed', description='nope')):
        await agent.run('hi', model=capture_tools(seen))

    # The override targets a tool the agent doesn't have, so every real tool is left untouched.
    assert [td.name for td in seen] == ['get_weather']
    assert seen[0].description is None


async def test_rename_collision_raises() -> None:
    capability = ManagedTool('get_weather', variable='collision_tool')
    agent = _weather_agent(capability)

    @agent.tool_plain
    def get_forecast() -> str:
        return 'forecast'  # pragma: no cover -- exists only to create a name collision; never called

    with capability._variable.override(ManagedToolOverride(name='get_forecast')):
        with pytest.raises(UserError, match='duplicate tool name'):
            await agent.run('hi', model=capture_tools([]))


# --- Resolution lifecycle ---


async def test_resolved_property_exposes_active_resolution() -> None:
    capability = ManagedTool('get_weather', variable='exposed_tool')
    agent = _weather_agent(capability)
    captured: list[str | None] = []

    @agent.tool_plain
    def grab() -> str:
        resolved = capability.resolved
        captured.append(resolved.value.description if resolved is not None else None)
        return 'ok'

    with capability._variable.override(ManagedToolOverride(description='live')):
        await agent.run('hi', model=TestModel(call_tools=['grab']))

    assert captured == ['live']
    # The resolution is cleared once the run completes.
    assert capability.resolved is None


def test_resolved_is_none_outside_run() -> None:
    capability = ManagedTool('outside_tool')
    assert capability.resolved is None


def test_current_overrides_empty_outside_run() -> None:
    # Outside a run nothing is resolved, so the wrapper is handed an empty override map.
    assert ManagedTool('outside_tool2')._current_overrides() == {}


async def test_resolved_once_per_run() -> None:
    capability = ManagedTool('once_tool')
    agent = _weather_agent(capability)

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        # TestModel issues one request to call the tool and another for the final output,
        # so tools are listed more than once, but the variable is resolved exactly once.
        await agent.run('hi')

    assert spy.call_count == 1


async def test_label_and_callable_targeting_and_attributes() -> None:
    capability = ManagedTool(
        'targeting_tool',
        label='production',
        targeting_key=lambda ctx: f'run:{ctx.run_step}',
        attributes=lambda ctx: {'tier': 'enterprise'},
    )
    agent = _weather_agent(capability)

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hi')

    spy.assert_called_once_with(
        targeting_key='run:0',
        attributes={'tier': 'enterprise'},
        label='production',
    )


async def test_static_targeting_and_attributes() -> None:
    capability = ManagedTool('static_tool', targeting_key='tenant-123', attributes={'tier': 'free'})
    agent = _weather_agent(capability)

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hi')

    spy.assert_called_once_with(targeting_key='tenant-123', attributes={'tier': 'free'}, label=None)


async def test_provider_backed_resolution_uses_remote_value(capfire: CaptureLogfire) -> None:
    config = VariablesConfig(
        variables={
            'tool__remote_tool': VariableConfig(
                name='tool__remote_tool',
                labels={
                    'production': LabeledValue(
                        version=2,
                        serialized_value='{"description": "The PRODUCTION weather tool."}',
                    )
                },
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    seen: list[ToolDefinition] = []
    with variables_provider(capfire, config):
        capability = ManagedTool('get_weather', variable='remote_tool', label='production')
        agent = _weather_agent(capability)
        await agent.run('hi', model=capture_tools(seen))

    assert seen[0].description == 'The PRODUCTION weather tool.'

    spans = capfire.exporter.exported_spans_as_dict()
    resolution = next(s for s in spans if s['attributes'].get('logfire.msg') == 'Resolve variable tool__remote_tool')
    assert resolution['attributes']['reason'] == 'resolved'
    assert resolution['attributes']['label'] == 'production'


async def test_provider_backed_override_serializes_both_fields(capfire: CaptureLogfire) -> None:
    # Exercises the real JSON deserialize path for both fields (name and description), not just the
    # in-memory `.override()` path other tests use.
    config = VariablesConfig(
        variables={
            'tool__full_remote': VariableConfig(
                name='tool__full_remote',
                labels={
                    'production': LabeledValue(
                        version=1,
                        serialized_value='{"name": "weather_v2", "description": "Get weather"}',
                    )
                },
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    seen: list[ToolDefinition] = []
    with variables_provider(capfire, config):
        capability = ManagedTool('get_weather', variable='full_remote', label='production')
        agent = _weather_agent(capability)
        await agent.run('hi', model=capture_tools(seen))
        assert seen[0].name == 'weather_v2'
        assert seen[0].description == 'Get weather'

        # The remote rename still routes the model's call back to the original implementation.
        result = await agent.run('hi', model=TestModel(call_tools=['weather_v2']))

    assert result.output == '{"weather_v2":"sunny in a"}'


async def test_concurrent_runs_are_isolated() -> None:
    # One shared capability instance, two interleaved runs with different overrides: the per-run
    # context variable must keep each run's resolution separate.
    capability = ManagedTool('get_weather', variable='concurrency_tool')
    agent = _weather_agent(capability)

    async def run_with(description: str, bucket: list[str | None]) -> None:
        with capability._variable.override(ManagedToolOverride(description=description)):
            await asyncio.sleep(0)  # yield so the two runs interleave inside their override blocks
            seen: list[ToolDefinition] = []
            await agent.run('hi', model=capture_tools(seen))
            bucket.append(seen[0].description)

    first: list[str | None] = []
    second: list[str | None] = []
    await asyncio.gather(run_with('first', first), run_with('second', second))

    assert first == ['first']
    assert second == ['second']


async def test_targeting_key_callable_can_return_none() -> None:
    @dataclass
    class Deps:
        user_id: str | None

    capability: ManagedTool[Deps] = ManagedTool('callable_none_tool', targeting_key=lambda ctx: ctx.deps.user_id)
    agent: Agent[Deps, str] = Agent(TestModel(), deps_type=Deps, capabilities=[capability])

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hi', deps=Deps(user_id=None))

    spy.assert_called_once_with(targeting_key=None, attributes=None, label=None)
