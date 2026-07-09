"""Tests for the optional-`name` behavior shared by every managed-variable capability.

When a capability is constructed without an explicit `name`, its backing variable is derived from
the running agent's own `name` on first run-time use -- `<prefix><agent name>` -- rather than built
at construction (this pydantic-ai version has no construction-time agent hook). The per-capability
prefix wiring is covered in each capability's own module; this module covers the shared derivation:
that it reads `ctx.agent.name` in the run-time hooks, resolves and auto-creates against the derived
name, raises `UserError` when the agent has no name, and leaves the explicit-name path untouched.

The managed value is driven through each capability's code `default` here (a nameless capability's
backing `Variable` doesn't exist until a run derives its name, so there is nothing to `override()`
before the run); the remote-value path is covered per capability in their own modules.
"""

from __future__ import annotations

from typing import Any

import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, Variable, VariableConfig, VariablesConfig
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

import pydantic_ai_harness.logfire._managed_variable as managed_variable
from pydantic_ai_harness.logfire import (
    ManagedAgentSpec,
    ManagedPrompt,
    ManagedSettings,
    ManagedSettingsValue,
    ManagedToolDefinitions,
    ToolDefinitionOverride,
)

from ._helpers import advertised, capture_tools, get_weather, variables_provider

pytestmark = pytest.mark.anyio

DEFAULT = 'You are a helpful assistant.'


def instructions_seen(result_messages: list[ModelMessage]) -> list[str]:
    return [m.instructions for m in result_messages if isinstance(m, ModelRequest) and m.instructions is not None]


# --- derivation from the agent's own name -----------------------------------------------------


async def test_prompt_derives_variable_from_agent_name() -> None:
    capability = ManagedPrompt(default=DEFAULT)
    agent = Agent(TestModel(), name='weather_agent', capabilities=[capability])

    # Nothing is built until the first run derives the name from the agent.
    assert capability._built_variable is None
    assert capability._name_omitted

    result = await agent.run('hello')

    assert capability._variable.name == 'prompt__weather_agent'
    assert instructions_seen(result.all_messages()) == [DEFAULT]


async def test_settings_derives_variable_from_agent_name() -> None:
    seen: list[ModelSettings | None] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart('from-function')])

    capability = ManagedSettings(default=ManagedSettingsValue(temperature=0.7))
    agent = Agent(FunctionModel(respond), name='checkout_assistant', capabilities=[capability])

    await agent.run('hello')

    assert capability._variable.name == 'agent__checkout_assistant'
    assert seen == [{'temperature': 0.7}]


async def test_tool_definitions_derives_variable_from_agent_name() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedToolDefinitions(default=[ToolDefinitionOverride(name='get_weather', description='Patched.')])
    agent = Agent(capture_tools(seen), name='weather_agent', tools=[get_weather], capabilities=[capability])

    await agent.run('hello')

    assert capability._variable.name == 'tool_definitions__weather_agent'
    assert advertised(seen) == {'get_weather': 'Patched.'}


async def test_agent_spec_derives_variable_from_agent_name() -> None:
    capability = ManagedAgentSpec(default=AgentSpec(instructions='Be concise.'))
    agent = Agent(TestModel(), name='checkout_assistant', capabilities=[capability])

    result = await agent.run('hello')

    assert capability._variable.name == 'agentspec__checkout_assistant'
    assert 'Be concise.' in instructions_seen(result.all_messages())


# --- `ctx.agent` is populated in the run-time hooks -------------------------------------------


async def test_ctx_agent_available_in_resolution_hook() -> None:
    seen_agents: list[object] = []

    # A bare lambda so `AgentDepsT` isn't pinned (the callable captures `ctx.agent` and returns no key).
    capability = ManagedPrompt(default=DEFAULT, targeting_key=lambda ctx: seen_agents.append(ctx.agent) or None)
    agent = Agent(TestModel(), name='probe_agent', capabilities=[capability])

    await agent.run('hello')

    # The targeting callable runs inside `_resolve`, where the run-time `RunContext` carries the agent
    # -- which is exactly what the nameless capability reads `.name` off of.
    assert seen_agents == [agent]
    assert capability._variable.name == 'prompt__probe_agent'


# --- no agent name to derive from -------------------------------------------------------------


async def test_nameless_without_agent_name_raises() -> None:
    capability = ManagedPrompt(default=DEFAULT)
    # No `name=` on the agent and `infer_name=False`, so the agent has no name to derive from.
    agent = Agent(TestModel(), capabilities=[capability])

    with pytest.raises(UserError, match='without an explicit `name`'):
        await agent.run('hello', infer_name=False)


# --- nameless model override (via `before_model_request` on an agent that has a model) ---------


async def test_nameless_model_override_via_before_model_request() -> None:
    # A nameless `ManagedSettings` can't source the model at run setup, but it can still override the
    # model per request on an agent that has a code-side model.
    def respond(  # pragma: no cover - must NOT be reached; `model='test'` swaps in `TestModel`
        _messages: list[ModelMessage], _info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart('from-function')])

    capability = ManagedSettings(default=ManagedSettingsValue(model='test'))
    agent = Agent(FunctionModel(respond), name='model_swap', capabilities=[capability])

    result = await agent.run('hello')

    # `model='test'` -> served by `TestModel`, not the code-side `FunctionModel`.
    assert result.output == 'success (no tool calls)'
    assert capability._variable.name == 'agent__model_swap'


def test_nameless_get_model_returns_none() -> None:
    # Documented edge: with no `RunContext` at run setup, a nameless capability can't derive its
    # variable name, so it sources no model. (Pass an explicit `name` to source the model.)
    assert ManagedSettings().get_model() is None
    assert ManagedAgentSpec().get_model() is None


# --- explicit-name path is unchanged ----------------------------------------------------------


def test_explicit_name_builds_eagerly() -> None:
    # Regression: an explicit name still builds the variable at construction (no agent needed),
    # exactly as before, and does not take the deferred path.
    capability = ManagedPrompt('support_agent', default=DEFAULT)
    assert capability._variable.name == 'prompt__support_agent'
    assert capability._deferred is None
    assert not capability._name_omitted


async def test_nameless_auto_create_uses_derived_name(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    # Auto-create targets the run-time-derived variable name, so a nameless capability creates
    # `<prefix><agent name>` just like an explicit name would.
    managed_variable._reset_auto_create_guard()
    created: list[str] = []

    def record_spawn(variable: Variable[Any]) -> None:
        created.append(variable.name)

    monkeypatch.setattr(managed_variable, '_spawn_create', record_spawn)

    # A provider that knows some *other* variable, but not the one this capability will derive.
    config = VariablesConfig(
        variables={
            'agent__known': VariableConfig(
                name='agent__known',
                labels={'production': LabeledValue(version=1, serialized_value='{}')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    capability = ManagedSettings()
    agent = Agent(TestModel(), name='autocreate', capabilities=[capability])
    with variables_provider(capfire, config):
        await agent.run('hello')

    # The provider has no `agent__autocreate`, so it is auto-created under the derived name.
    assert created == ['agent__autocreate']
