"""Tests for the `ManagedAgentSpec` capability and `ManagedAgent` sugar (source package
`pydantic_ai_harness.logfire`).

Shared fixtures (`anyio_backend`, Logfire configuration) live in `conftest.py`; the variable-naming
contract common to all managed-variable capabilities is covered in `test_managed_variable.py`. This
module focuses on `ManagedAgentSpec` resolving a whole [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec]
per run -- layering its instructions, model settings, model, and materialized capabilities onto the
code-defined agent, and degrading to the code-defined agent on a missing or invalid value.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import infer_model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from pydantic_ai_harness import ManagedAgent, ManagedAgentSpec
from pydantic_ai_harness.logfire import ManagedAgent as ManagedAgentFromPackage
from pydantic_ai_harness.logfire import ManagedAgentSpec as ManagedAgentSpecFromPackage
from pydantic_ai_harness.logfire import _managed_agent_spec
from pydantic_ai_harness.logfire._managed_agent_spec import _ResolvedAgentSpec

from ._helpers import variables_provider

pytestmark = pytest.mark.anyio


@dataclass
class ContributeInstructions(AbstractCapability[Any]):
    """A tiny custom capability that contributes a fixed instruction, for materialization tests."""

    text: str = 'from-custom-capability'

    def get_instructions(self) -> str:
        return self.text


def capture(seen: dict[str, Any]) -> FunctionModel:
    """A model that records the instructions, settings, and thinking it is shown, then ends the run."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen['instructions'] = [m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions]
        seen['settings'] = info.model_settings
        seen['thinking'] = info.model_request_parameters.thinking
        return ModelResponse(parts=[TextPart('from-function')])

    return FunctionModel(respond, profile=ModelProfile(supports_thinking=True))


# --- exports / construction -------------------------------------------------------------------


def test_reexported_from_top_level_and_package() -> None:
    # Both the top-level package and the `logfire` subpackage expose the public names.
    assert ManagedAgentSpec is ManagedAgentSpecFromPackage
    assert ManagedAgent is ManagedAgentFromPackage


def test_name_becomes_agentspec_variable_name() -> None:
    capability = ManagedAgentSpec('checkout_assistant')
    assert capability._variable.name == 'agentspec__checkout_assistant'


def test_default_not_required() -> None:
    # `default` is optional -- an empty `AgentSpec` means "nothing managed yet".
    capability = ManagedAgentSpec('no_default')
    assert capability._variable.default == AgentSpec()


def test_prebuilt_variable_prefix_warning() -> None:
    with pytest.warns(UserWarning, match="'agentspec__' prefix is added automatically"):
        capability = ManagedAgentSpec('agentspec__foo')
    assert capability._variable.name == 'agentspec__foo'


def test_prebuilt_variable_path() -> None:
    var = logfire.var(name='agentspec__prebuilt', type=AgentSpec, default=AgentSpec())
    capability = ManagedAgentSpec(var)
    assert capability._variable is var


def test_logfire_instance_with_prebuilt_variable_warns() -> None:
    var = logfire.var(name='agentspec__instance_conflict', type=AgentSpec, default=AgentSpec())
    with pytest.warns(UserWarning, match='is ignored when `name` is a `Variable`'):
        ManagedAgentSpec(var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)


# --- spec fields: instructions, settings, model -----------------------------------------------


async def test_absent_value_leaves_agent_exactly_as_coded() -> None:
    seen: dict[str, Any] = {}
    agent = Agent(
        capture(seen),
        instructions='code instructions',
        model_settings=ModelSettings(temperature=0.2, top_p=0.9),
        capabilities=[ManagedAgentSpec('unchanged')],
    )

    result = await agent.run('hello')

    # Empty spec -> the agent's own instructions, settings, and model are used verbatim.
    assert seen['instructions'] == ['code instructions']
    assert seen['settings'] == {'temperature': 0.2, 'top_p': 0.9}
    assert result.output == 'from-function'


async def test_spec_instructions_are_additive() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('additive')
    agent = Agent(capture(seen), instructions='code instructions', capabilities=[capability])

    with capability._variable.override(AgentSpec(instructions='spec instructions')):
        await agent.run('hello')

    # The spec's instructions add to the agent's own rather than replacing them.
    assert seen['instructions'] == ['code instructions\nspec instructions']


async def test_spec_instructions_list_form() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('instr_list')
    agent = Agent(capture(seen), capabilities=[capability])

    with capability._variable.override(AgentSpec(instructions=['first', 'second'])):
        await agent.run('hello')

    assert seen['instructions'] == ['first\nsecond']


async def test_spec_model_settings_merge_and_run_args_win() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('settings')
    agent = Agent(
        capture(seen),
        model_settings=ModelSettings(temperature=0.2, top_p=0.9),
        capabilities=[capability],
    )

    with capability._variable.override(AgentSpec(model_settings={'temperature': 0.7, 'max_tokens': 512})):
        await agent.run('hello')
        # Managed `temperature` wins over the agent default; `top_p` is kept; `max_tokens` is added.
        assert seen['settings'] == {'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 512}

        # Per-run `model_settings=` is merged last, so it beats the managed value.
        await agent.run('hello', model_settings=ModelSettings(temperature=0.1))
        assert seen['settings'] == {'temperature': 0.1, 'top_p': 0.9, 'max_tokens': 512}


async def test_managed_model_beats_constructor_model() -> None:
    # The spec's model is sourced at run setup via `get_model`, slotting in above the agent's
    # constructor model, so it replaces the code-side `FunctionModel`.
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('model_swap')
    agent = Agent(capture(seen), capabilities=[capability])

    with capability._variable.override(AgentSpec(model='test')):
        result = await agent.run('hello')

    # `model='test'` -> served by `TestModel`, not the code-side `FunctionModel`.
    assert result.output == 'success (no tool calls)'


async def test_model_less_agent_runs_via_get_model() -> None:
    # A fully model-less agent runs entirely off the spec's model, now that `get_model` sources it.
    capability = ManagedAgentSpec('model_less')
    agent = Agent(capabilities=[capability])

    with capability._variable.override(AgentSpec(model='test')):
        result = await agent.run('hello')

    assert result.output == 'success (no tool calls)'


async def test_run_model_beats_managed_model() -> None:
    # The wart fix: `get_model` slots the spec's model *below* a call-site `run(model=...)`, so the
    # run argument now wins (before the hook, `before_model_request` clobbered it every request).
    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('from-run-model')])

    capability = ManagedAgentSpec('run_model_wins')
    agent = Agent(capabilities=[capability])

    with capability._variable.override(AgentSpec(model='test')):
        result = await agent.run('hello', model=FunctionModel(respond))

    assert result.output == 'from-run-model'


async def test_spec_model_none_keeps_code_model() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('model_none')
    agent = Agent(capture(seen), capabilities=[capability])

    with capability._variable.override(AgentSpec(model=None)):
        result = await agent.run('hello')

    assert result.output == 'from-function'


async def test_callable_targeting_still_resolves_model() -> None:
    # Model selection happens before the run exists, so callable `targeting_key`/`attributes` can't
    # run; `get_model` uses the static path (callables -> None) rather than crashing.
    capability = ManagedAgentSpec(
        'callable_target',
        targeting_key=lambda ctx: 'user-key',
        attributes=lambda ctx: {'tier': 'gold'},
    )
    agent = Agent(capabilities=[capability])

    with capability._variable.override(AgentSpec(model='test')):
        result = await agent.run('hello')

    assert result.output == 'success (no tool calls)'


def test_get_model_reads_spec_model() -> None:
    # `get_model` is called on the construction-time capability (before any run), returning the
    # spec's model string, or `None` for an empty spec.
    capability = ManagedAgentSpec('get_model_unit')
    with capability._variable.override(AgentSpec(model='some-model')):
        assert capability.get_model() == 'some-model'
    with capability._variable.override(AgentSpec()):
        assert capability.get_model() is None


# --- model override fallback (older pydantic-ai without the `get_model` hook) ------------------
#
# On older pydantic-ai the framework doesn't source a capability model, so the spec's model is
# swapped per request in `_ResolvedAgentSpec.before_model_request` instead. We simulate that by
# forcing the module gate off and stubbing `get_model` to `None` (so the framework contributes no
# model and the agent falls back to its code-side model, which the per-request swap then overrides).


async def test_before_model_request_swaps_model_on_old_pydantic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_managed_agent_spec, '_FRAMEWORK_HAS_GET_MODEL', False)
    capability = ManagedAgentSpec('old_swap')
    monkeypatch.setattr(capability, 'get_model', lambda: None)
    agent = Agent(capture({}), capabilities=[capability])

    with capability._variable.override(AgentSpec(model='test')):
        result = await agent.run('hello')

    # The per-request swap replaces the code-side `FunctionModel` with `TestModel`.
    assert result.output == 'success (no tool calls)'


async def test_before_model_request_no_managed_model_keeps_code_on_old_pydantic_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_managed_agent_spec, '_FRAMEWORK_HAS_GET_MODEL', False)
    capability = ManagedAgentSpec('old_no_model')
    monkeypatch.setattr(capability, 'get_model', lambda: None)
    agent = Agent(capture({}), capabilities=[capability])

    with capability._variable.override(AgentSpec(model=None, instructions='x')):
        result = await agent.run('hello')

    # No spec model -> the fallback swap leaves the code-side model in place.
    assert result.output == 'from-function'


async def test_inferred_model_cached_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # The inferred-model cache lives on the fallback swap path, so exercise it with the gate off.
    monkeypatch.setattr(_managed_agent_spec, '_FRAMEWORK_HAS_GET_MODEL', False)

    calls: list[str] = []

    def counting_infer(model: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(model)
        return infer_model(model, *args, **kwargs)

    monkeypatch.setattr(_managed_agent_spec, 'infer_model', counting_infer)

    capability = ManagedAgentSpec('model_cache')
    monkeypatch.setattr(capability, 'get_model', lambda: None)
    agent = Agent(capture({}), capabilities=[capability])

    with capability._variable.override(AgentSpec(model='test')):
        await agent.run('hello')
        await agent.run('hello again')

    # The inferred model is cached per string on the (run-shared) capability, so two runs infer once.
    assert calls == ['test']


# --- spec capabilities materialize ------------------------------------------------------------


async def test_custom_capability_materialized_end_to_end() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('custom_cap', custom_capability_types=[ContributeInstructions])
    agent = Agent(capture(seen), instructions='code', capabilities=[capability])

    spec = AgentSpec.model_validate({'capabilities': [{'ContributeInstructions': {'text': 'materialized'}}]})
    with capability._variable.override(spec):
        await agent.run('hello')

    # The custom capability, referenced by name and built via `custom_capability_types`, takes effect.
    assert seen['instructions'] == ['code\nmaterialized']


async def test_builtin_registry_capability_materialized() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('builtin_cap')
    agent = Agent(capture(seen), capabilities=[capability])

    # `Thinking` is a built-in, serializable, safely-instantiable registry capability.
    spec = AgentSpec.model_validate({'capabilities': [{'Thinking': {'effort': 'high'}}]})
    with capability._variable.override(spec):
        await agent.run('hello')

    assert seen['thinking'] == 'high'


async def test_unknown_capability_name_warns_and_skips() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('unknown_cap')
    agent = Agent(capture(seen), instructions='code', capabilities=[capability])

    spec = AgentSpec.model_validate({'instructions': 'still here', 'capabilities': ['NoSuchCapability']})
    with capability._variable.override(spec):
        with pytest.warns(UserWarning, match="Skipping managed spec capability 'NoSuchCapability'"):
            result = await agent.run('hello')

    # The unknown capability is skipped, but the rest of the spec still applies and the run proceeds.
    assert seen['instructions'] == ['code\nstill here']
    assert result.output == 'from-function'


async def test_malformed_capability_config_warns_and_skips() -> None:
    seen: dict[str, Any] = {}
    capability = ManagedAgentSpec('malformed_cap')
    agent = Agent(capture(seen), capabilities=[capability])

    # An unexpected argument makes the capability's construction raise -> warn + skip.
    spec = AgentSpec.model_validate({'instructions': 'x', 'capabilities': [{'Thinking': {'bogus_field': 1}}]})
    with capability._variable.override(spec):
        with pytest.warns(UserWarning, match="Skipping managed spec capability 'Thinking'"):
            result = await agent.run('hello')

    assert result.output == 'from-function'


# --- fallback semantics -----------------------------------------------------------------------


async def test_invalid_payload_falls_back_to_code(capfire: CaptureLogfire) -> None:
    seen: dict[str, Any] = {}
    reasons: list[str | None] = []
    capability = ManagedAgentSpec('invalid_payload', label='production')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen['instructions'] = [m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions]
        resolved = capability.resolved
        reasons.append(resolved._reason if resolved is not None else None)  # pyright: ignore[reportPrivateUsage]
        return ModelResponse(parts=[TextPart('from-function')])

    config = VariablesConfig(
        variables={
            'agentspec__invalid_payload': VariableConfig(
                name='agentspec__invalid_payload',
                # `model` must be a string or null; a number fails `AgentSpec` validation.
                labels={'production': LabeledValue(version=1, serialized_value='{"model": 123}')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    with variables_provider(capfire, config):
        agent = Agent(FunctionModel(respond), instructions='code', capabilities=[capability])
        result = await agent.run('hello')

    # The bad remote value is rejected; the SDK falls back to the empty code default and the run
    # proceeds with exactly the agent the developer wrote.
    assert reasons == ['validation_error']
    assert seen['instructions'] == ['code']
    assert result.output == 'from-function'


# --- resolution exposure & isolation ----------------------------------------------------------


async def test_resolved_property_exposes_active_resolution() -> None:
    capability = ManagedAgentSpec('resolved_expose')
    captured: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        resolved = capability.resolved
        captured['model'] = resolved.value.model if resolved is not None else '<none>'
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[capability])

    # Outside a run nothing is resolved.
    assert capability.resolved is None

    with capability._variable.override(AgentSpec(model=None, instructions='hi')):
        await agent.run('hello')

    # During the run the active `ResolvedVariable` is exposed; afterwards it is cleared.
    assert captured['model'] is None
    assert capability.resolved is None


async def test_records_resolution_span(capfire: CaptureLogfire) -> None:
    # The resolution records a Logfire span carrying the value/reason. There are two per run: one
    # from `get_model` (sourcing the model at run setup) and one from the per-run `for_run` resolve
    # (baggage + callable targeting inputs). The second `.get()` is a cheap in-memory lookup.
    agent = Agent(TestModel(), capabilities=[ManagedAgentSpec('span_slug')])
    await agent.run('hello')

    resolve_spans = [
        s['name'] for s in capfire.exporter.exported_spans_as_dict() if s['name'].startswith('Resolve variable')
    ]
    assert resolve_spans == [
        'Resolve variable agentspec__span_slug',
        'Resolve variable agentspec__span_slug',
    ]


async def test_per_run_isolation_across_concurrent_runs() -> None:
    capability = ManagedAgentSpec('isolation')
    observed: dict[str, str | None] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions = next((m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None)
        resolved = capability.resolved
        # Record the resolved instructions against the value each concurrent run set.
        observed[resolved.value.instructions] = instructions  # type: ignore[index]
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[capability])

    async def run_with(instructions: str) -> None:
        with capability._variable.override(AgentSpec(instructions=instructions)):
            await agent.run('hello')

    await asyncio.gather(run_with('alpha'), run_with('beta'))

    # Each concurrent run saw its own override, with no cross-talk through the shared capability.
    assert observed == {'alpha': 'alpha', 'beta': 'beta'}


# --- `_ResolvedAgentSpec` unit coverage -------------------------------------------------------


def test_resolved_agent_spec_get_methods() -> None:
    # Cover the per-run capability's `get_*` surfaces directly, including the empty-spec `None`
    # branches. (The model is sourced by `ManagedAgentSpec.get_model`, not the per-run capability.)
    capability = ManagedAgentSpec('unit')
    resolution = capability._variable.get()

    populated = _ResolvedAgentSpec[None](
        spec=AgentSpec(model='some-model', instructions='hi', model_settings={'temperature': 0.5}),
        resolution=resolution,
        resolution_holder=ContextVar('holder', default=None),
        model_cache={},
        trigger_auto_create=lambda _resolution: None,
    )
    assert populated.get_instructions() == ['hi']
    assert populated.get_model_settings() == {'temperature': 0.5}

    empty = _ResolvedAgentSpec[None](
        spec=AgentSpec(),
        resolution=resolution,
        resolution_holder=ContextVar('holder', default=None),
        model_cache={},
        trigger_auto_create=lambda _resolution: None,
    )
    assert empty.get_instructions() is None
    assert empty.get_model_settings() is None


# --- `ManagedAgent` sugar ---------------------------------------------------------------------


def test_managed_agent_returns_agent_with_capability() -> None:
    agent = ManagedAgent('sugar', model=TestModel())
    assert isinstance(agent, Agent)

    seen: list[str] = []
    agent._root_capability.apply(lambda c: seen.append(type(c).__name__))
    assert 'ManagedAgentSpec' in seen


async def test_managed_agent_runs_end_to_end_with_fallback_model() -> None:
    seen: dict[str, Any] = {}
    agent = ManagedAgent('sugar_run', model=capture(seen), instructions='base')

    # With no published spec the fallback model serves the run and the agent's own instructions apply.
    result = await agent.run('hello')
    assert result.output == 'from-function'
    assert seen['instructions'] == ['base']


async def test_managed_agent_spec_overrides_fallback_model() -> None:
    agent = ManagedAgent('sugar_override', model=capture({}))
    # The sugar puts the `ManagedAgentSpec` first in the capability list, so grab it to override.
    managed = next(c for c in agent._root_capability.capabilities if isinstance(c, ManagedAgentSpec))

    with managed._variable.override(AgentSpec(model='test')):
        result = await agent.run('hello')

    # The managed spec's model overrides the fallback per request.
    assert result.output == 'success (no tool calls)'


async def test_managed_agent_extra_capabilities_compose() -> None:
    seen: dict[str, Any] = {}
    agent = ManagedAgent('sugar_extra', model=capture(seen), capabilities=[ContributeInstructions('extra')])

    await agent.run('hello')

    # A capability passed alongside the managed spec composes normally.
    assert seen['instructions'] == ['extra']
