"""Tests for the `ManagedSettings` capability (source package `pydantic_ai_harness.logfire`).

Shared fixtures (`anyio_backend`, Logfire configuration) live in `conftest.py`; the
variable-naming contract common to all managed-variable capabilities is covered in
`test_managed_variable.py`. This module focuses on `ManagedSettings` patching the agent's model
settings, overriding its model, and falling back to code on a bad remote value.
"""

from __future__ import annotations

from typing import Any

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from pydantic_ai_harness.logfire._managed_settings import (
    ManagedModelSettings,
    ManagedSettings,
    ManagedSettingsValue,
    _lower_settings,
)

from ._helpers import variables_provider

pytestmark = pytest.mark.anyio


def capture_settings(seen: list[ModelSettings | None]) -> FunctionModel:
    """A model that records the merged `model_settings` it is shown, then ends the run."""

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart('from-function')])

    return FunctionModel(respond)


# --- `_lower_settings` unit tests -------------------------------------------------------------


def test_lower_settings_all_canonical_fields() -> None:
    value = ManagedModelSettings(
        max_tokens=2048,
        temperature=0.4,
        top_p=0.9,
        top_k=40,
        seed=7,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        parallel_tool_calls=False,
        timeout=30.0,
        stop_sequences=['END'],
        thinking='high',
        service_tier='flex',
    )

    assert _lower_settings(value) == {
        'max_tokens': 2048,
        'temperature': 0.4,
        'top_p': 0.9,
        'top_k': 40,
        'seed': 7,
        'presence_penalty': 0.1,
        'frequency_penalty': 0.2,
        'parallel_tool_calls': False,
        'timeout': 30.0,
        'stop_sequences': ['END'],
        'thinking': 'high',
        'service_tier': 'flex',
    }


def test_lower_settings_omits_unset_fields() -> None:
    # Only fields that are actually set land in the dict, so a merge keeps code values elsewhere.
    assert _lower_settings(ManagedModelSettings(temperature=0.5)) == {'temperature': 0.5}


@pytest.mark.parametrize('thinking', [True, False, 'minimal', 'high'])
def test_lower_settings_thinking_scalar_union(thinking: Any) -> None:
    # The scalar union round-trips verbatim, including the `False` case (not dropped as "unset").
    assert _lower_settings(ManagedModelSettings(thinking=thinking)) == {'thinking': thinking}


def test_lower_settings_extra_canonical_key_passthrough() -> None:
    # A forward-compatible canonical key (unknown to this pydantic-ai version) flows through.
    value = ManagedModelSettings.model_validate({'temperature': 0.3, 'future_setting': 'x'})
    assert _lower_settings(value) == {'temperature': 0.3, 'future_setting': 'x'}


def test_lower_settings_provider_options_flattened() -> None:
    value = ManagedModelSettings(provider_options={'openai': {'reasoning_effort': 'high'}})
    assert _lower_settings(value) == {'openai_reasoning_effort': 'high'}


def test_lower_settings_provider_option_beats_canonical() -> None:
    # Provider-specific options are applied after canonical/extra keys, so they win on collision.
    value = ManagedModelSettings.model_validate(
        {'openai_reasoning_effort': 'low', 'provider_options': {'openai': {'reasoning_effort': 'high'}}}
    )
    assert _lower_settings(value) == {'openai_reasoning_effort': 'high'}


# --- variable naming / construction -----------------------------------------------------------


def test_name_becomes_agent_variable_name() -> None:
    capability = ManagedSettings('checkout_assistant')
    assert capability._variable.name == 'agent__checkout_assistant'


def test_default_not_required() -> None:
    # Unlike `ManagedPrompt`, `default` is optional -- an empty value means "nothing managed yet".
    capability = ManagedSettings('no_default')
    assert capability._variable.default == ManagedSettingsValue()


def test_prebuilt_variable_prefix_warning() -> None:
    with pytest.warns(UserWarning, match="'agent__' prefix is added automatically"):
        capability = ManagedSettings('agent__foo')
    assert capability._variable.name == 'agent__foo'


def test_logfire_instance_with_prebuilt_variable_warns() -> None:
    var = logfire.var(name='agent__instance_conflict', type=ManagedSettingsValue, default=ManagedSettingsValue())
    with pytest.warns(UserWarning, match='is ignored when `name` is a `Variable`'):
        ManagedSettings(var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)


def test_model_field_no_protected_namespace() -> None:
    # `model` collides with Pydantic's protected `model_` namespace; `protected_namespaces=()`
    # silences the warning, which under the suite's warnings-as-errors would otherwise fail at
    # class-definition (import) time. Confirm the field is usable and the guard is in place.
    assert ManagedSettingsValue(model='openai:gpt-5').model == 'openai:gpt-5'
    assert ManagedSettingsValue.model_config.get('protected_namespaces') == ()


# --- settings application ---------------------------------------------------------------------


async def test_no_remote_value_leaves_agent_unchanged() -> None:
    seen: list[ModelSettings | None] = []
    agent = Agent(
        capture_settings(seen),
        model_settings=ModelSettings(temperature=0.2, top_p=0.9),
        capabilities=[ManagedSettings('unchanged')],
    )

    result = await agent.run('hello')

    # No managed value -> the agent's own settings and model are used verbatim.
    assert seen == [{'temperature': 0.2, 'top_p': 0.9}]
    assert result.output == 'from-function'


async def test_settings_patch_overrides_and_keeps_unset() -> None:
    seen: list[ModelSettings | None] = []
    capability = ManagedSettings('patch')
    agent = Agent(
        capture_settings(seen),
        model_settings=ModelSettings(temperature=0.2, top_p=0.9),
        capabilities=[capability],
    )

    with capability._variable.override(ManagedSettingsValue(settings=ManagedModelSettings(temperature=0.7))):
        await agent.run('hello')

    # Managed `temperature` wins over the agent default; `top_p` is unset in the managed value
    # so the agent's value is kept.
    assert seen == [{'temperature': 0.7, 'top_p': 0.9}]


async def test_run_settings_win_over_managed() -> None:
    seen: list[ModelSettings | None] = []
    capability = ManagedSettings('run_wins')
    agent = Agent(capture_settings(seen), capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(settings=ManagedModelSettings(temperature=0.7))):
        await agent.run('hello', model_settings=ModelSettings(temperature=0.1))

    # Run-level `model_settings=` is merged last, so it beats the managed value.
    assert seen == [{'temperature': 0.1}]


async def test_provider_options_reach_model_settings() -> None:
    seen: list[ModelSettings | None] = []
    capability = ManagedSettings('provider_opts')
    agent = Agent(capture_settings(seen), capabilities=[capability])

    value = ManagedSettingsValue(
        settings=ManagedModelSettings(provider_options={'openai': {'reasoning_effort': 'high'}})
    )
    with capability._variable.override(value):
        await agent.run('hello')

    assert seen == [{'openai_reasoning_effort': 'high'}]


@pytest.mark.parametrize('thinking', [True, False, 'high'])
async def test_thinking_scalar_union_round_trips(thinking: Any) -> None:
    # The model layer consumes the unified `thinking` setting into the request parameters (only
    # for a thinking-capable profile), so observe it there rather than in `model_settings`.
    seen: list[Any] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.thinking)
        return ModelResponse(parts=[TextPart('from-function')])

    capability = ManagedSettings(f'thinking_{thinking}')
    agent = Agent(
        FunctionModel(respond, profile=ModelProfile(supports_thinking=True)),
        capabilities=[capability],
    )

    with capability._variable.override(ManagedSettingsValue(settings=ManagedModelSettings(thinking=thinking))):
        await agent.run('hello')

    assert seen == [thinking]


# --- model override (via the `get_model` hook on current pydantic-ai) -------------------------


async def test_managed_model_beats_constructor_model() -> None:
    # The managed model is sourced at run setup via `get_model`, slotting in above the agent's
    # constructor model, so it replaces the code-side `FunctionModel`.
    seen: list[ModelSettings | None] = []
    capability = ManagedSettings('model_override')
    agent = Agent(capture_settings(seen), capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model='test')):
        result = await agent.run('hello')

    # `model='test'` -> served by `TestModel`, not the code-side `FunctionModel`.
    assert result.output == 'success (no tool calls)'


async def test_model_less_agent_runs_via_get_model() -> None:
    # A fully model-less agent runs entirely off the managed model, now that `get_model` sources it.
    capability = ManagedSettings('model_less')
    agent = Agent(capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model='test')):
        result = await agent.run('hello')

    assert result.output == 'success (no tool calls)'


async def test_run_model_beats_managed_model() -> None:
    # The wart fix: `get_model` slots the managed model *below* a call-site `run(model=...)`, so the
    # run argument now wins (before the hook, `before_model_request` clobbered it every request).
    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('from-run-model')])

    capability = ManagedSettings('run_model_wins')
    agent = Agent(capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model='test')):
        result = await agent.run('hello', model=FunctionModel(respond))

    assert result.output == 'from-run-model'


async def test_model_none_keeps_code_model() -> None:
    seen: list[ModelSettings | None] = []
    capability = ManagedSettings('model_none')
    agent = Agent(capture_settings(seen), capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model=None)):
        result = await agent.run('hello')

    # `model=None` -> the code-side `FunctionModel` still serves the run.
    assert result.output == 'from-function'


async def test_callable_targeting_still_resolves_model() -> None:
    # Model selection happens before the run exists, so callable `targeting_key`/`attributes` can't
    # run; `get_model` uses the static path (callables -> None) rather than crashing.
    capability = ManagedSettings(
        'callable_target',
        targeting_key=lambda ctx: 'user-key',
        attributes=lambda ctx: {'tier': 'gold'},
    )
    agent = Agent(capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model='test')):
        result = await agent.run('hello')

    assert result.output == 'success (no tool calls)'


# --- model override fallback (older pydantic-ai without the `get_model` hook) ------------------
#
# On older pydantic-ai the framework doesn't source a capability model, so the model is swapped per
# request in `before_model_request` instead. We simulate that by forcing the module gate off and
# stubbing `get_model` to `None` (so the framework contributes no model and the agent falls back to
# its code-side model, which the per-request swap then overrides).


async def test_before_model_request_swaps_model_on_old_pydantic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic_ai_harness.logfire._managed_settings as module

    monkeypatch.setattr(module, '_FRAMEWORK_HAS_GET_MODEL', False)
    capability = ManagedSettings('old_swap')
    monkeypatch.setattr(capability, 'get_model', lambda: None)
    agent = Agent(capture_settings([]), capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model='test')):
        result = await agent.run('hello')

    # The per-request swap replaces the code-side `FunctionModel` with `TestModel`.
    assert result.output == 'success (no tool calls)'


async def test_before_model_request_no_managed_model_keeps_code_on_old_pydantic_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pydantic_ai_harness.logfire._managed_settings as module

    monkeypatch.setattr(module, '_FRAMEWORK_HAS_GET_MODEL', False)
    capability = ManagedSettings('old_no_model')
    monkeypatch.setattr(capability, 'get_model', lambda: None)
    agent = Agent(capture_settings([]), capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model=None)):
        result = await agent.run('hello')

    # No managed model -> the fallback swap leaves the code-side model in place.
    assert result.output == 'from-function'


async def test_inferred_model_cached_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic_ai_harness.logfire._managed_settings as module

    # The inferred-model cache lives on the fallback swap path, so exercise it with the gate off.
    monkeypatch.setattr(module, '_FRAMEWORK_HAS_GET_MODEL', False)

    calls: list[str] = []
    real_infer = module.infer_model

    def counting_infer(model: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(model)
        return real_infer(model, *args, **kwargs)

    monkeypatch.setattr(module, 'infer_model', counting_infer)

    capability = ManagedSettings('model_cache')
    monkeypatch.setattr(capability, 'get_model', lambda: None)
    agent = Agent(capture_settings([]), capabilities=[capability])

    with capability._variable.override(ManagedSettingsValue(model='test')):
        await agent.run('hello')
        await agent.run('hello again')

    # The inferred model is cached per string, so two runs infer exactly once.
    assert calls == ['test']


# --- fallback semantics -----------------------------------------------------------------------


async def test_invalid_remote_value_falls_back_to_code(capfire: CaptureLogfire) -> None:
    reasons: list[str | None] = []
    seen: list[ModelSettings | None] = []
    capability = ManagedSettings('invalid_remote', label='production')

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_settings)
        resolved = capability.resolved
        reasons.append(resolved._reason if resolved is not None else None)  # pyright: ignore[reportPrivateUsage]
        return ModelResponse(parts=[TextPart('from-function')])

    config = VariablesConfig(
        variables={
            'agent__invalid_remote': VariableConfig(
                name='agent__invalid_remote',
                labels={
                    'production': LabeledValue(
                        version=1, serialized_value='{"settings": {"temperature": "not-a-number"}}'
                    )
                },
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    with variables_provider(capfire, config):
        agent = Agent(
            FunctionModel(respond),
            model_settings=ModelSettings(temperature=0.2),
            capabilities=[capability],
        )
        result = await agent.run('hello')

    # The bad remote value is rejected; the SDK falls back to the code default and the run proceeds
    # with the agent's own settings and model.
    assert reasons == ['validation_error']
    assert seen == [{'temperature': 0.2}]
    assert result.output == 'from-function'
