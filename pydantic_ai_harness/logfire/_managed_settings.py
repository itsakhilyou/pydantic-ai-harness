"""Back an agent's model and model settings with a Logfire-managed variable."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from logfire.variables import Variable
from pydantic import BaseModel, ConfigDict
from pydantic_ai.models import Model, ModelRequestContext, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentModelSettings

# Logfire's "agent settings" surface exposes the managed model + settings for an agent with slug
# `<slug>` as a variable named `agent__<slug>`, hyphens replaced by underscores. `agent__` is
# reserved for these system-managed agent-settings variables.
_AGENT_SETTINGS_VARIABLE_PREFIX = 'agent__'


class ManagedModelSettings(BaseModel):
    """The cross-framework subset of model settings that can be managed from Logfire.

    Every field name matches a key in [`pydantic_ai.settings.ModelSettings`][pydantic_ai.settings.ModelSettings]
    so the payload lowers to it with no translation. `extra='allow'` lets forward-compatible
    canonical keys (added to `ModelSettings` in a newer pydantic-ai) flow through untouched.
    """

    model_config = ConfigDict(extra='allow')

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    parallel_tool_calls: bool | None = None
    timeout: float | None = None
    stop_sequences: list[str] | None = None
    thinking: bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | None = None
    service_tier: Literal['auto', 'default', 'flex', 'priority'] | None = None
    provider_options: dict[str, dict[str, Any]] | None = None
    """Per-provider escape hatch: `provider_options[provider][key]` lowers to the flat
    `<provider>_<key>` model setting (pydantic-ai's provider-prefix convention), applied after
    the canonical fields so a provider-specific value wins over its canonical counterpart."""


class ManagedSettingsValue(BaseModel):
    """The value backing a [`ManagedSettings`][pydantic_ai_harness.logfire.ManagedSettings] capability.

    An empty value (the default when nothing is configured in Logfire yet) leaves the agent's
    code-defined model and settings untouched.
    """

    # `model` collides with Pydantic's protected `model_` namespace; opt out so the field name
    # can mirror the pydantic-ai model string exactly without a spurious warning.
    model_config = ConfigDict(protected_namespaces=())

    model: str | None = None
    """A pydantic-ai model string (e.g. `'openai:gpt-5'`) to run with. `None` keeps the code model."""

    settings: ManagedModelSettings | None = None
    """Model settings to patch on top of the agent's code-defined settings. `None` changes nothing."""


def _lower_settings(value: ManagedModelSettings) -> ModelSettings:
    """Lower a managed settings payload to a `pydantic_ai.settings.ModelSettings` dict.

    Only fields that are actually set are included, so unset fields keep the agent's code-defined
    values once the result is merged. `provider_options[provider][key]` is flattened to the
    `<provider>_<key>` key and applied after the canonical fields, so a provider-specific value
    wins over its canonical counterpart (matching pydantic-ai's documented precedence).
    """
    # `ModelSettings` is a `TypedDict` with fixed keys, so build a plain dict for the dynamic
    # provider-prefixed keys and cast at the end.
    settings: dict[str, Any] = value.model_dump(exclude_none=True, exclude={'provider_options'})

    if value.provider_options:
        for provider, options in value.provider_options.items():
            for key, option in options.items():
                settings[f'{provider}_{key}'] = option

    return cast(ModelSettings, settings)


@dataclass
class ManagedSettings(ManagedVariableCapability[AgentDepsT, ManagedSettingsValue]):
    """Back an agent's model and model settings with a Logfire-managed variable.

    Pass an agent-settings name and (optionally) a code default and the capability declares the
    backing [managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
    for you -- a name of `checkout_assistant` resolves the variable `agent__checkout_assistant`,
    matching the naming Logfire's "agent settings" surface uses. You can steer the model and its
    settings from the Logfire UI -- versioned, labelled, and rolled out -- without redeploying,
    while the agent's code-defined model and settings keep it working when no remote value is
    available.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedSettings

    logfire.configure()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[ManagedSettings('checkout_assistant', label='production')],
    )
    result = agent.run_sync('Refund my last order.')
    ```

    The value is resolved **once per run**, inside the run's
    [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] hook, using the
    [`ResolvedVariable`][logfire.variables.ResolvedVariable] as a context manager that stays open
    for the whole run -- so the selected label and version are attached as baggage to every child
    span of the agent run.

    **Patch semantics:** every managed value is a patch on the agent's code-defined config. Unset
    fields keep the agent's code-defined values, and removing a field in Logfire is a deliberate
    revert-to-code -- not a reset to some SDK default.

    **Precedence:** managed settings merge **over** the agent's constructor `model_settings` but
    **under** per-run `model_settings=` passed to `run()`/`run_sync()`, so run arguments always win.

    **Model override:** when the managed value sets `model`, it overrides the model per request via
    [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request], for
    agents that already have a code-side model. Two known limits, both pending future pydantic-ai
    (run-spec) work: a fully model-less agent still requires a model today, and a managed model
    currently overrides even a per-run `model=` passed at the call site -- the hook can't
    distinguish a run argument from the agent default, so the run-arguments-win precedence that
    settings enjoy doesn't yet hold for the model itself.

    **Fallback semantics:** if the remote value is missing, invalid, or unreachable, the logfire
    SDK falls back to the code default and records the reason on the resolve span, so the run never
    crashes on a bad managed value.

    Pass an existing [`logfire.variables.Variable`][logfire.variables.Variable] as `name` instead
    of an agent-settings name when you want to use a variable you defined yourself.
    """

    name: str | Variable[ManagedSettingsValue]
    """The agent-settings name (declared as the variable `agent__<name>`), or a pre-built `logfire.Variable`."""

    default: ManagedSettingsValue | None = None
    """Code-default managed value. When omitted, an empty value is used -- nothing is managed until
    a value is configured in Logfire. Ignored when `name` is a `Variable`."""

    def __post_init__(self) -> None:
        self._resolved = self._new_resolved()
        # Inferred `Model` instances keyed by model string, so a repeated override isn't re-inferred.
        self._model_cache: dict[str, Model] = {}
        if not isinstance(self.name, str):
            self._warn_logfire_instance_ignored('name')
            self._variable = self.name
            return

        self._variable = self._build_managed_variable(
            self.name,
            prefix=_AGENT_SETTINGS_VARIABLE_PREFIX,
            value_type=ManagedSettingsValue,
            default=self.default or ManagedSettingsValue(),
        )

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Merge the resolved managed settings on top of the agent's settings, under run arguments."""

        def model_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            resolved = self.resolved
            if resolved is None or resolved.value.settings is None:
                # No active run, or nothing managed -- contribute no settings.
                return ModelSettings()
            return _lower_settings(resolved.value.settings)

        return model_settings

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Override the request's model when the resolved managed value sets one."""
        resolved = self.resolved
        if resolved is None or resolved.value.model is None:
            return request_context

        model_string = resolved.value.model
        model = self._model_cache.get(model_string)
        if model is None:
            model = self._model_cache[model_string] = infer_model(model_string)
        return dataclasses.replace(request_context, model=model)
