# Logfire-backed capabilities

Drive agent configuration from [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/),
so you can iterate on it from the Logfire UI -- versioned, labelled, and rolled out -- without redeploying.

Each capability manages one surface of the agent, so you adopt exactly as much as you want:

- [`ManagedPrompt`](#managedprompt) -- the agent's instructions
- [`ManagedToolDefinitions`](#managedtooldefinitions) -- the LLM-facing definitions (name,
  description, parameter docs) of the agent's tools
- [`ManagedSettings`](#managedsettings) -- the agent's model and model settings

...or manage the whole shape at once:

- [`ManagedAgentSpec`](#managedagentspec) -- the agent's instructions, model, settings, and
  capabilities together, as one versioned `AgentSpec`

They share one contract: **the code-defined agent is the fallback.** Every managed value is a
patch on what's written in code -- unset fields keep their code values, and a missing, invalid,
or unreachable remote value degrades to exactly the agent the developer wrote, never a crashed
run. Values resolve **once per run** and the resolved label + version ride as baggage on every
span of the run, so traces always show which version produced which behavior.

**Auto-create on first use:** when the backing variable doesn't exist in Logfire yet, it is
created in the background on first use -- from the code default, with the payload's JSON schema
and description -- so the Logfire UI becomes the editing surface without a manual create step.
Opt out per capability with `auto_create=False`.

Install the extra:

```bash
pip install 'pydantic-ai-harness[logfire]'
```

## `ManagedPrompt`

Back an agent's instructions with a Logfire-managed
[Prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/).

### The problem

Prompts are critical to agent behavior, but iterating on them through the normal
edit → review → deploy loop is slow, and you can't easily A/B test a change or roll it
back the moment it misbehaves in production.

### The solution

`ManagedPrompt` declares the backing managed variable for you and resolves it **once per
run**, feeding the value into the agent's instructions. The resolution happens inside the
run's `wrap_run` hook using the
[`ResolvedVariable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as a context manager that stays open for the whole run -- so the selected label and version
are attached as baggage to every child span of the agent run. You get a direct correlation
between a run's behavior and the exact prompt version that produced it, plus instant
iteration and rollback from the Logfire UI.

### Usage

Pass the prompt name and a default value. The name `support_agent` is declared as the managed
variable `prompt__support_agent` -- the naming Logfire's Prompt management uses (hyphens in a
name become underscores). The default keeps the agent working until a remote value is published.

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt

logfire.configure()

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are a helpful customer support agent. Be friendly and concise.',
            label='production',
        )
    ],
)

result = agent.run_sync('My order never arrived.')
print(result.output)
```

### Targeting

For deterministic A/B assignment (the same user always sees the same label), pass a
`targeting_key`. It can be a static string or a callable that derives the key from the
[`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) -- handy
when the key lives in your agent's `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt


@dataclass
class Deps:
    user_id: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are a helpful customer support agent.',
            targeting_key=lambda ctx: ctx.deps.user_id,
        ),
    ],
)
```

Pass `attributes` (or a callable returning them) for condition-based targeting rules.
When `label` is omitted, the variable's rollout and targeting rules pick the label;
when both `targeting_key` and `attributes` are omitted, Logfire falls back to its own
targeting context and then to the active trace id.

### Templating with deps

By default the resolved prompt is used verbatim. Pass `render_template=True` to render it as a
Handlebars template against the agent's `deps` -- the same mechanism as
[`TemplateStr`](https://ai.pydantic.dev/api/#pydantic_ai.TemplateStr) -- so `{{field}}` is filled
from `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt


@dataclass
class Deps:
    customer_name: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are helping {{customer_name}}. Be friendly and concise.',
            render_template=True,
        ),
    ],
)
```

Rendering requires `pydantic-handlebars` (install `pydantic-ai-slim[spec]`). It is off by default.

### Prompt-cache trade-off

The resolved value lands in the agent's **system instructions**. Provider prompt caches (Anthropic,
OpenAI, etc.) key strictly by prefix -- `tools → system → messages` -- so any change to the system
block invalidates the cached prefix for the affected runs.

| Mode | Cache impact |
| --- | --- |
| Pinned `label='production'`, no rollout split | **Cache-stable.** The value only changes on a deliberate prompt rollout, which is the same cost as a redeploy. |
| Percentage rollout across labels (no `label=`) | Different runs land on different labels → splits the cache into one lane per label. |
| `targeting_key` per user/tenant with multiple labels in play | Cache lanes per assigned label; deterministic per key but still N lanes overall. |
| Mid-traffic label flip in the Logfire UI | One-shot cold-invalidation for everyone on that label. |

In short: pinning a `label` keeps the cache hot; using `ManagedPrompt` as an A/B platform is opt-in
cache cost. If you don't need rollouts, `label='production'` is the recommended default.

### Using your own variable

Declaring the same name more than once is fine -- each `ManagedPrompt` builds its own backing
variable, so sharing a prompt across several agents just works. Pass an existing
[`logfire.variables.Variable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as the first argument instead of a name when you want to declare the variable yourself --
for example a `template_var`, or one registered for `variables_push`:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt

logfire.configure()

support_prompt = logfire.var(
    name='prompt__support_agent',
    type=str,
    default='You are a helpful customer support agent. Be friendly and concise.',
)

agent = Agent('openai:gpt-5', capabilities=[ManagedPrompt(support_prompt, label='production')])
```

When `name` is a prompt name, pass `logfire_instance=` to declare the variable on a specific
Logfire instance instead of the module-level default.

### Notes

- The prompt resolves to a `str`. By default it's used verbatim; set `render_template=True`
  to render `{{...}}` against `deps` (see [Templating with deps](#templating-with-deps)).
- Resolution is isolated per run via a context variable, so a single capability instance
  is safe to share across concurrent runs.
- `ManagedPrompt.resolved` exposes the active run's `ResolvedVariable` (value, label, version,
  reason) for inspection -- e.g. from inside a tool.
- The capability runs outermost (wrapping `Instrumentation`) so the resolved variable's baggage
  covers the agent run span as well as its children. On recent Logfire versions both the
  selected label and the version are propagated as separate baggage attributes.
- Resolution happens **once per run**. A label flip or rollout change that lands in Logfire
  mid-run is not picked up until the next run starts -- the trade-off for run-stable
  instructions and a single baggage scope across all child spans.
- For Logfire-side targeting that lives outside the agent (e.g. set once per request handler),
  use Logfire's
  [`targeting_context`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
  in an outer scope; `ManagedPrompt` only needs `targeting_key`/`attributes` when the key
  comes from the agent's `RunContext`.

## `ManagedToolDefinitions`

Override the LLM-facing definitions of an agent's tools -- name, description, and parameter
descriptions -- from one managed variable.

### The problem

Tools and their descriptions are half of what the model sees when deciding what to do -- as much
a part of the "prompt" as the instructions. But tool definitions live in code, so tuning how a
tool is framed to the model (or letting an optimizer propose that tuning) takes a redeploy per
tweak.

### The solution

Drop `ManagedToolDefinitions` onto the agent and every tool's *definition* becomes manageable
from Logfire, while the tool itself -- its implementation and its parameter schema structure --
stays exactly as written in code. A tool is the executable unit; the tool definition is the
LLM-facing spec, and only that spec is remotely patchable, so a remote value can never drift
from the validator the tool actually runs against.

### Usage

The name `checkout_assistant` is declared as the managed variable
`tool_definitions__checkout_assistant`, holding a list of per-tool overrides:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedToolDefinitions

logfire.configure()

def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny.'

agent = Agent(
    'openai:gpt-5',
    tools=[get_weather],
    capabilities=[ManagedToolDefinitions('checkout_assistant', label='production')],
)
```

Each entry in the list is a `ToolDefinitionOverride` keyed to a tool by its original (code-side)
`name`; every other field is optional and unset fields keep the tool's own definition:

```json
[
  {
    "name": "get_weather",
    "new_name": "lookup_weather",
    "description": "Look up the current weather for a city.",
    "parameter_descriptions": {"city": "City name, e.g. 'London'"}
  }
]
```

### Notes

- **Renames round-trip:** `new_name` changes the name the model is shown; a call to the renamed
  tool routes back to the original implementation, and `ctx.tool_name` inside the tool is the
  original name. A rename that collides with a name another tool already advertises is dropped
  with a warning (other patches still apply) rather than breaking the run.
- An override whose `name` matches no tool on the agent is inert -- that's the drift case (the
  tool was removed or renamed in code), and the Logfire UI is where it becomes visible.
- Only the `description` strings inside the parameter schema can be patched; parameter names,
  types, and required-ness are deliberately fixed in code.

## `ManagedSettings`

Back an agent's model and model settings with a Logfire-managed variable.

### The problem

Model choice and sampling settings are the cheapest knobs to tune and the most annoying to
redeploy for -- switching model for a canary, nudging temperature, or capping output tokens
shouldn't require a release.

### The solution

`ManagedSettings` resolves an `agent__<name>` variable whose value patches the agent's model
and model settings. Settings merge **over** the agent's constructor `model_settings` and
**under** per-run `model_settings=`, so run arguments always win.

### Usage

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedSettings

logfire.configure()

agent = Agent(
    'openai:gpt-5',
    capabilities=[ManagedSettings('checkout_assistant', label='production')],
)
```

The value patches the model and any of the canonical, cross-framework settings keys (they match
`pydantic_ai.settings.ModelSettings`), with a nested `provider_options` escape hatch for
provider-specific settings (`provider_options.openai.reasoning_effort` lowers to the
`openai_reasoning_effort` model setting, and a provider-specific value wins over its canonical
counterpart):

```json
{
  "model": "openai:gpt-5",
  "settings": {
    "temperature": 0.4,
    "max_tokens": 2048,
    "thinking": "high",
    "provider_options": {
      "anthropic": {"thinking": {"type": "enabled", "budget_tokens": 16384}}
    }
  }
}
```

### Notes

- **Model override limits (for now):** the managed `model` overrides per request via
  `before_model_request`, which requires the agent to have a code-side model and cannot yet
  distinguish a per-run `model=` argument from the agent default -- so unlike settings, the
  run-arguments-win precedence doesn't yet hold for the model itself. Both limits are pending
  run-spec work in pydantic-ai.
- `thinking` accepts `true`/`false` or an effort level (`'minimal'` ... `'xhigh'`), exactly like
  the unified `thinking` model setting; per-provider lowering (e.g. effort to budget tokens) is
  pydantic-ai's existing behavior.

## `ManagedAgentSpec`

Back a whole agent's shape -- instructions, model, model settings, and capabilities -- with a single
Logfire-managed [`AgentSpec`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.spec.AgentSpec).

### The problem

The per-surface capabilities above each manage one knob. Sometimes you want to steer the agent's
whole configuration together -- swap the model *and* nudge the instructions *and* enable a
capability -- and have that land as one atomic, versioned change you can roll out or roll back in a
single step, rather than coordinating several variables.

### The solution

`ManagedAgentSpec` resolves an `agentspec__<name>` variable whose value is an entire `AgentSpec`.
Its instructions, model, settings, and `capabilities` all layer onto the code-defined agent, so one
managed value drives the whole shape. It composes with the per-surface capabilities and with your
code-defined tools and capabilities -- the spec adds, it never removes.

### Usage

The name `checkout_assistant` is declared as the managed variable `agentspec__checkout_assistant`,
matching the naming Logfire's "Agent Specs" surface uses:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedAgentSpec

logfire.configure()

agent = Agent(
    'openai:gpt-5',
    capabilities=[ManagedAgentSpec('checkout_assistant', label='production')],
)
```

The value is a JSON `AgentSpec`: its `instructions` add to the agent's own, `model_settings` merge
over the agent's (under per-run `model_settings=`), `model` overrides per request, and each entry in
`capabilities` is materialized from the capability registry:

```json
{
  "model": "openai:gpt-5",
  "instructions": "Be concise and always confirm the order id before refunding.",
  "model_settings": {"temperature": 0.3},
  "capabilities": [{"Thinking": {"effort": "high"}}]
}
```

Reference your own capability classes by name by passing them as `custom_capability_types`; built-in
capability names (e.g. `Thinking`) are always available.

For the common case -- an agent whose whole shape is managed -- the `ManagedAgent` sugar builds the
agent for you in one call:

```python
import logfire

from pydantic_ai_harness.logfire import ManagedAgent

logfire.configure()

agent = ManagedAgent('checkout_assistant', model='openai:gpt-5', label='production')
result = agent.run_sync('Refund my last order.')
```

`ManagedAgent` returns a real `Agent` (not a builder); the managed values just flow in per run.
Pass a fallback `model` so the agent can run before any spec is published -- the spec's `model`, when
set, then overrides it per request.

### Notes

- **Additive, never destructive:** a missing, invalid, or unreachable value degrades to exactly the
  agent the developer wrote. Local tools, toolsets, and code-defined capabilities always stay in code.
- **Capabilities materialize per run:** an unknown capability name, or one whose construction fails,
  is skipped with a warning rather than crashing the run.
- **Model override limits (for now):** as with [`ManagedSettings`](#managedsettings), the managed
  `model` overrides per request via `before_model_request`, so it needs a code-side (or `ManagedAgent`
  fallback) model and can't yet distinguish a per-run `model=` argument from the agent default. A
  forward-compatible `get_model` hook is already wired up for when pydantic-ai grows the framework
  surface for it. Both limits are pending run-spec work in pydantic-ai.
- The spec resolves **once per run**, in `for_run` (earlier than the per-surface capabilities, since
  the resolved spec decides what the run is assembled from), and its label + version ride as baggage
  on every span of the run. `ManagedAgentSpec.resolved` exposes the active run's `ResolvedVariable`.
