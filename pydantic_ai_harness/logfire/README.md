# Logfire-backed capabilities

Drive agent configuration from [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/),
so you can iterate on it from the Logfire UI -- versioned, labelled, and rolled out -- without redeploying.

Install the extra:

```bash
pip install 'pydantic-ai-harness[logfire]'
```

## `ManagedPrompt`

Back an agent's instructions with a Logfire-managed
[Prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/).

> A broader, first-party `Managed` capability is in flight in
> [pydantic-ai#5107](https://github.com/pydantic/pydantic-ai/pull/5107) and will eventually be
> importable as `pydantic_ai.managed.logfire.Managed` -- covering instructions, model settings,
> and whole-spec variables. Until then, `ManagedPrompt` is the supported path for backing
> instructions with a Logfire-managed prompt.

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

## `ManagedTool`

Back a single tool's advertised **name**, **description**, and **parameter descriptions** with a
Logfire-managed variable, so you can iterate on how a tool is framed to the model from the Logfire
UI -- versioned, labelled, and rolled out -- without redeploying.

### The problem

A tool's name, description, and parameter wording steer when and how the model calls it. Tuning that
wording -- or A/B testing two phrasings, or rolling back framing that causes bad calls -- normally
means an edit → review → deploy loop, even though the tool's implementation never changes.

### The solution

`ManagedTool` wraps the agent's assembled toolset and applies a remotely-resolved override to one
tool. The tool keeps its code-defined implementation **and parameter schema structure**; only what
the model is **shown** changes. A tool named `get_weather` resolves the variable `tool__get_weather`,
which holds a `ManagedToolOverride` -- a patch whose unset fields leave the tool's own definition
untouched. The override resolves **once per run** inside the run's `wrap_run` hook, so the selected
label and version are attached as baggage to every child span (the same mechanism as `ManagedPrompt`).

### Usage

Pass a tool **name** to manage a tool registered elsewhere on the agent:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedTool

logfire.configure()

agent = Agent('openai:gpt-5', capabilities=[ManagedTool('get_weather', label='production')])


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny.'
```

Or pass a `Tool` to have the capability both **provide and manage** it (wrap a bare function with
`Tool(...)`); the variable name then defaults to the tool's name:

```python
import logfire
from pydantic_ai import Agent, Tool

from pydantic_ai_harness.logfire import ManagedTool

logfire.configure()


def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny.'


agent = Agent('openai:gpt-5', capabilities=[ManagedTool(Tool(get_weather), label='production')])
```

The code default is an empty override (no changes), so the agent keeps using the tool's
code-defined framing until a remote value is published. Publish a `ManagedToolOverride` to the
`tool__get_weather` variable to override only the fields you set:

```python
from pydantic_ai_harness.logfire import ManagedToolOverride

# In Logfire, this serializes to the variable's JSON value; here it's the code default.
ManagedToolOverride(
    description='Look up the current weather for a city. Prefer this over guessing.',
    parameter_descriptions={'city': 'City name, e.g. "London" or "London, GB".'},
)
```

### What can and can't be overridden

You can override the tool's **name**, its **description**, and the **description of each parameter**
(keyed by name in `parameter_descriptions`). What you **can't** change is the parameter schema's
*structure* -- parameter names, types, and which are required. That structure is defined in code
alongside the tool's argument validator; letting a remote value reshape it without changing the
validator it's checked against would let the two silently drift, so a published override could make
every call fail validation. Parameter descriptions are safe because the validator ignores them.
Names in `parameter_descriptions` that aren't real parameters are ignored.

### Targeting and labels

`label`, `targeting_key`, and `attributes` work exactly as they do for
[`ManagedPrompt`](#targeting): pin a `label` (e.g. `'production'`) for stable framing, or pass a
`targeting_key` (static or a callable over the `RunContext`) for deterministic A/B assignment.

### Notes

- One `ManagedTool` manages one tool. Add several capabilities to manage several tools, or use
  [`ManagedToolset`](#managedtoolset) to manage a group from one variable.
- The first argument is the tool's **original** name, or a `Tool` to own. Pass `variable=` to back it
  with a differently-named variable (or a pre-built `logfire.variables.Variable`); it defaults to the
  tool name.
- Renaming a tool re-advertises it under the new name and routes the model's calls back to the
  original implementation. A rename that collides with another tool raises a `UserError`.
- If the managed tool isn't present in the toolset for a run (for example a deferred Tool Search
  tool that hasn't been discovered yet), the override is a no-op for that run.
- `ManagedTool.resolved` exposes the active run's `ResolvedVariable` (value, label, version, reason)
  for inspection -- e.g. from inside a tool.
- The `tool__` prefix is a harness convention (Logfire reserves `prompt__` for managed prompts but
  has no first-party tool-management feature); it namespaces the backing variable and groups it with
  `ManagedPrompt`'s variables.

## `ManagedToolset`

Where `ManagedTool` manages one tool per variable, `ManagedToolset` manages a whole **group** of
tools from one Logfire variable -- a `dict` mapping tool name to `ManagedToolOverride`. A group named
`weather` resolves the variable `toolset__weather`. This is the ergonomic choice for reframing a
whole MCP server or a large toolset from one place in the UI.

Pass `tools` to provide and manage them, or omit it to manage tools registered elsewhere on the
agent. The resolved map's keys select which tools are overridden by their original name; keys that
match no tool are ignored.

```python
import logfire
from pydantic_ai import Agent, Tool

from pydantic_ai_harness.logfire import ManagedToolset

logfire.configure()


def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny.'


def get_forecast(city: str) -> str:
    return f'The forecast for {city} is sunny.'


agent = Agent(
    'openai:gpt-5',
    capabilities=[ManagedToolset('weather', tools=[Tool(get_weather), Tool(get_forecast)], label='production')],
)
```

Publish a map to the `toolset__weather` variable to override any subset of the group:

```python
from pydantic_ai_harness.logfire import ManagedToolOverride

# In Logfire, this serializes to the variable's JSON value.
{
    'get_weather': ManagedToolOverride(description='Current conditions for a city.'),
    'get_forecast': ManagedToolOverride(name='weather_forecast', description='Multi-day forecast.'),
}
```

Each entry follows the same rules as `ManagedTool`: name, description, and parameter descriptions are
overridable; the schema structure stays fixed in code. Renames route calls back to the original
implementation, and a rename that collides with another tool raises a `UserError`. `label`,
`targeting_key`, and `attributes` behave exactly as for `ManagedTool`, and `ManagedToolset.resolved`
exposes the active run's `ResolvedVariable`.
