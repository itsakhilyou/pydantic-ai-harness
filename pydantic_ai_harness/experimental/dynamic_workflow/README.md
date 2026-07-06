# Dynamic Workflow

Let one agent coordinate a whole team of sub-agents by writing a small Python script.

> **Experimental**
>
> Importing `pydantic_ai_harness.experimental.dynamic_workflow` emits a
> `HarnessExperimentalWarning`. The API can change in any release. When you have accepted that,
> silence it with:
>
> ```python
> import warnings
> from pydantic_ai_harness.experimental import HarnessExperimentalWarning
>
> warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
> ```

## The idea

Say you have a few specialist agents. One reviews code, one summarizes findings, one writes the
final note. Individually each is easy to call. The hard part is the choreography between them:
review three files at once, keep only the reports that found something, summarize those, and hand
the summary to the writer.

The usual way to do this is one tool call per step. The agent calls the reviewer, waits, reads the
result, calls it again, waits, and so on. Every intermediate result travels back into the agent's
context, and every step that depends on the previous one is a separate model turn.

`DynamicWorkflow` takes a different route. You hand it a catalog of named sub-agents, and it gives
the model a single tool, `run_workflow`. Inside that tool the model writes ordinary Python, where
each of your sub-agents is an `async` function it can call, loop over, and combine. The script runs
to completion in one tool call, and only its final value comes back to the model.

The choreography moves out of the conversation and into code.

> **Tip**
>
> If you have met [Code Mode](../../code_mode/README.md), this will feel familiar. It is the same
> sandbox and the same "write a script instead of many tool calls" idea. The difference is what the
> script gets to call: in Code Mode it calls the agent's own tools, here it calls whole sub-agents.

## Install

The script runs inside the [Monty](https://github.com/pydantic/monty) sandbox, so install the extra:

```bash
uv add "pydantic-ai-harness[dynamic-workflow]"
```

## Your first workflow

Let's build the smallest thing that works. Two sub-agents, one orchestrator.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.dynamic_workflow import DynamicWorkflow

reviewer = Agent('openai:gpt-5', name='reviewer', description='Reviews code for bugs.')
summarizer = Agent('openai:gpt-5', name='summarizer', description='Summarizes findings.')

orchestrator = Agent(
    'openai:gpt-5',
    capabilities=[DynamicWorkflow(agents=[reviewer, summarizer])],
)
```

That is the whole setup. Let's look at what each piece does:

1. `reviewer` and `summarizer` are plain agents. Nothing special about them, they are the same
   `Agent` you already know.
2. Their `name` becomes the function name the model calls in the script, so pick names that are
   valid Python identifiers.
3. Their `description` tells the model what each one is for. The model reads these to decide how to
   wire them together, so write them as if you were documenting a function.
4. `DynamicWorkflow(agents=[...])` bundles them into one capability and hands the orchestrator a
   single `run_workflow` tool.

## What the model does with it

When the orchestrator decides to use the tool, it does not call your sub-agents one at a time. It
writes a script. For a "review these two files and summarize" task, the script it writes looks like
this:

```python
import asyncio

reports = await asyncio.gather(
    reviewer(task="Review auth.py for bugs:\n<file contents>"),
    reviewer(task="Review parser.py for bugs:\n<file contents>"),
)
await summarizer(task="Summarize these review findings:\n" + "\n\n".join(reports))
```

A few things are worth pointing out here, because they are the core of how you use this capability:

- Each sub-agent is an `async` function. You call it with `await`.
- You pass the work as a single keyword argument, `task`. Always by keyword: write
  `reviewer(task="...")`, not `reviewer("...")`.
- `asyncio.gather(...)` runs the two reviews at the same time instead of one after the other.
- The last line's value becomes the result the model sees. The intermediate `reports` list never
  leaves the sandbox.

> **Info: what "call a sub-agent" actually means**
>
> Each call is a full `Agent.run`. It has its own model loop, its own message history, its own
> tools, and its own typed output. It is not a lightweight function, it is a real agent doing real
> work. Two consequences follow from that, and both matter when you write or debug workflows:
>
> - **Calls are isolated.** A sub-agent remembers nothing from an earlier call. Put everything it
>   needs into `task`.
> - **Calls cost tokens and take time.** That is why this capability gives you budgets, which we
>   get to below.

## Sub-agents can return structured data

A sub-agent returns whatever its `output_type` produces. By default that is a string. But give a
sub-agent a Pydantic model, and the script receives a `dict`:

```python
from pydantic import BaseModel

class Score(BaseModel):
    value: int
    reason: str

critic = Agent('openai:gpt-5', name='critic', description='Scores an answer 0-10.', output_type=Score)
```

Inside the script, the model reads the fields by subscript, the way you read a JSON object:

```python
result = await critic(task="Score this answer: ...")
result["value"]   # not result.value
```

> **Note**
>
> Structured output arrives as a plain `dict`, so fields are `result["value"]`, not
> `result.value`. This is the one place people trip. The type annotation the model sees in the
> catalog spells out the fields for it, so the model usually gets this right on its own.

## A complete, runnable example

Now let's put it together into something you can actually run. This orchestrator runs a small
tournament: draft three candidate answers in parallel, score each one, pick the winner, and refine
it. All of that happens in a single tool call.

You need an Anthropic key and the `anthropic` package:

```bash
export ANTHROPIC_API_KEY=sk-...
uv run --with 'pydantic-ai-harness[dynamic-workflow]' --with anthropic --with logfire python wf.py
```

```python
# wf.py
import asyncio

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.experimental.dynamic_workflow import DynamicWorkflow

# With Logfire configured, the trace shows the orchestrator turn, the run_workflow call (including
# the exact script the model wrote), and every sub-agent run nested underneath it.
logfire.configure(send_to_logfire='if-token-present', service_name='dynamic-workflow')
logfire.instrument_pydantic_ai()

MODEL = 'anthropic:claude-sonnet-4-6'  # or 'anthropic:claude-opus-4-8'


class Score(BaseModel):
    value: int  # 0-10
    reason: str


drafter = Agent(
    MODEL,
    name='drafter',
    description='Writes one candidate answer to a task.',
    instructions='Write one concise candidate answer to the task.',
)
critic = Agent(
    MODEL,
    name='critic',
    description='Scores a candidate answer 0-10, returns {value, reason}.',
    output_type=Score,
    instructions='Score the candidate 0-10 with a one-line reason.',
)
editor = Agent(
    MODEL,
    name='editor',
    description='Improves an answer given a critique.',
    instructions='Improve the given answer using the critique. Return only the answer.',
)

orchestrator = Agent(
    MODEL,
    instructions=(
        'Use run_workflow to: draft 3 candidate answers in parallel, score each with the critic, '
        'pick the highest-scoring one, then have the editor refine it using that critique. '
        'Return the refined answer.'
    ),
    capabilities=[DynamicWorkflow(agents=[drafter, critic, editor])],
)


async def main() -> None:
    result = await orchestrator.run(
        'Explain, for a new hire, why our service uses idempotency keys on payment requests.',
        usage_limits=UsageLimits(request_limit=20),
    )
    logfire.info('done', answer=result.output, requests=result.usage.requests)


asyncio.run(main())
```

Given just those three sub-agents and the instructions, the model writes and runs a script along
these lines:

```python
import asyncio

# 1. Draft three candidates at the same time.
drafts = await asyncio.gather(
    drafter(task="explain idempotency keys on payments"),
    drafter(task="explain idempotency keys on payments"),
    drafter(task="explain idempotency keys on payments"),
)
# 2. Score each one. Structured output arrives as {"value": int, "reason": str}.
scores = await asyncio.gather(*[critic(task="Score this answer:\n" + d) for d in drafts])
# 3. Pick the winner and refine it, all in plain Python, no extra model turns.
best = max(range(len(drafts)), key=lambda i: scores[i]["value"])
await editor(task="Answer:\n" + drafts[best] + "\n\nCritique:\n" + scores[best]["reason"])
```

Read that script and notice what did not happen. The three drafts, the three scores, and the
selection logic never traveled back through the orchestrator's context. The model issued one tool
call and got back one answer. The comparison, the `max(...)`, the string assembly are ordinary
Python running in the sandbox.

> **Tip**
>
> The [Logfire](https://pydantic.dev/logfire) trace is the best way to see what a workflow did.
> Each sub-agent run appears nested under the `run_workflow` span, and the span carries the exact
> `code` argument the model wrote, so you can read the script it actually ran.

## How results come back

The value of the script's last expression becomes the tool result. The model does not `print()` it.

For the common cases, that is all you need to know. If you want the exact rules, including what
happens when the script also prints for debugging, here they are:

> **Info: the precise return shape**
>
> | Sub-agent `output_type` | Value inside the script |
> | --- | --- |
> | `str` (the default) | the string |
> | a Pydantic model | a `dict`, read as `r['field']` |
> | list or scalar | the list or scalar |
>
> And how the final tool result is shaped:
>
> | The script... | The model receives |
> | --- | --- |
> | ends in a value, no print | that value directly (or `{}` if it is `None`) |
> | prints and ends in a value | `{"output": "<printed text>", "result": <value>}` |
> | prints and ends in `None` | `{"output": "<printed text>"}` |
>
> `print()` is for debug logging. It stringifies, so it is the wrong tool for returning structured
> data. Let the last expression carry the real result.

## Keeping it safe: budgets

A sub-agent is non-deterministic and costs tokens, and it can fan out into more sub-agents. So a
workflow needs two kinds of ceiling: a cap on *how many* sub-agent runs happen, and a cap on *how
much* they spend. `DynamicWorkflow` gives you both, plus a guard against runaway sandbox scripts.

### `max_agent_calls`: an exact count

```python
DynamicWorkflow(agents=[...], max_agent_calls=50)  # 50 is the default
```

This is a hard, host-enforced ceiling on the number of sub-agent runs in one parent run. It holds
exactly, even when the script fans out with `asyncio.gather`. When the budget runs out, the workflow
stops calling sub-agents and returns a terminal result telling the model to conclude with what it
has. That result includes the sub-agent results that did complete, so nothing already paid for is
wasted.

> **Note**
>
> `max_agent_calls` is the only knob that bounds the number of runs exactly. Reach for it when you
> need a guarantee. The token-based limits below are budgets, not guarantees.

### `sub_agent_usage_limits` and `forward_usage`: bounding cost

`sub_agent_usage_limits` is a `UsageLimits` applied to each sub-agent run. How tight a ceiling it
gives depends on `forward_usage`, which controls whether the whole tree shares one usage counter:

| `forward_usage` | Counter | What the limit means |
| --- | --- | --- |
| `True` (default) | the parent's `usage` is shared across the tree | a tree-wide cap, checked against the shared counter. Under concurrent fan-out it is best-effort: several sub-agents can pass the check before any of them adds to the count. |
| `False` | each sub-agent run counts on its own | per-run limits. A per-run `total_tokens_limit` of `T` with `max_agent_calls` of `N` bounds the tree to roughly `N * T` tokens. |

> **Warning**
>
> The `usage_limits` you pass to the parent `run()` is not forwarded into sub-agents. Core does not
> expose that limit value to the capability, so it is re-checked only at the parent's own request
> boundaries. If you want to bound sub-agents, set `sub_agent_usage_limits`. If you want an exact
> ceiling on the number of runs, use `max_agent_calls`.

### `resource_limits`: guarding the script itself

These limits guard the orchestration script's own memory and allocations, not the sub-agents it
calls. The default backstop is 256 MB and 50 million allocations, with no time limit.

```python
DynamicWorkflow(agents=[...], resource_limits={'max_duration_secs': 30})
```

There is deliberately no default wall-clock cap, and the reason is worth understanding:

> **Info: why no default time limit**
>
> The sandbox's duration timer counts total wall-clock time, and that includes the time the script
> spends awaiting sub-agents fanned out with `asyncio.gather`. A default cap would abort ordinary
> parallel workflows, not just runaway ones. So there is no default. Set `max_duration_secs`
> yourself to put a ceiling on a whole orchestration's runtime. It is also the only guard against a
> pure-CPU `while True` loop, which would otherwise block the event loop.
>
> Pass `'unlimited'` to remove all limits. A partial dict such as `{'max_memory': ...}` is merged
> onto the backstop, overriding only the caps you name and leaving the rest at their default.

### Workflows do not nest

A sub-agent cannot start its own workflow. If one tries, the nested `run_workflow` call returns a
terminal error instead of running.

> **Tip**
>
> The practical rule: do not give the sub-agents in your catalog the `DynamicWorkflow` capability.
> They are the leaves of the orchestration, not orchestrators themselves.

## Renaming a sub-agent: `WorkflowAgent`

By default, a sub-agent shows up in the script under its own `name` and `description`. Sometimes you
want a different name or a different description for one particular workflow, without editing the
agent itself. Wrap it in a `WorkflowAgent`:

```python
from pydantic_ai_harness.experimental.dynamic_workflow import WorkflowAgent

DynamicWorkflow(
    agents=[
        WorkflowAgent(
            reviewer,
            name='check',
            description='Checks one code change and returns actionable review findings.',
        ),
    ],
)
```

Now the model calls `check(task=...)` instead of `reviewer(task=...)`. Passing a bare agent is just
shorthand for wrapping it in a `WorkflowAgent` with no overrides.

## Adding sub-agents while a run is going: `reveal()`

The catalog is fixed when a run starts, which keeps it in the prompt-cache prefix across turns. But
sometimes you learn during a run that a new sub-agent should be available, say once a fixer agent has
been provisioned. Keep a reference to the `DynamicWorkflow` instance and call `reveal()`:

```python
workflow = DynamicWorkflow(agents=[reviewer])
orchestrator = Agent('openai:gpt-5', deps_type=MyDeps, capabilities=[workflow])

# later, from the host or from another tool:
workflow.reveal(fixer)
```

The revealed sub-agent becomes callable on the next step. The model learns about it through a short
announcement message carrying the new function's signature. The `run_workflow` description itself
stays frozen at the agents present when the run started, so even a runtime reveal never moves the
prompt-cache prefix.

> **Note**
>
> `reveal()` is append-only. Once a sub-agent appears it stays for the rest of the run, and there
> is no way to remove or hide it again. Plan the catalog as something that only grows.
>
> It validates right away: a missing name, an invalid identifier, a reserved keyword, or a name
> collision raises `UserError` at the call site. And if you share one `DynamicWorkflow` instance
> across concurrent runs, `reveal()` reaches all in-flight runs and joins the baseline for runs that
> start afterward.

## Loading it only when needed: `defer_loading`

`DynamicWorkflow` carries a fair amount of instruction text, and most turns do not need it. You can
keep it collapsed to a one-line entry until the model actually loads it, which pays close to zero
tokens on turns that never orchestrate:

```python
DynamicWorkflow(
    agents=[reviewer, summarizer],
    id='workflow',
    defer_loading=True,
)
```

`defer_loading=True` needs a stable `id`. See
[on-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
for the full picture.

## What runs in the sandbox

The script runs in Monty, a subset of Python. The subset is what makes the sandbox safe, so it is
worth knowing where the edges are:

- No class definitions, and no third-party libraries.
- Useful standard-library modules: `asyncio`, `math`, `json`, `re`, `typing`. Import what you use.
  Other modules are unavailable or stubbed.
- No wall-clock or timing primitives: no `asyncio.sleep`, no `datetime.now()`, no `time` module.
- `asyncio.gather(...)` runs sub-agents concurrently, but it does not support
  `return_exceptions=True`.

> **Warning: errors abort the whole script**
>
> A sub-agent that raises cannot be caught inside the script. One failure aborts the whole script,
> and the model retries it. So write scripts where sub-agents do not depend on catching each other's
> errors. If a script does fail after some sub-agents already finished, the retry prompt lists those
> completed results, so the model can reuse them as plain values instead of paying for the same
> calls again.

## What is coming

Running the script on Monty opens a door that a plain function call does not. A suspended Monty
program is a small serializable value you can dump, reload, and fork. Two patterns are built toward
this, and do not ship yet:

- **Best-of-N from a shared prefix.** Build the expensive context once, then fork the snapshot into
  N branches that each explore a different candidate, without re-running the setup per branch.
- **Durable, resumable workflows.** Persist the snapshot at each sub-agent suspension. After a crash
  or a redeploy in a fresh process, reload it and the workflow continues from exactly where it
  paused, with every variable and partial result intact.

The Monty engine already supports this. A suspended program dumps to roughly 500 bytes, reloads, and
forks. `DynamicWorkflow` does not expose it yet: today `run_workflow` runs the script straight
through and returns the result. Two smaller extensions are also planned: structured sub-agent inputs
(a `parameters` schema per `WorkflowAgent`, instead of only `task: str`) and first-class progress
streaming. Until then, set `event_stream_handler` on each sub-agent `Agent`, or use Logfire, to
watch sub-agent runs inside the one tool call.

## Recap

To let one agent orchestrate a team of sub-agents:

- Give `DynamicWorkflow` a catalog of named agents. It exposes one `run_workflow` tool.
- The model writes a Python script where each sub-agent is an `async` function, called with a
  keyword `task`. It fans out with `asyncio.gather`, chains with plain assignments, and loops with
  ordinary control flow. Only the final value returns.
- Each call is a real, isolated `Agent.run`, so pass full context in `task`, and read structured
  output as a `dict`.
- Bound the count with `max_agent_calls`, the cost with `sub_agent_usage_limits` plus
  `forward_usage`, and the script itself with `resource_limits`.
- Reshape the catalog with `WorkflowAgent`, grow it at runtime with `reveal()`, and hide it until
  needed with `defer_loading`.

## API

```python
DynamicWorkflow(
    agents,                       # Sequence[AbstractAgent | WorkflowAgent], required
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    sub_agent_usage_limits=None,  # UsageLimits per sub-agent run; None -> pydantic-ai default
    resource_limits=None,         # None -> backstop (256 MB, 50M allocs, no time cap);
                                  # 'unlimited' -> off; a dict is merged onto the backstop
    id=None,                      # required when defer_loading=True
    description=None,             # one-line catalog entry shown while deferred
    defer_loading=False,
)

workflow.reveal(agent)            # AbstractAgent | WorkflowAgent; validates before appending

WorkflowAgent(
    agent,                        # Agent, required, positional
    name=None,                    # sandbox function name; falls back to agent.name
    description=None,             # function docstring; falls back to agent.description
)
```

## Further reading

- [Code Mode](../../code_mode/README.md), the same sandbox, calling the agent's own tools instead of
  sub-agents.
- [Tool use via code](https://www.anthropic.com/engineering/code-execution-with-mcp) (Anthropic),
  the mechanism this applies to sub-agents.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
  (Anthropic), the orchestration patterns a script can express.
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) and
  [on-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities).
- [Monty](https://github.com/pydantic/monty), the sandbox.
</content>
</invoke>
