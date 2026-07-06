# Dynamic Workflow

Let an agent orchestrate several sub-agents from a single Python script it writes itself.

> **Experimental.** Importing `pydantic_ai_harness.experimental.dynamic_workflow` emits a
> `HarnessExperimentalWarning`: the API may change in any release. Silence it with
> `warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)` once you've accepted that.

You give `DynamicWorkflow` a catalog of named sub-agents. It exposes one tool, `run_workflow`; the
model calls it by writing a Python script in which each sub-agent is an `async` function. The
script runs to completion in a single tool call, and only its return value goes back to the model.

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

Each callable is a full `Agent.run` -- its own model loop, message history, tools, and typed
`output_type` -- not a plain function. Inside the script the model composes them with ordinary
control flow: `asyncio.gather` to fan out, a `max(...)` to pick a winner, a `for`/`while` to loop.

> If you know [Code Mode](../../code_mode/README.md): this is the same sandbox and the same idea, with
> sub-agents as the callables instead of the agent's own tools.

## Why

Coordinating sub-agents one-per-tool-call has two costs. Any step that depends on a previous result
-- score these drafts, pick the best, refine it, loop until it passes -- must be its own model
turn, because the orchestrator has to see each result before issuing the next call. And every
intermediate result lands back in the orchestrator's context, growing the prompt and pulling the
model off the goal.

`DynamicWorkflow` moves that coordination out of the model's turns and into a script. The scoring,
the selection, and the refine step are plain Python that runs in one tool call; the intermediate
drafts and scores stay in the sandbox as local variables, and only the final value returns to the
model. A generate-score-refine tournament that is three sequential turns one-per-call (with every
draft and score traveling back through context) becomes a single turn where the model sees only the
winner.

This is the mechanism Anthropic describes in
[*code execution with MCP*](https://www.anthropic.com/engineering/code-execution-with-mcp), applied
to sub-agents. The orchestration patterns the script can express (chaining, parallelization,
orchestrator-workers, evaluator-optimizer) are catalogued in
[*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents), in
the spirit of
[Claude Code's dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code).

Because each callable is a real agent run -- non-deterministic, token-expensive, and itself capable
of orchestrating -- `DynamicWorkflow` adds an exact ceiling on sub-agent calls (`max_agent_calls`),
usage shared across the tree, and a guard against nesting workflows. See
[Budget and safety](#budget-and-safety).

## Installation

`DynamicWorkflow` runs scripts in the [Monty](https://github.com/pydantic/monty) sandbox:

```bash
uv add "pydantic-ai-harness[dynamic-workflow]"
```

## Runnable example

Copy-paste and run it (needs an Anthropic key and the `anthropic` package):

```bash
export ANTHROPIC_API_KEY=sk-...
uv run --with 'pydantic-ai-harness[dynamic-workflow]' --with anthropic --with logfire python wf.py
```

```python
# wf.py -- an orchestrator that runs a generate -> score -> refine tournament in one tool call.
import asyncio

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.experimental.dynamic_workflow import DynamicWorkflow

# Instrumentation: the trace shows the orchestrator turn, the `run_workflow` tool call (its
# `code` argument is the exact script the model wrote), and every sub-agent run nested under it.
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
        'pick the highest-scoring one, then have the editor refine it using that critique. Return the refined answer.'
    ),
    capabilities=[DynamicWorkflow(agents=[drafter, critic, editor])],
)


async def main() -> None:
    result = await orchestrator.run(
        'Explain, for a new hire, why our service uses idempotency keys on payment requests.',
        usage_limits=UsageLimits(request_limit=20),  # checked at the parent's request boundaries
    )
    logfire.info('done', answer=result.output, requests=result.usage.requests)


asyncio.run(main())
```

Given only the three sub-agents and `run_workflow`, the model writes and runs a script like this
-- a parallel tournament with typed scoring, selection, and a refine step, in a single turn:

```python
import asyncio

# 1. Draft three candidates concurrently.
drafts = await asyncio.gather(
    drafter(task="explain idempotency keys on payments"),
    drafter(task="explain idempotency keys on payments"),
    drafter(task="explain idempotency keys on payments"),
)
# 2. Score them concurrently. Structured output arrives as a dict: {"value": int, "reason": str}.
scores = await asyncio.gather(*[critic(task="Score this answer:\n" + d) for d in drafts])
# 3. Pick the winner and refine it using its critique -- plain Python, no extra model turns.
best = max(range(len(drafts)), key=lambda i: scores[i]["value"])
await editor(task="Answer:\n" + drafts[best] + "\n\nCritique:\n" + scores[best]["reason"])
```

Every call is a real, isolated `Agent.run`, run concurrently by `asyncio.gather`. The handoff is
typed the Pydantic way: the critic returns a `Score`, so the script reads `scores[i]["value"]`
instead of parsing a string.

## Sub-agent catalog

Each raw `Agent` or `WorkflowAgent` in `agents` becomes an async function in the sandbox. Its
catalog name is a valid Python identifier, unique across the workflow. A wrapper `name` wins first,
then the agent's own `name`. The catalog description is rendered as the function's docstring:
wrapper `description`, then agent `description`, then no docstring, so the model sees only the
signature plus any return schema. The return annotation is rendered from the sub-agent's
`output_type`; Pydantic model outputs include TypedDict-style field definitions.

```python
DynamicWorkflow(
    agents=[reviewer, summarizer],
)
```

Use `WorkflowAgent` when this workflow needs a different sandbox function name or description than
the agent's own metadata:

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

The catalog is fixed at the start of each run, so it stays in the prompt-cache prefix across turns.

### Revealing sub-agents at runtime

Keep a reference to the `DynamicWorkflow` instance (often via `deps`) and call `reveal()` with a
raw agent or a `WorkflowAgent`. The revealed sub-agent becomes callable on the next step:

```python
workflow = DynamicWorkflow(agents=[reviewer])
orchestrator = Agent('openai:gpt-5', deps_type=MyDeps, capabilities=[workflow])

# later, from the host or another tool -- e.g. once a fixer agent is provisioned:
workflow.reveal(fixer)
```

`reveal()` validates eagerly. A missing name, invalid Python identifier, reserved keyword, or name
collision raises `UserError` at the call site.

The newcomer is announced to the model with a short message (its function signature) via the
auto-injected `PendingMessageDrainCapability`. The `run_workflow` description itself stays frozen
at the agents present when the run started, so even a runtime reveal never moves the prompt-cache
prefix.

If one `DynamicWorkflow` instance is shared across concurrent runs, `reveal()` reveals to all
in-flight runs and joins the baseline catalog for runs that start afterwards.

Reveal is append-only: once a sub-agent has appeared it stays for the rest of the run; there is no
way to remove or hide it again. Plan the catalog as growing.

## Return values

A sub-agent function returns that agent's output serialized to a JSON-compatible value:

| Sub-agent `output_type` | Value in the sandbox |
| ----------------------- | -------------------- |
| `str` (default)         | the string           |
| a Pydantic model        | a `dict` (access fields as `r['field']`, not `r.field`) |
| list / scalar           | the list / scalar    |

The value of the script's last expression becomes the `run_workflow` result -- do not `print()` it.
The exact shape has three cases: no print returns the value directly, or `{}` when the value is
`None`; print plus a non-`None` value returns `{"output": "<printed text>", "result": <last
expression>}`; print plus `None` returns `{"output": "<printed text>"}`.

## Budget and safety

Sub-agents are non-deterministic and token-expensive, so a workflow needs both a ceiling on *how
many* run and a ceiling on *how much* they spend.

**Count: `max_agent_calls`** (default `50`) is an exact, host-enforced ceiling on sub-agent runs
per parent run. It holds even under concurrent fan-out. When exhausted, the workflow returns a
terminal `{"error": ...}` result telling the model to conclude. The result always includes
`last_error`, holding the displayed sandbox error; under concurrent batches, that display may come
from an unrelated failure from the same batch. This is the only knob that bounds the number of runs
exactly. Completed sub-agent results from the failed script are returned under `completed` so the
model can use them when concluding.

**Cost: `sub_agent_usage_limits`** is a `UsageLimits` applied to every sub-agent run. How tight a
ceiling it gives depends on `forward_usage`:

| `forward_usage` | Usage counter | Limit semantics |
| --------------- | ------------- | --------------- |
| `False` | each sub-agent run has its own usage counter; requests inside one run happen one at a time | Request limits are enforced per run. A per-run `total_tokens_limit` of `T` with `max_agent_calls` of `N` bounds the tree to roughly `N * T` tokens; each run can overshoot by the final response, because core checks token limits after a response arrives. |
| `True` (default) | the parent's `usage` accumulator is shared across the whole tree | The limit is checked against the shared counter -- a tree-wide cap, but best-effort under concurrent fan-out (sub-agents can pass the check before any of them increments it). |

The parent's own `usage_limits` on `run()` is **not** forwarded into sub-agents (`RunContext`
doesn't expose it); it is re-checked only at the parent's own request boundaries. For an exact
ceiling on sub-agent *runs*, reach for `max_agent_calls`.

If a non-budget runtime error aborts a script after some sub-agent calls completed, the retry
prompt lists those completed results so the model can reuse them as literals instead of spending the
same calls again.

**Sandbox: `resource_limits`** are Monty limits on the *script's own* memory/allocations (default:
256 MB, 50M allocations). There is deliberately **no default `max_duration_secs`**: the sandbox's
duration timer counts total wall-clock *including* time awaiting sub-agents fanned out with
`asyncio.gather`, so a default cap would abort ordinary parallel workflows, not just a runaway. Set
one explicitly to bound a whole orchestration's runtime (it's also the only guard against a
pure-CPU `while True`). Pass `'unlimited'` to remove all limits; a partial dict (e.g.
`{'max_memory': ...}`) is merged onto the backstop, overriding only the caps it names.

**Nesting: workflows do not nest.** A sub-agent that tries to start its own workflow is refused.
The nested workflow tool call returns a terminal `{"error": ...}` result instead of retrying.
Don't give the sub-agents in `agents` the `DynamicWorkflow` capability -- they are leaves of the
orchestration, not orchestrators.

## On-demand loading

`DynamicWorkflow` carries a fair amount of instruction text but isn't needed on most turns -- a
good fit for [deferred loading](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities).
Set `defer_loading=True` with a stable `id` to collapse it to a one-line catalog entry until the
model loads it, paying near-zero tokens on turns that don't orchestrate:

```python
DynamicWorkflow(
    agents=[reviewer, summarizer],
    id='workflow',
    defer_loading=True,
)
```

## Sandbox restrictions

The script runs in Monty, a Python subset:

- No classes, no third-party libraries.
- Useful standard-library modules: `asyncio`, `math`, `json`, `re`, `typing`. Other modules are
  unavailable or stubbed; don't rely on them.
- No wall-clock or timing primitives (`asyncio.sleep`, `datetime.now()`, the `time` module).
- `asyncio.gather(...)` runs sub-agents concurrently but does **not** support
  `return_exceptions=True`. A sub-agent that raises aborts the script (the model retries); it
  cannot be caught inside the script today.

## API

```python
DynamicWorkflow(
    agents,                  # Sequence[AbstractAgent | WorkflowAgent] -- required
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    sub_agent_usage_limits=None,  # UsageLimits applied to each sub-agent run; None -> pydantic-ai default
    resource_limits=None,    # None -> backstop (256 MB, 50M allocs, no time cap); 'unlimited' -> off;
                             # WorkflowResourceLimits dict -> merged onto the backstop
    id=None,                 # required when defer_loading=True
    description=None,        # one-line catalog entry shown while deferred
    defer_loading=False,
)

workflow.reveal(agent)       # AbstractAgent | WorkflowAgent; validates before appending

WorkflowAgent(
    agent,                   # Agent -- required, positional
    name=None,               # sandbox function name; falls back to agent.name
    description=None,        # function docstring; falls back to agent.description, then bare signature
)
```

## Planned: fork and durable resume

Running the script on Monty opens a door a plain function call doesn't: a *suspended* Monty program
is a small serializable value you can dump, reload, and **fork**. That is the foundation for two
patterns the capability is built toward but does **not** ship yet:

- **Best-of-N from a shared prefix** -- build the expensive context once, then fork the snapshot
  into N branches that each explore a different candidate, without re-running the setup per branch.
- **Durable, resumable workflows** -- persist the snapshot at each sub-agent suspension; after a
  crash or a redeploy in a fresh process, reload it and the workflow continues from exactly where
  it paused, every variable and partial result intact.

The Monty engine already supports this -- a suspended program dumps to ~500 bytes, reloads, and
forks -- but `DynamicWorkflow` does not expose it yet: today `run_workflow` runs the model's script
straight through to completion and returns the result. The synchronous fan-out / chaining / voting /
loop patterns shown above ship now; fork and durable resume are planned.

Two smaller extensions are also planned:

- **Structured sub-agent inputs** -- today the sandbox function contract is the single
  `task: str` keyword; a `parameters` schema per `WorkflowAgent` is the planned extension.
- **First-class progress streaming** -- today, set `event_stream_handler` on each sub-agent
  `Agent`, or use Logfire instrumentation, to observe sub-agent runs inside the one tool call.

## Further reading

- [Code Mode](../../code_mode/README.md) -- the same sandbox, calling the agent's own tools instead of sub-agents.
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) ·
  [On-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
- [Monty](https://github.com/pydantic/monty)
