# Dynamic Workflow

Let an agent orchestrate several sub-agents from a single Python script it writes itself.

## Why

When an orchestrator coordinates sub-agents by calling them **one per tool call**, two costs add
up. Each step that depends on a previous result — score these drafts, pick the best, refine it,
loop until it passes — has to be its own model turn, because the orchestrator must see each result
before it can issue the next call. And every one of those intermediate results lands back in the
orchestrator's context, growing the prompt and pulling the model's attention off the goal.

`DynamicWorkflow` moves that coordination out of the model's turns and into code. The orchestrator
writes one Python script that calls the sub-agents, and ordinary control flow does the rest —
`asyncio.gather` to run them in parallel, a `max(...)` to choose a winner, a `for`/`while` to loop.
The script runs to completion in a **single** tool call, and only its final value returns to the
model; the intermediate drafts and scores stay in the sandbox as local variables.

> Score three drafts, pick the best, refine it. One-per-call, that's three sequential model turns
> after drafting — and all three drafts and all three scores travel back through the orchestrator's
> context to get there. As a script it's one tool call: the scoring, the `max(...)`, and the refine
> are plain Python, and the model only ever sees the winner.

The mechanism — control flow as code, with intermediate results kept out of the model's context —
is the one Anthropic describes in
[*code execution with MCP*](https://www.anthropic.com/engineering/code-execution-with-mcp); the
orchestration patterns it expresses (chaining, parallelization, orchestrator-workers,
evaluator-optimizer) are catalogued in
[*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents).
`DynamicWorkflow` applies both to sub-agents, in the spirit of
[Claude Code's dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code).

## What it is

You give `DynamicWorkflow` a **catalog of named sub-agents**. It exposes a single `run_workflow`
tool; the model calls it by writing a Python script in which each sub-agent is an `async` function.
The script runs in [Monty](https://github.com/pydantic/monty), a restricted-Python sandbox — it can
fan out with `asyncio.gather`, chain one agent's output into the next, vote across several, and
loop, all before returning. Only the script's result goes back to the model; the intermediate
results stay in the sandbox.

Each callable is a full `Agent.run`, not a plain function — its own model loop, its own message
history, its own tools and typed `output_type`. Because those "tools" are themselves
non-deterministic, token-expensive, and capable of orchestrating, `DynamicWorkflow` adds an exact
ceiling on sub-agent calls (`max_agent_calls`), usage shared across the tree, and a guard against
nesting workflows.

> If you know [Code Mode](../code_mode/README.md): this is the same sandbox and the same idea, with
> sub-agents as the callables instead of the agent's own tools.

## Runnable example

Copy-paste and run it (needs an Anthropic key and the `anthropic` package):

```bash
export ANTHROPIC_API_KEY=sk-...
uv run --with 'pydantic-ai-harness[code-mode]' --with anthropic --with logfire python wf.py
```

```python
# wf.py — an orchestrator that runs a generate → score → refine tournament in one tool call.
import asyncio

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness import DynamicWorkflow, WorkflowAgent

# Instrumentation: the trace shows the orchestrator turn, the `run_workflow` tool call (its
# `code` argument is the exact script the model wrote), and every sub-agent run nested under it.
logfire.configure(send_to_logfire='if-token-present', service_name='dynamic-workflow')
logfire.instrument_pydantic_ai()

MODEL = 'anthropic:claude-sonnet-4-6'  # or 'anthropic:claude-opus-4-8'


class Score(BaseModel):
    value: int  # 0-10
    reason: str


drafter = Agent(MODEL, name='drafter', instructions='Write one concise candidate answer to the task.')
critic = Agent(MODEL, name='critic', output_type=Score, instructions='Score the candidate 0-10 with a one-line reason.')
editor = Agent(MODEL, name='editor', instructions='Improve the given answer using the critique. Return only the answer.')

orchestrator = Agent(
    MODEL,
    instructions=(
        'Use run_workflow to: draft 3 candidate answers in parallel, score each with the critic, '
        'pick the highest-scoring one, then have the editor refine it using that critique. Return the refined answer.'
    ),
    capabilities=[
        DynamicWorkflow(
            agents=[
                WorkflowAgent(agent=drafter, description='Writes one candidate answer to a task.'),
                WorkflowAgent(agent=critic, description='Scores a candidate answer 0-10, returns {value, reason}.'),
                WorkflowAgent(agent=editor, description='Improves an answer given a critique.'),
            ],
        )
    ],
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
— a parallel tournament with typed scoring, selection, and a refine step, in a single turn:

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
# 3. Pick the winner and refine it using its critique — plain Python, no extra model turns.
best = max(range(len(drafts)), key=lambda i: scores[i]["value"])
await editor(task="Answer:\n" + drafts[best] + "\n\nCritique:\n" + scores[best]["reason"])
```

Every call is a real, isolated `Agent.run`, run concurrently by `asyncio.gather`. The handoff is
typed the Pydantic way — the critic returns a `Score`, so the script reads `scores[i]["value"]`
instead of parsing a string.

## In practice

Each sub-agent runs in isolation: its own message history, never the parent conversation. The
parent's `deps` are forwarded, and by default its `usage` accumulator is shared, so the whole
tree's spend is tallied in one place. For a hard cap on sub-agent runs use `max_agent_calls`; see
[Budget and safety](#budget-and-safety) for why a shared `usage_limits` is only best-effort.

## Installation

`DynamicWorkflow` runs scripts in the [Monty](https://github.com/pydantic/monty) sandbox:

```bash
pip install "pydantic-ai-harness[code-mode]"
```

## Sub-agent catalog

Each `WorkflowAgent` in `agents` becomes an async function in the sandbox. Its `name` is the
function name (a valid Python identifier, unique across the workflow) and falls back to the
agent's own `name`. The `description` is rendered as the function's docstring; set it to tell the
model what the sub-agent does and what to pass as `task`. Omit it and the model sees only the bare
signature:

```python
DynamicWorkflow(
    agents=[
        WorkflowAgent(agent=reviewer, description='Reviews a code change and returns a list of issues.'),
        WorkflowAgent(agent=summarizer, description='Condenses findings into a short summary.'),
    ],
)
```

The agent, its sandbox name, and its description travel together on one object — no second
mapping to keep in sync. The catalog is fixed at the start of each run, so it stays in the
prompt-cache prefix across turns.

### Revealing sub-agents at runtime

Pass `agents` as a **mutable `list`** and keep a reference to it (often via `deps`). Appending a
`WorkflowAgent` mid-run makes it callable on the next step:

```python
agents = [WorkflowAgent(agent=reviewer)]
orchestrator = Agent('openai:gpt-5', deps_type=MyDeps, capabilities=[DynamicWorkflow(agents=agents)])

# later, from the host or another tool — e.g. once a fixer agent is provisioned:
agents.append(WorkflowAgent(agent=fixer, description='Applies a fix for a reported issue.'))
```

The newcomer is announced to the model with a short message (its function signature) via the
auto-injected `PendingMessageDrainCapability`. The `run_workflow` description itself stays frozen
at the agents present when the run started — so even a runtime reveal never moves the prompt-cache
prefix.

`agents` is a `list` precisely so this works — you hold the reference and append. Reveal is
**append-only**: once a sub-agent has appeared it stays for the rest of the run; there is no way to
remove or hide it again. Plan the catalog as growing.

## Return values

A sub-agent function returns that agent's output serialized to a JSON-compatible value:

| Sub-agent `output_type` | Value in the sandbox |
| ----------------------- | -------------------- |
| `str` (default)         | the string           |
| a Pydantic model        | a `dict` (access fields as `r['field']`, not `r.field`) |
| list / scalar           | the list / scalar    |

The value of the script's last expression becomes the `run_workflow` result — do not `print()` it.

## Budget and safety

- **`max_agent_calls`** (default `50`) — an exact, host-enforced ceiling on sub-agent runs per
  parent run. It holds even under concurrent fan-out. When exhausted, the workflow returns a
  terminal message telling the model to conclude.
- **`max_agent_calls` bounds count, not cost.** For a token ceiling, set **`sub_agent_usage_limits`**
  — a `UsageLimits` applied to every sub-agent run. With `forward_usage=False` each sub-agent run is
  sequential, so its own limit is enforced exactly: a per-sub-agent `total_tokens_limit` of `T`
  together with `max_agent_calls` of `N` give a **hard worst-case ceiling of `N * T` tokens**. With
  `forward_usage=True` the parent's `usage` accumulator is shared so the whole tree's spend tallies
  in one place, and the limit is checked against that shared counter — a tree-wide cap, but
  best-effort under concurrent fan-out (sub-agents can pass the check before any of them increments
  it). The parent's own `usage_limits` on `run()` is **not** forwarded into sub-agents (`RunContext`
  doesn't expose it); it is re-checked only at the parent's request boundaries. Use `max_agent_calls`
  for an exact ceiling on sub-agent *runs*.
- **`resource_limits`** — Monty limits on the *script's own* CPU and memory (default: 30s CPU,
  256 MB). `max_duration_secs` counts only sandbox CPU, not time awaiting sub-agents, so a runaway
  `while` loop is stopped without penalising slow sub-agents. Pass `'unlimited'` to remove all
  limits; a partial dict (e.g. `{'max_memory': ...}`) is merged onto the backstop, overriding only
  the caps it names and leaving the others at their default.
- **Workflows do not nest.** A sub-agent that tries to start its own workflow is refused. Don't
  give the sub-agents in `agents` the `DynamicWorkflow` capability — they are leaves of the
  orchestration, not orchestrators.

## On-demand loading

`DynamicWorkflow` carries a fair amount of instruction text but isn't needed on most turns — a
good fit for [deferred loading](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities).
Set `defer_loading=True` with a stable `id` to collapse it to a one-line catalog entry until the
model loads it, paying near-zero tokens on turns that don't orchestrate:

```python
DynamicWorkflow(
    agents=[WorkflowAgent(agent=reviewer), WorkflowAgent(agent=summarizer)],
    id='workflow',
    defer_loading=True,
)
```

## Sandbox restrictions

The script runs in Monty, a Python subset:

- No classes, no third-party libraries.
- Importable standard-library modules: `asyncio`, `math`, `json`, `re`, `typing`.
- No wall-clock or timing primitives (`asyncio.sleep`, `datetime.now()`, the `time` module).
- `asyncio.gather(...)` runs sub-agents concurrently but does **not** support
  `return_exceptions=True`. A sub-agent that raises aborts the script (the model retries); it
  cannot be caught inside the script today.

## API

```python
DynamicWorkflow(
    agents,                  # list[WorkflowAgent] — required; append mid-run to reveal sub-agents
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    sub_agent_usage_limits=None,  # UsageLimits applied to each sub-agent run; None -> pydantic-ai default
    resource_limits=None,    # None -> backstop (30s CPU, 256 MB); 'unlimited' -> off; dict -> merged onto backstop
    id=None,                 # required when defer_loading=True
    defer_loading=False,
)

WorkflowAgent(
    agent,                   # Agent — required
    name=None,               # sandbox function name; falls back to agent.name
    description=None,        # function docstring shown to the model; omitted -> bare signature
)
```

## Planned: fork and durable resume

Running the script on Monty opens a door a plain function call doesn't: a *suspended* Monty program
is a tiny serializable value you can dump, reload, and **fork**. That is the foundation for two
patterns the capability is built toward but does **not** ship yet:

- **Best-of-N from a shared prefix** — build the expensive context once, then fork the snapshot
  into N branches that each explore a different candidate, without re-running the setup per branch.
- **Durable, resumable workflows** — persist the snapshot at each sub-agent suspension; after a
  crash or a redeploy in a fresh process, reload it and the workflow continues from exactly where
  it paused, every variable and partial result intact.

The Monty engine already supports this — a suspended program dumps to ~500 bytes, reloads, and
forks — but `DynamicWorkflow` does not expose it yet: today `run_workflow` runs the model's script
straight through to completion and returns the result. The synchronous fan-out / chaining / voting /
loop patterns shown above ship now; fork and durable resume are planned.

## Further reading

- [Code Mode](../code_mode/README.md) — the same sandbox, calling the agent's own tools instead of sub-agents.
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) ·
  [On-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
- [Monty](https://github.com/pydantic/monty)
