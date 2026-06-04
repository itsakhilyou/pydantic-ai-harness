# Dynamic Workflow

Let an agent orchestrate several sub-agents from a single Python script it writes itself.

## The problem

Some tasks are too big for one agent in a single pass — a bug hunt across a whole service, a
migration that touches hundreds of files, a plan you want stress-tested from every angle. A
single long-running context drifts from the goal, stops early, and over-trusts its first answer.
Anthropic's [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents)
catalogs the patterns that fix this — prompt chaining, parallelization (sectioning and voting),
orchestrator-workers, evaluator-optimizer. They share a shape: a coordinator fans work out to
sub-agents, composes the results, and loops until they converge.

Delegating one tool call at a time expresses that shape badly. You *can* fan out — parallel tool
calls run several sub-agents at once — but the rest lives in the model's turns: chaining, voting,
and looping are each a round-trip, and every intermediate result flows back through the
orchestrator's context before it can act. [Code execution](https://www.anthropic.com/engineering/code-execution-with-mcp)
is Anthropic's fix one level down: write the control flow as code, so *"loops, conditionals, and
error handling can be done with familiar code patterns rather than chaining individual tool
calls"* and intermediate results *"stay in the execution environment"* instead of the model's
context. [Claude Code's dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
apply that to sub-agents: Claude writes a script that fans out independent sub-agents, has them
refute each other, and iterates until the answers converge.

## What it is

`DynamicWorkflow` is [Code Mode](../code_mode/README.md) with sub-agents as the callables. Code
Mode wraps the agent's own tools into one sandboxed `run_code` tool; `DynamicWorkflow` wraps a
catalog of named sub-agents into one `run_workflow` tool. The model writes a Python script — same
[Monty](https://github.com/pydantic/monty) sandbox — that composes the calls with ordinary
control flow (fan-out, chaining, voting, loops) in **one** step. Only the script's result returns
to the model; the intermediate results stay in the sandbox, the way Code Mode keeps tool results
out of the conversation.

The difference that makes it its own capability: each callable is a full `Agent.run`, not a plain
function — its own model loop, its own message history, its own tools and typed `output_type`.
That makes the "tools" non-deterministic, token-expensive, and themselves capable of
orchestrating — so `DynamicWorkflow` adds what Code Mode doesn't need: an exact ceiling on
sub-agent calls (`max_agent_calls`), usage shared across the tree, and a no-nesting guard.

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
        usage_limits=UsageLimits(request_limit=20),  # bounds the whole tree
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
parent's `deps` are forwarded, and by default its `usage` is shared, so a parent `usage_limits`
bounds the whole tree.

## Installation

`DynamicWorkflow` runs scripts in the [Monty](https://github.com/pydantic/monty) sandbox:

```bash
pip install "pydantic-ai-harness[code-mode]"
```

## Sub-agent catalog

Each `WorkflowAgent` in `agents` becomes an async function in the sandbox. Its `name` is the
function name (a valid Python identifier, unique across the workflow) and falls back to the
agent's own `name`. The model sees `description` as the function's description, falling back to
the name:

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
- **`max_agent_calls` bounds count, not cost.** For a token ceiling, set `usage_limits` on the
  parent `run()`; with `forward_usage=True` (default) it applies across the tree. A shared
  `usage_limits` is best-effort under concurrent fan-out — sub-agents can pass the check before
  any of them increments it — so use `max_agent_calls` for an exact ceiling.
- **`resource_limits`** — Monty limits on the *script's own* CPU and memory (default: 30s CPU,
  256 MB). `max_duration_secs` counts only sandbox CPU, not time awaiting sub-agents, so a runaway
  `while` loop is stopped without penalising slow sub-agents. Pass `{}` to disable.
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
    agents,                  # Sequence[WorkflowAgent] — required; pass a list to reveal at runtime
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    resource_limits=None,    # None -> safe backstop (30s CPU, 256 MB); {} -> no limits
    id=None,                 # required when defer_loading=True
    defer_loading=False,
)

WorkflowAgent(
    agent,                   # Agent — required
    name=None,               # sandbox function name; falls back to agent.name
    description=None,        # catalog description; falls back to agent.name
)
```

## Roadmap: orchestration state as data

Running the script on Monty opens a door a plain function call doesn't: a *suspended* Monty
program isn't just a paused function, it's a tiny serializable value you can **dump, reload, and
fork**.

```python
from pydantic_monty import Monty, load_snapshot

# Run a workflow up to its first external call, then snapshot the suspended state.
prog = Monty('shared = expensive_setup()\nchoice = pick_branch()\nshared + choice')
state = prog.start()          # suspends at the first external call (pick_branch)
blob = state.dump()           # the entire in-flight program state...
print(len(blob))              # ...is ~420 bytes

# Fork: reload the SAME snapshot N times and drive each down a different branch —
# the expensive prefix ran once, the branches diverge for free.
branch_a = load_snapshot(blob).resume({'return_value': 10})
branch_b = load_snapshot(blob).resume({'return_value': 1000})
```

That foundation (verified above: ~420-byte snapshots, real forks) is what the capability builds
toward:

- **Best-of-N from a shared prefix.** Build context once, fork the snapshot into N branches that
  each explore a different candidate — no re-running the setup per branch.
- **Durable, resumable workflows.** Persist the snapshot at each sub-agent suspension; after a
  crash or a redeploy in a fresh process, load it back and the workflow continues from exactly
  where it paused, every variable and partial result intact.
- **Cheap checkpoint/rewind** for goal-anchoring across long orchestrations.

Today `DynamicWorkflow` ships the synchronous fan-out / chaining / voting / loop patterns above.
Snapshot-based **fork and durable resume are planned** — see
[decisions and open problems](../../DYNAMIC_WORKFLOW_DECISIONS.md).

## Further reading

- [Code Mode](../code_mode/README.md) — the same sandbox, calling the agent's own tools instead of sub-agents.
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) ·
  [On-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
- [Monty](https://github.com/pydantic/monty)
