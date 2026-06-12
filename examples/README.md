# DynamicWorkflow

`DynamicWorkflow` is the harness take on Anthropic's
[dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code).
It gives the orchestrating model one tool, `run_workflow`. Instead of delegating to one sub-agent
per model turn, the model writes a Python script that calls the sub-agents as async functions and
runs it in a sandbox: fan out with `asyncio.gather`, chain one result into the next, loop until
done. Only the script's final value returns to the model; the intermediate results stay in the
sandbox.

## How to use it

Register sub-agents with the capability. Each one becomes an async function the script can call,
named after the agent and documented by its `description`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.dynamic_workflow import DynamicWorkflow, WorkflowAgent

migrator = Agent('anthropic:claude-sonnet-4-6', name='migrator', instructions='Rewrite the given file from os.path to pathlib.')
reviewer = Agent('anthropic:claude-sonnet-4-6', name='reviewer', instructions='Review the given file; approve only if no os.path remains.')

orchestrator = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        DynamicWorkflow(
            agents=[
                WorkflowAgent(agent=migrator, description='Rewrites one file from os.path to pathlib.'),
                WorkflowAgent(agent=reviewer, description='Reviews one migrated file; returns approval and issues.'),
            ],
        )
    ],
)
```

The [`DynamicWorkflow` README](../pydantic_ai_harness/experimental/dynamic_workflow/README.md)
covers the full API: call budgets, usage forwarding, sandbox limits, and revealing sub-agents
mid-run.

## Why a script instead of turn-by-turn delegation

Coordination logic like "re-dispatch only the files that failed review, with the reviewer's issues
attached, for up to two more rounds" is ordinary Python. In a `run_workflow` script it looks like
this:

```python
import asyncio
import json

pending = ["area.py", "perimeter.py", "io_utils.py"]
tasks = {path: path for path in pending}  # each file's task text, grown with reviewer issues on retry
outcomes = {}

for _ in range(3):  # one initial pass plus up to two retries
    if not pending:
        break
    # Migrate every pending file at once, then review each result at once.
    reports = await asyncio.gather(*[migrator(task=tasks[path]) for path in pending])
    reviews = await asyncio.gather(*[reviewer(task=r["path"]) for r in reports])

    still_pending = []
    for report, review in zip(reports, reviews):
        path = report["path"]
        outcomes[path] = {**report, "approved": review["approved"], "issues": review["issues"]}
        if not review["approved"]:
            tasks[path] = path + "\n\nReviewer issues to fix:\n" + "\n".join(review["issues"])
            still_pending.append(path)
    pending = still_pending

json.dumps(list(outcomes.values()))
```

The model could run this loop itself, delegating one sub-agent per tool call. But every iteration
would then cost a model round-trip: one turn per migrate, per review, and per retry, with each
intermediate draft flowing back through the orchestrator's context. And each of those turns would
spend the model, a non-deterministic component, on a step that is plain control flow: compare the
reviews, rebuild the task list, dispatch again. Written as a script, the deterministic coordination
runs as code, the sub-agent calls inside it run concurrently, and the orchestrator spends one model
request for the whole tree, with only the final result entering its context.

## Further reading

- [`DynamicWorkflow` README](../pydantic_ai_harness/experimental/dynamic_workflow/README.md)
- [Monty](https://github.com/pydantic/monty)
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) ·
  [On-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
