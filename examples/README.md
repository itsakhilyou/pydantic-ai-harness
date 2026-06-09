# DynamicWorkflow example

`DynamicWorkflow` gives the orchestrating model one tool, `run_workflow`. Instead of delegating to
one sub-agent per model turn, the model writes a single Python script that calls the sub-agents as
async functions — fan out with `asyncio.gather`, chain one into the next, loop until something
converges — in one tool call. Only the script's final value returns to the model; the intermediate
results stay in the sandbox.

[`dynamic_workflow.py`](./dynamic_workflow.py) is a deliberately small package — three short files,
so the run is fast and cheap — shaped to exercise every coordination pattern end to end in a single
`run_workflow` call: parallel fan-out, read/write confinement, an adversarial check, a feedback
loop, typed fan-in synthesis, and a Logfire trace over the entire tree. Each sub-agent is a full
Pydantic AI `Agent`, not a stand-in.

## What it does

The example plants a small package that still uses `os.path` into a fresh temp dir (so your repo is
never touched), then hands it to the orchestrator. In one `run_workflow` call the model writes a
script that:

1. migrates every file in parallel — one `migrator` sub-agent per file reads it, rewrites it to
   `pathlib`, and writes it back (the files differ, so the parallel edits never collide);
2. reviews every file in parallel — a read-only `reviewer` sub-agent approves the result only if no
   `os.path` survives and behaviour (including each function's return type) is preserved;
3. loops — any rejected file goes back to the migrator with the reviewer's issues appended to its
   task, for up to two extra rounds, then reports each file's final status; and
4. synthesizes — a `synthesizer` sub-agent turns the per-file outcomes into one typed report.

Each sub-agent is a Pydantic AI `Agent` with a typed `output_type` and exactly the filesystem access
it needs: the migrator may write, the reviewer is read-only, and the synthesizer has none.

## Why this needs `run_workflow`, not turn-by-turn delegation

The retry loop is the part you cannot express as one sub-agent per turn. Re-dispatching only the
files that failed review, round after round, is ordinary Python — `asyncio.gather`, a `while` loop,
a list of pending files:

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
            # Re-dispatch only this file next round, with the reviewer's issues appended so the
            # migrator has something new to act on.
            tasks[path] = path + "\n\nReviewer issues to fix:\n" + "\n".join(review["issues"])
            still_pending.append(path)
    pending = still_pending

# Fan in: one synthesizer turns every outcome into a single typed report.
report = await synthesizer(task=json.dumps(list(outcomes.values())))
report
```

Sub-agent outputs arrive in the script as plain dicts (`report["path"]`, `review["approved"]`), so
the loop is ordinary Python end to end. Delegated one sub-agent per tool call, the same task would
be a model round-trip per migrate, per review, and per retry — every file's draft flowing back
through the orchestrator's context and bloating it as it goes. As one script it is a single
`run_workflow` call: the migrations and reviews run concurrently inside it, the loop re-dispatches
only what failed (with the reviewer's issues attached), and the orchestrator's context only ever
gains the final report. The example prints `result.usage.requests` so you can see the orchestrator
made one model request, not one per sub-agent.

## Run it

From a checkout of this repo (the `--extra code-mode` refers to this project's optional dependency):

```bash
export ANTHROPIC_API_KEY=sk-...
export LOGFIRE_TOKEN=...   # optional, for a shareable trace of the whole tree
uv run --extra code-mode --with anthropic --with logfire \
    python examples/dynamic_workflow.py
```

To run it outside this repo, pull the extra from the published package instead:
`uv run --with 'pydantic-ai-harness[code-mode]' --with anthropic --with logfire python dynamic_workflow.py`.

The example prints the exact script the model wrote, then every migrated file, then the typed report
(representative output, yours will vary):

```text
The orchestrator wrote this script and ran it in ONE tool call:

    import asyncio, json
    pending = ["area.py", "perimeter.py", "io_utils.py"]
    ...

Migrated files:

  area.py:
    from pathlib import Path
    def area_table_path(name):
        return str(Path(__file__).parent / 'tables' / f'{name}.csv')
    ...

The orchestrator returned a typed MigrationSummary in 1 request(s):

  Migrated all 3 files from os.path to pathlib; every file passed review.

  - area.py: approved
  - perimeter.py: approved
  - io_utils.py: approved
```

`result.usage.requests` is `1`: the orchestrator made a single model request, and the per-file
migrate/review/retry work all happened inside that one `run_workflow` call. With `LOGFIRE_TOKEN` set,
one trace covers that orchestrator turn, the `run_workflow` call (whose `code` argument is the exact
script the model wrote), and every nested migrator, reviewer, and synthesizer run. With no Anthropic
key the example still plants the sources and prints the task, so you can see the setup offline.

## In one line

`DynamicWorkflow` moves sub-agent coordination from turn-by-turn delegation into one script the
model writes and runs on Monty — typed handoffs, a cache-stable prompt, and a hard worst-case token
ceiling across the whole tree. (Snapshot-based fork and durable resume are planned — see the
[`DynamicWorkflow` README](../pydantic_ai_harness/experimental/dynamic_workflow/README.md).)

## Further reading

- [`DynamicWorkflow` README](../pydantic_ai_harness/experimental/dynamic_workflow/README.md)
- [Monty](https://github.com/pydantic/monty)
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) ·
  [On-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
