# DynamicWorkflow examples

`DynamicWorkflow` gives the orchestrating model one tool, `run_workflow`. Instead of delegating to
one sub-agent per model turn, the model writes a single Python script that calls the sub-agents as
async functions — fan out with `asyncio.gather`, chain one into the next, vote across several, loop
— in **one** tool call. Only the script's final value returns to the model; the intermediate
results stay in the sandbox.

Both examples are real tasks on real code, each a full Pydantic AI `Agent` per sub-agent — no toy
stand-ins.

| File | What it shows | Needs |
| --- | --- | --- |
| [`dynamic_workflow_audit.py`](./dynamic_workflow_audit.py) | **Audit a whole package in one tool call** — a reviewer per file in parallel, a verifier that *refutes* each finding, then a synthesizer that ranks the survivors. Read-only, and Logfire-traced. | an Anthropic key |
| [`dynamic_workflow_migrate.py`](./dynamic_workflow_migrate.py) | **Migrate a package, in one tool call** — one migrator per file in parallel rewrites `os.path` to `pathlib`, an adversarial reviewer checks each, and a retry loop runs until every file converges. Real edits, in a throwaway temp dir. | an Anthropic key |

## audit — orchestrate a whole codebase in one tool call

`dynamic_workflow_audit.py` shows the pattern the capability is built for: **scale plus adversarial
convergence**. Point it at any Python package — it only reads. In one `run_workflow` call the model
writes a script that:

1. reviews every file in parallel — one **reviewer** sub-agent each — collecting typed findings;
2. **refutes** every finding in parallel — a separate **verifier** re-reads the code, and a finding
   survives only if it holds up (this kills the false positives a single pass waves through); then
3. hands the survivors to a **synthesizer** that dedupes and ranks them into one report.

Concretely — here is the script a **Claude model actually wrote** for this task, given the exact
sub-agent catalog and instructions the orchestrator receives (reproduced verbatim, and verified to
execute in the Monty sandbox). The whole audit, in one turn:

```python
import asyncio
import json

files = ["__init__.py", "_capability.py", "_toolset.py"]

# 1. Review every file at once — one reviewer sub-agent per file.
reviews = await asyncio.gather(*[reviewer(task=f) for f in files])

findings = []
for review in reviews:
    if review:
        findings.extend(review)

# 2. Refute every finding at once — it survives only if an independent verifier confirms it.
verdicts = await asyncio.gather(
    *[verifier(task=json.dumps(finding)) for finding in findings]
)
confirmed = [
    finding
    for finding, verdict in zip(findings, verdicts)
    if verdict["confirmed"]
]

# 3. Rank the survivors into one report — the only value the orchestrator ever sees.
report = await synthesizer(task=json.dumps(confirmed))
report
```

Why this is the amazing part — the same audit delegated one sub-agent per tool call would be, for
12 files: ~12 review turns, then ~12 verify turns, then a synthesis turn — **25+ model round-trips**,
with every file's findings flowing back through the orchestrator's context and bloating it as it
goes. As one script it is **a single `run_workflow` call**: the 12 reviews and 12 verifications run
concurrently inside it, and the orchestrator's context only ever gains the final report. More files
makes the gap wider, not the orchestrator's job harder.

Run it with a key (and `LOGFIRE_TOKEN` for a shareable trace of the whole tree — the orchestrator
turn, the `run_workflow` call, and every nested reviewer / verifier / synthesizer run):

```bash
export ANTHROPIC_API_KEY=sk-...
export LOGFIRE_TOKEN=...   # optional, for a shareable public trace
uv run --extra code-mode --with anthropic --with logfire \
    python examples/dynamic_workflow_audit.py path/to/some/package
```

## migrate — orchestration that writes

`dynamic_workflow_migrate.py` is where the agents *change* code, not just read it. It plants a small
package that still uses `os.path` into a fresh temp dir (so your repo is never touched), then hands
it to the orchestrator. In one `run_workflow` call the model writes a script that:

1. fans out a **migrator** sub-agent per file, in parallel — each reads its file, rewrites it to
   `pathlib`, and writes it back (the files differ, so the parallel edits never collide);
2. fans out a read-only **reviewer** sub-agent per file that *adversarially* checks the result —
   approve only if no `os.path` survives and behaviour is preserved; and
3. loops — any rejected file goes back to the migrator with the reviewer's issues, until the whole
   package converges.

That whole plan — parallel fan-out, an adversarial check, and a retry loop over a list of pending
files — is ordinary Python in a single model turn. The example prints the exact script the model
wrote and every migrated file, so you can see the orchestration that happened:

```bash
export ANTHROPIC_API_KEY=sk-...
uv run --extra code-mode --with anthropic python examples/dynamic_workflow_migrate.py
```

## In one line

`DynamicWorkflow` moves sub-agent coordination from turn-by-turn delegation into one script the
model writes and runs on Monty — typed handoffs, a cache-stable prompt, and an exact budget across
the whole tree. (Snapshot-based fork and durable resume are planned — see the
[`DynamicWorkflow` README](../pydantic_ai_harness/dynamic_workflow/README.md).)

## Further reading

- [`DynamicWorkflow` README](../pydantic_ai_harness/dynamic_workflow/README.md)
- [Monty](https://github.com/pydantic/monty)
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) ·
  [On-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
