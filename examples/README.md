# DynamicWorkflow in practice

`DynamicWorkflow` is **inspired by [Claude Code's dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)**:
instead of delegating to one sub-agent per tool call, the model writes a single orchestration
script that fans work out, chains it, votes across it, and loops until done.

We take that idea further on two foundations:

- **[Monty](https://github.com/pydantic/monty) runs the script.** The orchestration is real,
  sandboxed Python — and a *suspended* Monty program is a tiny serializable value you can
  snapshot, fork, and resume. That turns "run a script" into "checkpoint, branch, and survive a
  crash."
- **The Pydantic way: typed all the way through.** Sub-agents are full Pydantic AI `Agent`s with
  Pydantic `output_type`s, a shared typed `deps` object threads the whole tree, and even the
  sandbox snapshot states are narrowed types. No stringly-typed handoffs.

And it stays **cache-stable by construction**: the orchestration tool defers its instructions
until needed, and new sub-agents can be revealed mid-run without ever changing the prompt-cache
prefix.

These two examples show it working:

| File | Runs | What it demonstrates |
| --- | --- | --- |
| [`dynamic_workflow_speculative_repair.py`](./dynamic_workflow_speculative_repair.py) | needs an Anthropic key | Typed handoffs, sub-agents with **confined capabilities**, **deferred loading**, **runtime sub-agent reveal** — a whole repair tournament in one tool call. |
| [`dynamic_workflow_fork_and_resume.py`](./dynamic_workflow_fork_and_resume.py) | no key, no agents | The Monty foundation: **fork a live orchestration** from a 508-byte snapshot and **resume it across processes**. |

## Example 1 — a self-verifying repair tournament

`dynamic_workflow_speculative_repair.py`. The orchestrator is handed a failing test and, in
**one** tool call, writes a script that:

1. runs a **scout** sub-agent to reproduce the failure and return a typed `Diagnosis`,
2. fans out three **fixer** sub-agents in parallel, each pursuing a different strategy and
   returning a typed `FixProposal` (it *proposes* a diff, it doesn't apply one — so the fan-out
   never collides),
3. has a **referee** sub-agent score each proposal, and
4. returns the highest-scoring, smallest diff — in ordinary Python control flow.

What makes it more than "spawn N agents":

- **Each leaf is a confined capability surface.** The scout gets a read-only `FileSystem` and a
  `Shell` allowlisted to `pytest`/`git`; the fixer gets `CodeMode` over a `FileSystem` that
  can't touch the test dir. Containment lives on each sub-agent, exactly where the blast radius
  is.
- **Typed all the way down.** `Diagnosis`, `FixProposal`, `Score` are Pydantic models; the
  script indexes real fields. A shared typed `deps` threads through every sub-agent.
- **Loads on demand, stays cache-stable.** `DynamicWorkflow` is `defer_loading=True` — about one
  line of prompt until the model actually orchestrates. And a heavyweight `security_audit`
  sub-agent is **revealed at runtime**, appended to the live catalog only when the diagnosis
  flags auth code — *without* changing the cached `run_workflow` description, so the prompt-cache
  prefix never moves.

- **One ceiling on the whole tree.** `max_agent_calls` is an exact, host-enforced cap that holds
  even under concurrent fan-out, alongside Monty CPU/memory limits on the script itself.

## Example 2 — fork a live orchestration, resume across a crash

`dynamic_workflow_fork_and_resume.py` runs against the real `pydantic_monty` API — no agents, no
key. Verified output:

```
snapshot at the decision point: 508 bytes
prefix executed: 1 time (the expensive context built once)
prefix executed after 3 forks: 1 time (still once — branches diverged for free)
  branch 'codemod': shared(100) + strategy -> 105
  branch 'rewrite': shared(100) + strategy -> 130
  branch 'shim':    shared(100) + strategy -> 112
winner: 'rewrite' (the host keeps it; the losing forks are discarded)
resumed from the persisted snapshot in a fresh load: {'after-restart': 107}
```

A suspended Monty program — locals and all — is ~half a kilobyte of bytes you can `dump`,
`load`, and `load` *again* to fork. The example drives the orchestration to a decision point,
snapshots the live state, then:

- **Best-of-N from a shared live prefix.** Build context once (map the repo, gather the
  diagnosis), snapshot, fork into N strategy branches. The proof it's shared and not recomputed:
  `prefix executed: 1 time` holds after three forks — the branches diverge for free.
- **Durable, cross-process resume.** The snapshot is just bytes — write it to a database (or a
  durable-execution engine like Temporal/DBOS) at each suspension, and after a crash or a redeploy
  in a fresh process you load those bytes back and the orchestration picks up exactly where it
  paused, with every variable and partial result intact.

Each `resume` step is narrowed to its snapshot type (`FunctionSnapshot` → `MontyComplete`), so
even the checkpoint machinery is typed. Tournament-from-prefix and durable resume are the
roadmap for the capability — this example exists to show the foundation is real and measured.

## In one line

Claude Code showed that moving the plan into a script beats turn-by-turn delegation.
`DynamicWorkflow` puts that script on Monty and does it the Pydantic way — typed end to end,
cache-stable, and built on a suspended state you can fork and persist.

## Further reading

- [`DynamicWorkflow` README](../pydantic_ai_harness/dynamic_workflow/README.md)
- [Monty](https://github.com/pydantic/monty)
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) ·
  [On-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
