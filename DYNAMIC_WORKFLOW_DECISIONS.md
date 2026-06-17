# DynamicWorkflow — decisions, issues, and open problems

Working notes for the `DynamicWorkflow` capability (branch `claude/dynamic-workflow`). Context:
a Monty-native take on Claude Code's "dynamic workflows" — the model writes a Python orchestration
script that spawns and coordinates sub-agents, instead of delegating one tool-call at a time.

Framing follows the primary sources (mirror these in public docs, don't invent our own pitch):
[*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents) for
the pattern vocabulary (chaining, parallelization/voting, orchestrator-workers, evaluator-optimizer);
[*Code execution with MCP*](https://www.anthropic.com/engineering/code-execution-with-mcp) for the
mechanism (control flow as code, intermediate results stay out of the model's context — *not* "fan-out
is impossible", which parallel tool calls already do); [*dynamic workflows in Claude Code*](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
for the lead angle (scale + adversarial convergence: independent sub-agents that refute each other and
iterate until they agree).

## What shipped in v1

A `DynamicWorkflow` capability that exposes one `run_workflow` tool over a catalog of named
sub-agents. The model writes Python (run in the Monty sandbox); each sub-agent is an async
function; `asyncio.gather` fans them out concurrently. Reuses the Monty execution loop extracted
into `pydantic_ai_harness/_monty_exec.py` (shared with `code_mode`). 100% branch coverage,
lint/pyright clean.

Runtime agent reveal also shipped, with **zero new public API**: `for_run` does `replace(self)`,
keeping the *same* `agents` list reference the caller passed, so appending a `WorkflowAgent` mid-run
is visible to the running toolset. Each step `get_tools` calls `_reveal_pending`, which diffs the
live list against `_by_name` (baseline + already-revealed), folds each newcomer in so `dispatch`
resolves it, and `ctx.enqueue`s an announcement. The description renders from a frozen
`_baseline_catalog` snapshotted at run start, so the cached prompt prefix never changes
(cache-stable by construction). `_reveal_pending` is sync and await-free between snapshotting
`agents` and mutating `_by_name`, so a concurrent `dispatch` never sees a half-revealed agent.
Caveat: `ctx.enqueue` is not durable, so the reveal path is non-durable for now (fine — the whole
capability is non-durable today); requires a `PendingMessageDrainCapability` in the run
(auto-injected by core) so the announcement drains.

## Decisions taken (and why)

| Decision | Rationale | Alternatives rejected |
| --- | --- | --- |
| **Named-agent registry** (`agents=[WorkflowAgent(agent=...)]`), each becomes a sandbox function | Cache-stable catalog (static listing stays in the prompt prefix); bounded blast radius; the model can't conjure arbitrary agents | Parameterized template (`agent(task, system_prompt=...)`) — more flexible but uncacheable and unbounded; deferred to maybe-v2 |
| **`agents` is a `Sequence[WorkflowAgent]`**, each bundling the agent + its sandbox `name` (falls back to `agent.name`) + optional `description` | One object carries the three facts that belong together, so there is no second `descriptions` map keyed by the same names to keep in sync. The only residual runtime check is name validity + uniqueness, which the type system can't express. Extensible — future per-agent knobs get a field, not a third parallel map. | Two parallel `Mapping`s (`agents` + `descriptions`) — required a runtime sync check and overloaded the dict key as both join-key and function name; rejected. `Mapping[str, WorkflowAgent]` — kills the sync bug but keeps the key-as-identifier overload |
| **Plain capability owning one `run_workflow` tool** via `get_toolset` | A `WrapperToolset`/`wrap_run` is bound once at run start and is a silent no-op for a capability loaded mid-run via `defer_loading` (verified). A plain tool works post-load. | WrapperToolset (like `code_mode`) — incompatible with deferral |
| **Defer-loadable** (`defer_loading=True` + `id`) | Orchestration instructions are heavy and rarely needed; collapse to a one-line catalog entry until the model loads it. Framework-enforced "free until needed", beating prompt-discipline. | Always-loaded (kept available, just not the default selling point) |
| **Non-durable v1** | Durability needs unsolved core/Monty work (below). Ship the synchronous patterns now. | Durable-from-start — blocks on core |
| **Sub-agent surface = `name(task: str) -> <output>`**, output serialized via `to_jsonable_python` | Simple, predictable; structured outputs arrive as dicts the script can index | Returning typed objects — needs Monty dataclass-registry work (below) |
| **Exact `max_agent_calls`** counted host-side in `dispatch` (check+increment with no `await` between → atomic under asyncio) | A real, exact ceiling that holds under concurrent fan-out, unlike a shared `usage_limits` (see open problems) | Relying on `usage_limits` alone |
| **Workflows do not nest** — a boolean `ContextVar` flag set during script execution; a sub-agent that tries to run a workflow is refused | Recursive orchestration is a footgun with no clear use; a hard "no nesting" rule is simpler than a configurable depth and removes a public knob (`max_depth`). The flag is reliable for asyncio (the only backend), which copies the context into each `asyncio.gather` task. Sub-agents must not be given this capability (documented). | Configurable `max_depth` counter (the original — pointless flexibility); no guard at all (silent recursion until the shared usage limit catches it) |
| **`resource_limits` resolved inline from a `None` sentinel** (not a field default) | `None` = "apply backstop", `{}` = "no limits", populated = custom. Keeps the backstop's actual numbers in one place (`_default_resource_limits` in the toolset) so the capability layer forwards `None` without knowing them; a `field(default_factory=...)` would duplicate the numbers across the capability/toolset layers or couple them. Also sidesteps a mutable-dict default. | Baking the limits into the field default (couples the two layers; `{}`-means-off less obvious) |
| **No `event_stream_handler` knob** for sub-agents | `agent.run()` already honors a sub-agent's own agent-level `event_stream_handler` (`handler or self.event_stream_handler`), so per-sub-agent streaming works with zero new surface. A forwarded orchestrator-level handler would override the sub-agent's own, interleave events from concurrent fan-out without attribution, and add API — defer until there's a concrete need. | Forwarding a unified handler to every sub-agent run now |
| **`resource_limits` on the Monty REPL** (default 30s CPU / 256 MB) | Backstop against a runaway script. `max_duration_secs` counts only sandbox CPU, not sub-agent wait (verified) — safe to set tight. | Bare `MontyRepl()` with no limits (the original; an unbounded-hang hole) |
| **Budget exhaustion returns a terminal result**, not `ModelRetry` | A retry can never succeed (budget is per-run) and would burn `max_retries` into a hard `UnexpectedModelBehavior` crash. A terminal message tells the model to conclude. | Raising `ModelRetry` (the original — crashed the run) |
| **Sub-agent errors sanitized** to `sub-agent 'x' raised <ErrorType>` | A raw host exception reaching the model leaks file paths / deps / agent reprs (verified leak). | Surfacing the full traceback |
| **`get_serialization_name() -> None`** | `agents` holds live `Agent` objects; the default (`cls.__name__`) would advertise it as YAML/`from_spec`-constructible and then raise. | Default (broken `from_spec`) |
| **Flat test functions** (no `class Test...`) | Maintainer preference. (Note: `code_mode`/`shell`/`filesystem` use class-based tests; this diverges intentionally.) | Class-based (the exemplar's style) |

## Issues faced and resolved

These came out of an adversarial review pass (correctness, API, security lenses) and were fixed:

- **No sandbox resource limits** → unbounded CPU/memory/hang, blocking the event loop. Fixed by adding `resource_limits` (configurable; safe default).
- **Budget exhaustion crashed the whole run** (retry that always re-fails). Fixed → terminal result.
- **Sub-agent error messages leaked host internals.** Fixed → sanitized to agent name + error type.
- **`from_spec` silently broken.** Fixed → opt out via `get_serialization_name`.
- **Parallel `agents` + `descriptions` maps could drift** (typo keys silently ignored), empty `agents` produced a useless tool. Fixed → collapsed the two maps into one `Sequence[WorkflowAgent]` (the sync bug is now structurally impossible), plus `__post_init__` validation for empty input, invalid/duplicate names, and a missing name (`UserError`).
- **Docstring claimed a `/docstring` description fallback that didn't exist.** Fixed.
- **Test backend**: the shared loop is asyncio-only; pinned `anyio_backend='asyncio'` (trio runs failed on `asyncio.ensure_future`).

## Dependency situation (needs its own PR)

`defer_loading` requires pydantic-ai **>= ~1.97**; the harness was pinned at **1.95.1**. Bumped to
**1.105.0** (has both `defer_loading` and `ctx.enqueue`). The bump has collateral that is **not**
related to this feature and should land separately:

- Core ToolSearch changed discovered-tool tracking from flipping `defer_loading=False` to an
  availability model (`ctx.available_tool_names`). This breaks `code_mode`'s native/sandbox
  fold-in. Currently **xfailed** (`test_tool_search_toolset_discovered_tool_in_run_code`) with a
  reason; the real fix belongs with the bump.
- Regenerated one `managed_prompt` inline snapshot (new tool-def fields from the bump).

**Recommendation:** land `bump + code_mode ToolSearch fix` as its own PR; `DynamicWorkflow` rides
on top. PR #243 (DouweM, `code_mode` `dynamic_catalog`) is the blueprint for cache-safe
append-reveal (catalog → instructions, new tools announced via `ctx.enqueue`) and is a *different*
problem from the `available_tool_names` regression — but it would also need rebasing onto >=1.105.

## Open problems (not yet solved)

1. **`usage_limits` is not enforced exactly under concurrent fan-out (core bug).** Check-then-
   increment in the run loop is split by an `await`, so N concurrent sub-agents can all pass the
   limit check before any increments — a shared `usage_limits` can overshoot (measured ~20×).
   Mitigated here by `max_agent_calls` (exact count), but a *token/cost* ceiling across the tree is
   still best-effort. We forward the parent's `usage` accumulator (shared `RunUsage`) so the tree's
   spend is accounted in one place, but `RunContext` exposes `usage` and **not** `usage_limits`, so
   the capability cannot forward the parent's actual limit value — sub-agents enforce only the
   default `UsageLimits(request_limit=50)` against the shared counter; the parent's configured limit
   is re-checked only at the parent's own request boundaries. Two core asks: (a) surface
   `usage_limits` on `RunContext` so it can be forwarded, and (b) atomic reserve-then-request for an
   exact concurrent ceiling. **File upstream.**

2. **Sub-agent exceptions aren't catchable inside the sandbox.** Monty delivers a deferred-future
   exception as a top-level `MontyRuntimeError`, not at the `await` site, so a script can't
   `try/except` a failing sub-agent — one failure aborts the whole workflow (the model retries).
   Also `asyncio.gather(..., return_exceptions=True)` is unsupported in the sandbox. Affects
   `code_mode` too. Needs a Monty change to raise deferred exceptions at the await site.

3. **The Monty loop blocks the event loop during pure-CPU sandbox code.** `feed_start`/`resume`
   run synchronously; a runaway script freezes the host until `max_duration_secs` (bounded, but a
   real freeze). Sub-agent waits do *not* block (they suspend back to the host). Fixing the pure-CPU
   case properly means running the VM off-thread — a larger change / possible core discussion.

4. **Durability is not implemented (the pinned v2 killer feature).** Non-durable in v1; design now
   de-risked by throwaway experiments against real Monty + a real DBOS runtime (local SQLite, true
   cross-process `os._exit` crash + `recover_pending_workflows`). Coming back to it later; findings:

   - **Durable resume does NOT need Monty snapshots.** A full crash-recovery cycle worked with zero
     snapshots: the `run_workflow` tool body (the parent durable step) re-runs from the top on
     recovery, the Monty VM re-derives the same dispatch sequence deterministically (Monty already
     bans wall-clock/random), and each completed sub-agent returns its journaled result — i.e.
     journal-replay, which the durable runtime gives for free via step/child-workflow memoization.
     Verified end-to-end with the actual `MontyExecutor`: across a crash after 2 of 4 sub-agents,
     every sub-agent ran **exactly once** and the workflow reached SUCCESS.
   - **So "what we need from Monty" is essentially nothing.** Determinism is already guaranteed, and
     `dump`/`load` (REPL snapshots) already round-trips through our exact async/`FutureSnapshot`
     pattern including the multi-pending `gather` case. Snapshots stay an *optional* optimization
     (skip re-running the cheap pure-CPU prefix) and the enabler for fork / best-of-N — already
     functional, so even that needs no new Monty work, only an API-stability commitment if adopted.
   - **What durability actually needs (all harness/runtime work, not Monty):**
     1. **Durable sub-agent runs** — dispatch must invoke `agent.run()` as a durable step (on DBOS:
        sub-agents wrapped as `DBOSAgent`, so each `.run()` is a memoized child workflow).
     2. **`global_sequential=True` is mandatory in durable mode.** Memoization keys are *positional*:
        child-workflow IDs are `{parent}-{function_id}`, `function_id` being a per-operation counter
        assigned in execution order (observed: `...-1/3/5/7`). Concurrent `asyncio.gather` fan-out
        assigns those IDs in a racy order → on replay the IDs shift → memoization misses →
        sub-agents re-run (duplicate cost/side-effects). The `global_sequential` path in
        `MontyExecutor` is exactly the fix. (Possible enhancement: explicit `SetWorkflowID` keyed by
        a stable hash of `(agent_name, task)` — but the `getResult` op's positional `function_id`
        still drifts under concurrency, so sequential stays the safe baseline.)
     3. **Runtime agent reveal / `ctx.enqueue` must be disabled in durable mode.** Root cause of
        "enqueue isn't durable": `enqueue` appends to an *in-memory* `pending_messages` queue, and
        `_reveal_pending` reads the *live mutable* `agents` list the host mutates out-of-band —
        neither is journaled, so the reveal is non-deterministic on replay (different announcements,
        divergent message history). Durable mode must freeze the catalog at run start (simplest) or
        make each reveal its own durable step.
   - **DBOS first; Temporal is harder.** DBOS journals steps without a deterministic-sandbox
     requirement, so it "just works". Temporal wraps *every tool call as one activity*
     (`durable_exec/temporal/_function_toolset.py`), so `run_workflow` would be a single atomic
     activity and activities can't spawn child activities — nested sub-agent durability is impossible
     there unless that tool's activity config is `False` so the body runs as workflow-sandbox code,
     then Monty (a compiled extension) runs inside Temporal's import-restricted sandbox (unverified).
     The sync snapshot API was chosen with this sandbox in mind, but it needs a real Temporal pass
     before claiming support.
   - **Already replay-safe** (re-derived cleanly on parent re-run, no special handling): the
     `max_agent_calls` counter and the `_in_workflow` ContextVar (reset and recomputed
     deterministically), and the script itself (the model request that produced it is a memoized
     step, so the same `code` re-runs).

5. **Structured output arrives as a dict, not the typed model.** `r['field']` works; `r.field` and
   `isinstance(r, Schema)` do not. Attribute access + typed round-trip is achievable host-side via
   Monty's `dataclass_registry` (+ a BaseModel→dataclass bridge); in-sandbox `isinstance` needs
   Monty to bind registry names into the sandbox namespace. v2.

6. **Error debuggability vs. leak-safety tradeoff.** Sanitizing sub-agent errors to type-only means
   the model loses the failure detail it might use to adapt. Acceptable for v1; a structured,
   redacted error channel would be better.
