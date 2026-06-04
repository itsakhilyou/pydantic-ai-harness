"""The Monty foundation: fork a *live* orchestration and resume it across processes.

`DynamicWorkflow` is inspired by Claude Code's dynamic workflows — the model writes one script
that orchestrates many sub-agents instead of delegating one at a time. Running that script on
Monty takes it a step further, because a *suspended* Monty program is not just a paused
function: it is a tiny, serializable value.

You can `dump` it, `load` it back, and `load` the *same* snapshot many times to fork it. That
one foundation — orchestration state as data — buys two things plain script execution can't:

1. **Best-of-N from a shared live prefix.** Build the expensive context once (map the repo,
   gather the diagnosis), snapshot at the decision point, then fork into N branches that each
   explore a different strategy. The prefix — including everything already computed and sitting
   in local variables — is shared for free; only the branches diverge.

2. **Durable, cross-process resume.** The snapshot is just bytes, so you can write it to a
   database (or a durable-execution engine like Temporal/DBOS) at each suspension. After a crash
   or a redeploy in a different process, load those bytes back and the orchestration continues
   from exactly where it paused — every local variable and partial result intact, nothing to redo.

This script demonstrates both against the real `pydantic_monty` API — no agents or API key
needed. Run it:

    uv run --with 'pydantic-ai-harness[code-mode]' python examples/dynamic_workflow_fork_and_resume.py
"""

from __future__ import annotations

from pydantic_monty import ExternalReturnValue, FunctionSnapshot, Monty, MontyComplete, load_snapshot

# In a real DynamicWorkflow these external calls are sub-agent runs. Here they are plain Python so
# the example is self-contained and deterministic. `prefix_runs` proves the prefix executes ONCE.
prefix_runs = {'n': 0}


def map_the_repo() -> int:
    """Stand-in for the expensive shared prefix: a scout sub-agent building context."""
    prefix_runs['n'] += 1
    return 100  # "context score" the branches build on


def reach_decision_point() -> int:
    """The fork point — where the host decides to explore strategies in parallel."""
    return 0


def explore(strategy_bonus: int) -> int:
    """Stand-in for a branch's strategy-specific sub-agent work."""
    return strategy_bonus


# The orchestration the model writes. Note `shared` is a local computed in the prefix; after the
# fork every branch sees the SAME `shared` without recomputing it.
ORCHESTRATION = """
shared = map_the_repo()
reach_decision_point()
result = shared + explore()
result
"""


def drive_prefix() -> bytes:
    """Run the orchestration up to the decision point and snapshot the live state.

    Each `resume` returns the next suspension as the broad snapshot union, so we narrow with
    `isinstance` before driving on — which also documents where the program is paused.
    """
    state = Monty(ORCHESTRATION).start()  # suspends at the first external call: map_the_repo()
    assert isinstance(state, FunctionSnapshot)
    state = state.resume(
        ExternalReturnValue(return_value=map_the_repo())
    )  # shared = 100; pauses at reach_decision_point
    assert isinstance(state, FunctionSnapshot)
    return state.dump()  # the entire in-flight program, locals included


def best_of_n(snapshot: bytes, strategies: dict[str, int]) -> dict[str, int]:
    """Fork the one snapshot into N branches; each diverges from the shared prefix for free."""
    results: dict[str, int] = {}
    for name, bonus in strategies.items():
        state = load_snapshot(snapshot)  # the suspended prefix, reloaded from bytes
        assert isinstance(state, FunctionSnapshot)
        state = state.resume(ExternalReturnValue(return_value=reach_decision_point()))  # past the fork point
        assert isinstance(state, FunctionSnapshot)
        state = state.resume(ExternalReturnValue(return_value=explore(bonus)))  # branch-specific work
        assert isinstance(state, MontyComplete)
        results[name] = state.output  # the script's last expression
    return results


def main() -> None:
    snapshot = drive_prefix()
    print(f'snapshot at the decision point: {len(snapshot)} bytes')
    print(f'prefix executed: {prefix_runs["n"]} time (the expensive context built once)')

    # 1. Best-of-N from the shared prefix. Three strategies, one prefix.
    results = best_of_n(snapshot, {'codemod': 5, 'rewrite': 30, 'shim': 12})
    print(f'prefix executed after 3 forks: {prefix_runs["n"]} time (still once — branches diverged for free)')
    for name, score in results.items():
        print(f'  branch {name!r}: shared(100) + strategy -> {score}')
    winner = max(results, key=lambda k: results[k])
    print(f'winner: {winner!r} (the host keeps it; the losing forks are discarded)')

    # 2. Durable, cross-process resume. The snapshot is just bytes — persist it anywhere.
    #    Here we round-trip it to prove the reloaded program is the real, runnable state.
    revived = best_of_n(snapshot, {'after-restart': 7})
    print(f'resumed from the persisted snapshot in a fresh load: {revived}')
    print('  (in production: write these bytes to a DB or a durable-execution engine at each')
    print('   suspension; a crash reloads the orchestration exactly where it paused)')


if __name__ == '__main__':
    main()
