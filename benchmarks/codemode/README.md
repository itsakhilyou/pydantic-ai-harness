# CodeMode + pydantic-monty verification/benchmark harness

A reusable local harness that drives a real `Agent(capabilities=[CodeMode(...)])` through scripted
`run_code` calls against whatever `pydantic-monty` is installed, checks that tool-bridging produces
the expected results, and times each scenario. Use it to verify a monty upgrade or a harness change
and to A/B two builds.

It exercises the full path -- `Agent.run` -> `CodeMode` -> monty sandbox -> host tool dispatch ->
`Agent` -- not just the toolset in isolation, so a green run means the round trip works end to end.

## What it covers

`scenarios.py` defines the suite. Current scenarios:

- `sync_call`, `multiple_tools` -- sync tool dispatch from inside the sandbox
- `async_gather` -- concurrent async tool calls via `asyncio.gather`
- `print_and_result` -- the `{output, result}` return shape
- `repl_state` -- REPL state persisting across two `run_code` calls
- `syntax_error_retry`, `type_error_retry`, `runtime_error_retry` -- the three `ModelRetry` channels

Add a scenario by appending a `Scenario` to `SCENARIOS` in `scenarios.py`. Each `Step` is one
`run_code` call with an optional `expect_return` (the tool-return content the model would observe)
or `expect_retry` (a substring of the retry message). `bench.py` needs no changes.

## Running

The harness runs on whatever env you point at it. Because monty betas ship wheels only up to
cp313, use a Python 3.13 environment for beta builds:

```bash
# Build a 3.13 env with the monty beta (add UV_OFFLINE=1 to resolve from the uv cache only)
benchmarks/codemode/setup_env.sh .venv-monty-beta 3.13

# Run the suite
.venv-monty-beta/bin/python benchmarks/codemode/bench.py
```

Options:

- `--json PATH` -- write machine-readable results (monty version, per-scenario pass/wall_ms/calls)
- `--filter SUBSTR` -- only run scenarios whose name contains `SUBSTR`
- `--repeat N` -- run the suite N times and keep the fastest wall time per scenario (steadier numbers)

Exit code is non-zero if any scenario fails, so it drops into CI or a pre-upgrade check.

## A/B comparing two monty (or harness) builds

```bash
# Build A
MONTY_SPEC="pydantic-monty==0.0.18 pydantic-monty-runtime==0.0.18" \
  benchmarks/codemode/setup_env.sh .venv-a 3.13
.venv-a/bin/python benchmarks/codemode/bench.py --repeat 5 --json before.json

# Build B
MONTY_SPEC="pydantic-monty==0.0.19b2 pydantic-monty-runtime==0.0.19b2" \
  benchmarks/codemode/setup_env.sh .venv-b 3.13
.venv-b/bin/python benchmarks/codemode/bench.py --repeat 5 --json after.json

# Diff before.json / after.json for correctness drift and timing deltas.
```

To benchmark a harness change instead, install the two harness revisions into two envs at the same
monty version and compare.

## Notes

- The model is a `FunctionModel` that emits the scripted `run_code` calls -- no API keys, no
  network, fully deterministic.
- `wall_ms` is scenario wall-clock (agent run including sandbox startup and dispatch). It is a
  relative signal for A/B, not an absolute latency budget.
