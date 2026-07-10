"""Reusable local verification + benchmark harness for CodeMode over pydantic-monty.

Drives a real `Agent(capabilities=[CodeMode(...)])` through scripted `run_code` calls against
whatever `pydantic-monty` is installed in the active environment, checks that tool-bridging
produces the expected results, and reports per-scenario wall-clock timing.

Point it at different monty builds (see `setup_env.sh`) and diff the JSON output to A/B a harness
or monty change:

    python benchmarks/codemode/bench.py --json before.json      # on monty A
    python benchmarks/codemode/bench.py --json after.json        # on monty B
    # then compare before.json / after.json

Add a scenario by appending to `SCENARIOS` in `scenarios.py`; nothing here needs to change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import CodeMode

# Sentinel so a step can opt out of return-value checking (benchmark-only step).
_UNSET = object()


@dataclass
class Step:
    """One `run_code` call in a scenario, with an optional expectation.

    Exactly one of `expect_return` / `expect_retry` should be set to assert an outcome; leave both
    unset to run the step for timing only.
    """

    code: str
    expect_return: object = _UNSET
    """Expected `run_code` tool-return content (the dict/value the model would observe)."""
    expect_retry: str | None = None
    """Substring expected in the `ModelRetry` message when this step is meant to fail."""


@dataclass
class Scenario:
    """A named sequence of `run_code` calls exercising one aspect of tool-bridging."""

    name: str
    description: str
    tools: Sequence[Callable[..., object]]
    steps: Sequence[Step]


@dataclass
class ScenarioResult:
    """The outcome of running one scenario: pass/fail, wall-clock, and any mismatch details."""

    name: str
    passed: bool
    wall_ms: float
    run_code_calls: int
    failures: list[str] = field(default_factory=list[str])


@dataclass
class _Outcome:
    kind: Literal['return', 'retry']
    content: object


def _build_model(steps: Sequence[Step]) -> FunctionModel:
    """A model that emits the next scenario step as a `run_code` call, then a final text.

    The next step is chosen by counting the `run_code` calls already in history, so each step maps
    to exactly one call regardless of whether earlier steps returned or retried.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        emitted = sum(
            1
            for message in messages
            for part in message.parts
            if isinstance(part, ToolCallPart) and part.tool_name == 'run_code'
        )
        if emitted < len(steps):
            call_id = f'rc_{emitted}'
            return ModelResponse(
                parts=[ToolCallPart(tool_name='run_code', args={'code': steps[emitted].code}, tool_call_id=call_id)]
            )
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model_fn)


def _collect_outcomes(messages: Sequence[ModelMessage]) -> list[_Outcome]:
    """Pull each `run_code` response (return or retry) out of the message history, in order."""
    outcomes: list[_Outcome] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                outcomes.append(_Outcome('return', part.content))
            elif isinstance(part, RetryPromptPart) and part.tool_name == 'run_code':
                outcomes.append(_Outcome('retry', part.model_response()))
    return outcomes


def _check_step(index: int, step: Step, outcome: _Outcome | None) -> str | None:
    """Return a failure message if the step's outcome does not match its expectation, else None."""
    if outcome is None:
        return f'step {index}: no run_code outcome recorded'
    if step.expect_retry is not None:
        if outcome.kind != 'retry':
            return f'step {index}: expected a retry, got a return ({outcome.content!r})'
        if step.expect_retry not in str(outcome.content):
            return f'step {index}: retry message missing {step.expect_retry!r}; got {outcome.content!r}'
        return None
    if step.expect_return is not _UNSET:
        if outcome.kind != 'return':
            return f'step {index}: expected a return, got a retry ({outcome.content!r})'
        if outcome.content != step.expect_return:
            return f'step {index}: return {outcome.content!r} != expected {step.expect_return!r}'
    return None


async def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Run one scenario end-to-end through `Agent.run` and check + time it."""
    agent: Agent[object, str] = Agent(_build_model(scenario.steps), capabilities=[CodeMode[object]()])
    for tool in scenario.tools:
        agent.tool_plain(tool)

    start = time.perf_counter()
    result = await agent.run(scenario.description)
    wall_ms = (time.perf_counter() - start) * 1000

    outcomes = _collect_outcomes(result.all_messages())
    failures: list[str] = []
    for index, step in enumerate(scenario.steps):
        failure = _check_step(index, step, outcomes[index] if index < len(outcomes) else None)
        if failure is not None:
            failures.append(failure)

    return ScenarioResult(
        name=scenario.name,
        passed=not failures,
        wall_ms=wall_ms,
        run_code_calls=len(outcomes),
        failures=failures,
    )


def _monty_version() -> str:
    try:
        import pydantic_monty

        return pydantic_monty.__version__
    except Exception as exc:  # pragma: no cover - reported, not raised
        return f'<unavailable: {exc}>'


async def _run_all(scenarios: Sequence[Scenario]) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        results.append(await run_scenario(scenario))
    return results


def _print_table(results: Sequence[ScenarioResult]) -> None:
    name_width = max((len(r.name) for r in results), default=4)
    print(f'{"scenario":<{name_width}}  {"result":<6}  {"wall_ms":>8}  calls')
    print('-' * (name_width + 26))
    for r in results:
        status = 'ok' if r.passed else 'FAIL'
        print(f'{r.name:<{name_width}}  {status:<6}  {r.wall_ms:>8.1f}  {r.run_code_calls}')
        for failure in r.failures:
            print(f'    - {failure}')


def main() -> int:
    """Parse args, run the selected scenarios, print a table, and optionally write JSON."""
    parser = argparse.ArgumentParser(description='CodeMode + pydantic-monty verification/benchmark harness.')
    parser.add_argument('--json', type=str, default=None, help='Write machine-readable results to this path.')
    parser.add_argument(
        '--filter', type=str, default=None, help='Only run scenarios whose name contains this substring.'
    )
    parser.add_argument(
        '--repeat', type=int, default=1, help='Run the suite N times; report the fastest wall time per scenario.'
    )
    args = parser.parse_args()

    from scenarios import SCENARIOS

    selected = [s for s in SCENARIOS if args.filter is None or args.filter in s.name]
    if not selected:
        print(f'no scenarios match filter {args.filter!r}')
        return 2

    monty_version = _monty_version()
    from importlib.metadata import version as _pkg_version

    harness_version = _pkg_version('pydantic-ai-harness')
    print(f'pydantic-monty {monty_version}  |  pydantic-ai-harness {harness_version}\n')

    best: dict[str, ScenarioResult] = {}
    for _ in range(max(1, args.repeat)):
        for result in asyncio.run(_run_all(selected)):
            prior = best.get(result.name)
            if prior is None or result.wall_ms < prior.wall_ms:
                best[result.name] = result

    results = [best[s.name] for s in selected]
    _print_table(results)

    all_passed = all(r.passed for r in results)
    print(f'\n{sum(r.passed for r in results)}/{len(results)} scenarios passed')

    if args.json:
        payload = {
            'monty_version': monty_version,
            'harness_version': harness_version,
            'scenarios': [
                {'name': r.name, 'passed': r.passed, 'wall_ms': round(r.wall_ms, 2), 'run_code_calls': r.run_code_calls}
                for r in results
            ],
        }
        with open(args.json, 'w') as handle:
            json.dump(payload, handle, indent=2)
        print(f'wrote {args.json}')

    return 0 if all_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
