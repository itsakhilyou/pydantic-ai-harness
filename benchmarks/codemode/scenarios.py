"""Scenario registry for the CodeMode + pydantic-monty harness.

Each `Scenario` is a sequence of `run_code` calls plus the outcome each should produce. They cover
the tool-bridging surface -- sync and async dispatch, REPL state across calls, and the three retry
channels (syntax, type, runtime). Append new scenarios to `SCENARIOS`; `bench.py` picks them up.
"""

from __future__ import annotations

import asyncio

from bench import Scenario, Step


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def fetch_price(item: str) -> dict[str, object]:
    """Look up the price of an item (simulated async I/O)."""
    await asyncio.sleep(0)
    return {'item': item, 'price': len(item)}


def divide(a: int, b: int) -> float:
    """Divide two numbers (raises on divide-by-zero, to exercise the runtime-error channel)."""
    return a / b


SCENARIOS: list[Scenario] = [
    Scenario(
        name='sync_call',
        description='add 4 and 6',
        tools=[add],
        steps=[Step('result = await add(a=4, b=6)\nresult', expect_return=10)],
    ),
    Scenario(
        name='print_and_result',
        description='add with a print',
        tools=[add],
        steps=[
            Step(
                'result = await add(a=4, b=6)\nprint(f"add returned {result}")\nresult',
                expect_return={'output': 'add returned 10\n', 'result': 10},
            )
        ],
    ),
    Scenario(
        name='async_gather',
        description='fetch two prices concurrently',
        tools=[fetch_price],
        steps=[
            Step(
                'import asyncio\n'
                "results = await asyncio.gather(fetch_price(item='pen'), fetch_price(item='pencil'))\n"
                'results',
                expect_return=[{'item': 'pen', 'price': 3}, {'item': 'pencil', 'price': 6}],
            )
        ],
    ),
    Scenario(
        name='multiple_tools',
        description='add then multiply',
        tools=[add, multiply],
        steps=[
            Step(
                's = await add(a=2, b=3)\np = await multiply(a=s, b=4)\np',
                expect_return=20,
            )
        ],
    ),
    Scenario(
        name='repl_state',
        description='keep state across two run_code calls',
        tools=[add],
        steps=[
            Step('x = await add(a=10, b=5)\nx', expect_return=15),
            Step('x * 2', expect_return=30),
        ],
    ),
    Scenario(
        name='syntax_error_retry',
        description='malformed code produces a syntax retry',
        tools=[add],
        steps=[Step('def (', expect_retry='Syntax error in code')],
    ),
    Scenario(
        name='type_error_retry',
        description='a type mismatch produces a type retry',
        tools=[add],
        steps=[Step('"hello" + 1', expect_retry='Type error in code')],
    ),
    Scenario(
        name='runtime_error_retry',
        description='a tool raising produces a runtime retry',
        tools=[divide],
        steps=[Step('await divide(a=1, b=0)', expect_retry='Runtime error')],
    ),
]
