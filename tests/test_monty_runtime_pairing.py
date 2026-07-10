"""Guard: `pydantic-monty` and `pydantic-monty-runtime` must be the same version.

They are a matched pair -- a given monty build only works with the runtime of
the same version -- but monty declares its runtime dependency loosely
(`pydantic-monty-runtime>=...`). An unlocked resolution can therefore install a
newer runtime than the monty it pairs with, and that mismatch does not fail
cleanly: it hangs the code-execution suites (`code_mode`, `dynamic_workflow`) to
the CI job timeout. This test turns that silent hang into an explicit, fast
failure, and enforces the lockstep the `pydantic-monty-runtime` pin in
`pyproject.toml` (`[tool.uv].constraint-dependencies`) relies on.
"""

from __future__ import annotations

import importlib.metadata

import pytest

pytest.importorskip('pydantic_monty')


def test_monty_and_runtime_versions_match() -> None:
    monty = importlib.metadata.version('pydantic-monty')
    runtime = importlib.metadata.version('pydantic-monty-runtime')
    assert monty == runtime, (
        f'pydantic-monty ({monty}) and pydantic-monty-runtime ({runtime}) resolved to different '
        'versions. They are a matched pair; a mismatch hangs the code-execution suites. Update the '
        "pydantic-monty-runtime pin in pyproject.toml's [tool.uv] constraint-dependencies to match "
        'the pydantic-monty version.'
    )
