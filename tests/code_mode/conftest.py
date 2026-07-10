"""Collection guard for the code_mode test modules."""

from __future__ import annotations

import importlib.util

# `pydantic-monty` is gated to Python < 3.14 (no cp314 wheel yet) and lives behind the
# `code-mode` extra, so 3.14 and monty-less CI runs can't import these modules. Ignore them at
# collection when monty is absent. A conditional expression rather than an `if` statement:
# branch coverage traces statement arcs, and no single environment takes both arms of an
# install-dependent branch.
collect_ignore = (
    ['test_code_mode.py', 'test_temporal.py', 'test_dbos.py']
    if importlib.util.find_spec('pydantic_monty') is None
    else []
)
