"""Collection guard for the dynamic_workflow test module."""

from __future__ import annotations

import importlib.util

# `pydantic-monty` is gated to Python < 3.14 (no cp314 wheel yet) and lives behind the
# `dynamic-workflow` extra, so 3.14 and monty-less CI runs can't import this module. Ignore it at
# collection when monty is absent. A conditional expression rather than an `if` statement:
# branch coverage traces statement arcs, and no single environment takes both arms of an
# install-dependent branch.
collect_ignore = ['test_dynamic_workflow.py'] if importlib.util.find_spec('pydantic_monty') is None else []
