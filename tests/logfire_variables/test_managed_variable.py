"""Tests for the shared `ManagedVariableCapability` base (source `pydantic_ai_harness.logfire`).

`ManagedPrompt`, `ManagedTool`, and `ManagedToolset` all derive their backing variable through the
base's `_build_managed_variable`: the `<prefix><name>` naming, hyphen-to-underscore normalization,
accidental-prefix stripping, and invalid-name rejection are identical across the three. Rather than
re-assert that shared contract once per capability, it is exercised here once through a minimal
concrete subclass. Each capability's own test module keeps only a smoke test that it wires its name,
prefix, value type, and default into this base correctly, plus its capability-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import logfire
import pytest

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

_PREFIX = 'var__'


@dataclass
class _StrVariable(ManagedVariableCapability[None, str]):
    """The smallest possible capability: a `str` variable declared from a bare name."""

    raw_name: str

    def __post_init__(self) -> None:
        self._resolved = self._new_resolved()
        self._variable = self._build_managed_variable(self.raw_name, prefix=_PREFIX, value_type=str, default='')


def test_name_becomes_prefixed_variable_name() -> None:
    assert _StrVariable('greeting')._variable.name == 'var__greeting'


def test_hyphenated_name_is_normalized() -> None:
    assert _StrVariable('welcome-email')._variable.name == 'var__welcome_email'


def test_prefix_in_name_warns_and_is_stripped() -> None:
    with pytest.warns(UserWarning, match='added automatically') as caught:
        capability = _StrVariable('var__already_prefixed')
    assert capability._variable.name == 'var__already_prefixed'
    # The warning's filename should be this test module (the user's call site), not the
    # library's internal `_managed_variable.py`. Anchors the `stacklevel` against regressions.
    assert caught[0].filename == __file__


def test_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match='invalid variable name'):
        _StrVariable('has spaces')


def test_duplicate_construction_is_idempotent() -> None:
    # Each instance builds its own backing variable directly, so the same name can be declared
    # repeatedly (e.g. shared across agents) without the duplicate-registration error `logfire.var`
    # would raise.
    first = _StrVariable('shared')
    second = _StrVariable('shared')
    assert first._variable.name == second._variable.name == 'var__shared'


def test_explicit_logfire_instance_is_used() -> None:
    # Exercises the explicit-instance branch of variable construction (the default-instance branch is
    # covered by every other construction).
    capability = _StrVariable('with_instance', logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)
    assert capability._variable.name == 'var__with_instance'
