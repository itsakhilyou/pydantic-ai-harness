"""Tests for the experimental-capability warning convention."""

from __future__ import annotations

import importlib
import importlib.util
import warnings

import pytest

from pydantic_ai_harness.experimental import HarnessExperimentalWarning
from pydantic_ai_harness.experimental._warn import warn_experimental

# `dynamic_workflow` imports `pydantic-monty`, which is gated to Python < 3.14 (no cp314 wheel).
_MONTY_ABSENT = importlib.util.find_spec('pydantic_monty') is None


class TestExperimentalWarning:
    def test_message_names_feature_and_carries_silence_snippet(self) -> None:
        with pytest.warns(HarnessExperimentalWarning) as rec:
            warn_experimental('compaction')
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert '`pydantic_ai_harness.experimental.compaction`' in msg
        # The message must hand the user the exact, category-wide silence line.
        assert "warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)" in msg

    def test_one_filter_silences_every_capability(self) -> None:
        # A single category filter mutes all experimental warnings — no per-capability lines.
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # baseline: any warning is an error
            warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
            warn_experimental('compaction')
            warn_experimental('some_future_capability')  # also silenced, same filter

    @pytest.mark.parametrize(
        'feature',
        [
            'compaction',
            'subagents',
            pytest.param(
                'dynamic_workflow',
                marks=pytest.mark.skipif(_MONTY_ABSENT, reason='pydantic-monty is gated to Python < 3.14'),
            ),
        ],
    )
    def test_importing_a_capability_warns(self, feature: str) -> None:
        module = importlib.import_module(f'pydantic_ai_harness.experimental.{feature}')
        with pytest.warns(HarnessExperimentalWarning):
            importlib.reload(module)

    @pytest.mark.parametrize('feature', ['step_persistence', 'media'])
    def test_importing_step_persistence_or_media_warns(self, feature: str) -> None:
        module = importlib.import_module(f'pydantic_ai_harness.experimental.{feature}')
        with pytest.warns(HarnessExperimentalWarning, match=feature):
            importlib.reload(module)
