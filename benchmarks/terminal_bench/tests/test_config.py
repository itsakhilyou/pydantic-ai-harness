"""Tests for run configuration: model mapping, usage limits, cost table."""

from __future__ import annotations

import pytest

from pydantic_ai_harness_terminal_bench.config import (
    COST_ESTIMATES,
    DEFAULT_REQUEST_LIMIT,
    DEFAULT_TOOL_CALLS_LIMIT,
    build_usage_limits,
    convert_model_name,
)


@pytest.mark.parametrize(
    ('harbor_name', 'expected'),
    [
        ('anthropic/claude-opus-4-6', 'anthropic:claude-opus-4-6'),
        ('openai/gpt-5.2', 'openai:gpt-5.2'),
        ('vertex/gemini-3.1-pro', 'google-cloud:gemini-3.1-pro'),
        ('google-cloud/gemini-3.1-pro', 'google-cloud:gemini-3.1-pro'),
        ('gemini/gemini-3.1-pro', 'google:gemini-3.1-pro'),
        ('google/gemini-3.1-pro', 'google:gemini-3.1-pro'),
        # Already a Pydantic AI id: unchanged.
        ('anthropic:claude-opus-4-6', 'anthropic:claude-opus-4-6'),
        # Bare model name, no provider: unchanged.
        ('gpt-5.2', 'gpt-5.2'),
    ],
)
def test_convert_model_name(harbor_name: str, expected: str) -> None:
    assert convert_model_name(harbor_name) == expected


def test_usage_limits_defaults() -> None:
    limits = build_usage_limits()
    assert limits.request_limit == DEFAULT_REQUEST_LIMIT
    assert limits.tool_calls_limit == DEFAULT_TOOL_CALLS_LIMIT


def test_cost_estimates_shape() -> None:
    assert COST_ESTIMATES
    for row in COST_ESTIMATES:
        label, shape, usd = row
        assert label and shape
        assert usd.startswith('$')
