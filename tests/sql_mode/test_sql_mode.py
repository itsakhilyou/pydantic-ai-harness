"""Tests for the `SQLModeBuilder` and the `SQLModeToolset` it builds.

Style follows `tests/code_mode/test_code_mode.py`: module-level
`pytestmark = pytest.mark.anyio`, an `anyio_backend` fixture, and a
`build_run_context` factory for invoking the toolset directly.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import SQLModeBuilder
from pydantic_ai_harness.sql_mode import SQLModeToolset
from pydantic_ai_harness.sql_mode._builder import _unwrap_optional  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def build_run_context() -> RunContext[None]:
    """Build a minimal `RunContext` for invoking the toolset directly in tests."""
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


async def run_sql(toolset: SQLModeToolset, query: str) -> Any:
    """Run `query` through the toolset's `run_sql` tool and return the result."""
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    return await toolset.call_tool('run_sql', {'query': query}, ctx, tools['run_sql'])


# ---------------------------------------------------------------------------
# Sample tool functions used by tests
# ---------------------------------------------------------------------------


class Location(BaseModel):
    """A latitude/longitude pair."""

    lat: float
    lon: float


def add_ints(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def shout(text: str) -> str:
    return text.upper()


def geocode(city: str) -> Location:
    """Look up a city's coordinates."""
    return Location(lat=48.85, lon=2.35)


async def fetch_weather(place: Location) -> dict[str, float]:
    """Get the weather at a location."""
    return {'temp_c': 21.0, 'lat': place.lat}


def explode(x: int) -> int:
    """A tool that always fails."""
    raise RuntimeError('tool exploded')


# ---------------------------------------------------------------------------
# Builder behaviour
# ---------------------------------------------------------------------------


def test_unwrap_optional() -> None:
    """`_unwrap_optional` peels `X | None` but leaves multi-member unions unchanged."""
    assert _unwrap_optional(int) is int
    assert _unwrap_optional(int | None) is int
    assert _unwrap_optional(int | str) == (int | str)


def test_build_requires_a_tool() -> None:
    """`build` refuses to produce a toolset with no registered tools."""
    with pytest.raises(ValueError, match='at least one tool'):
        SQLModeBuilder().build()


def test_register_tool_rejects_invalid_name() -> None:
    """A name that is not a valid SQL identifier is rejected."""
    with pytest.raises(ValueError, match='not a valid SQL identifier'):
        SQLModeBuilder().register_tool(add_ints, name='bad name')


def test_register_tool_rejects_duplicate_name() -> None:
    """Registering two tools under the same name is rejected."""
    builder = SQLModeBuilder().register_tool(add_ints)
    with pytest.raises(ValueError, match='already registered'):
        builder.register_tool(add_ints)


def test_register_tool_rejects_var_args() -> None:
    """Tools using `*args`/`**kwargs` cannot be registered."""

    def variadic(*args: int) -> int:
        return sum(args)

    assert variadic(1, 2) == 3
    with pytest.raises(ValueError, match=r'\*args'):
        SQLModeBuilder().register_tool(variadic, name='variadic')


async def test_description_lists_tools_and_schemas() -> None:
    """The `run_sql` description renders signatures, docstrings, and JSON schemas."""
    toolset = (
        SQLModeBuilder()
        .register_tool(add_ints)
        .register_tool(shout)
        .register_tool(geocode)
        .register_tool(fetch_weather)
        .build()
    )
    tools = await toolset.get_tools(build_run_context())
    description = tools['run_sql'].tool_def.description
    assert description is not None
    assert 'add_ints(a BIGINT, b BIGINT) -> BIGINT' in description
    assert 'Add two integers.' in description
    assert 'shout(text VARCHAR) -> VARCHAR' in description
    assert 'geocode(city VARCHAR) -> JSON' in description
    assert 'returns JSON matching:' in description
    assert 'fetch_weather(place JSON) -> JSON' in description
    assert '`place` is JSON matching:' in description
    assert tools['run_sql'].tool_def.metadata == {'code_arg_name': 'query', 'code_arg_language': 'sql'}
    assert toolset.id is None


async def test_description_uses_override() -> None:
    """An explicit `description=` overrides the function's docstring."""
    toolset = SQLModeBuilder().register_tool(shout, name='yell', description='Yell loudly').build()
    tools = await toolset.get_tools(build_run_context())
    description = tools['run_sql'].tool_def.description
    assert description is not None
    assert 'yell(text VARCHAR) -> VARCHAR' in description
    assert 'Yell loudly' in description


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


async def test_sync_scalar_tool() -> None:
    """A sync tool with scalar parameters runs and returns native column types."""
    toolset = SQLModeBuilder().register_tool(add_ints).build()
    result = await run_sql(toolset, 'SELECT add_ints(2, 3) AS total')
    assert result['columns'] == [{'name': 'total', 'type': 'BIGINT'}]
    assert result['rows'] == [{'total': 5}]
    assert result['row_count'] == 1


async def test_string_tool() -> None:
    """A tool with a string parameter and string return runs."""
    toolset = SQLModeBuilder().register_tool(shout).build()
    result = await run_sql(toolset, "SELECT shout('hi') AS loud")
    assert result['rows'] == [{'loud': 'HI'}]


async def test_json_return_extracted_with_arrow() -> None:
    """A JSON-returning tool's fields can be read with the `->>` operator."""
    toolset = SQLModeBuilder().register_tool(geocode).build()
    result = await run_sql(toolset, "SELECT geocode('Paris')->>'lat' AS lat")
    assert result['rows'] == [{'lat': '48.85'}]


async def test_json_columns_are_parsed() -> None:
    """JSON-typed result columns are parsed; non-JSON and NULL columns are left alone."""
    toolset = SQLModeBuilder().register_tool(geocode).register_tool(add_ints).build()
    result = await run_sql(toolset, "SELECT geocode('Paris') AS loc, add_ints(1, 2) AS n, NULL::JSON AS missing")
    assert result['rows'] == [{'loc': {'lat': 48.85, 'lon': 2.35}, 'n': 3, 'missing': None}]


async def test_async_tool_runs_via_portal() -> None:
    """An async tool is bridged to the event loop and can consume another tool's JSON."""
    toolset = SQLModeBuilder().register_tool(geocode).register_tool(fetch_weather).build()
    result = await run_sql(toolset, "SELECT fetch_weather(geocode('Paris'))->>'temp_c' AS temp")
    assert result['rows'] == [{'temp': '21.0'}]


async def test_fan_out_with_unnest() -> None:
    """A tool can be called once per row by pairing it with `unnest`."""
    toolset = SQLModeBuilder().register_tool(add_ints).build()
    result = await run_sql(toolset, 'SELECT add_ints(x, 100) AS v FROM (SELECT unnest([1, 2, 3]) AS x) ORDER BY v')
    assert [row['v'] for row in result['rows']] == [101, 102, 103]


async def test_result_is_truncated_to_max_rows() -> None:
    """Result sets larger than `max_rows` are truncated and flagged."""
    toolset = SQLModeBuilder(max_rows=2).register_tool(add_ints).build()
    result = await run_sql(toolset, 'SELECT add_ints(x, 0) AS v FROM (SELECT unnest([1, 2, 3, 4]) AS x)')
    assert result['row_count'] == 2
    assert result['truncated'] is True
    assert '4 rows' in result['note']


async def test_statement_with_no_rows() -> None:
    """A statement that produces no rows returns an empty result set."""
    toolset = SQLModeBuilder().register_tool(add_ints).build()
    result = await run_sql(toolset, 'CREATE TABLE t (x INTEGER)')
    assert result['rows'] == []
    assert result['row_count'] == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_sql_syntax_error_raises_model_retry() -> None:
    """A malformed query surfaces as `ModelRetry`."""
    toolset = SQLModeBuilder().register_tool(add_ints).build()
    with pytest.raises(ModelRetry, match='SQL error'):
        await run_sql(toolset, 'SELCT nonsense')


async def test_lockdown_blocks_filesystem_access() -> None:
    """The locked-down database refuses to read local files."""
    toolset = SQLModeBuilder().register_tool(add_ints).build()
    with pytest.raises(ModelRetry, match='SQL error'):
        await run_sql(toolset, "SELECT * FROM read_csv('/etc/passwd')")


async def test_tool_exception_raises_model_retry() -> None:
    """An exception raised inside a tool surfaces as `ModelRetry`."""
    toolset = SQLModeBuilder().register_tool(explode).build()
    with pytest.raises(ModelRetry, match='SQL error'):
        await run_sql(toolset, 'SELECT explode(1)')


async def test_invalid_json_argument_raises_model_retry() -> None:
    """A JSON argument that fails pydantic validation surfaces as `ModelRetry`."""
    toolset = SQLModeBuilder().register_tool(fetch_weather).build()
    with pytest.raises(ModelRetry, match='SQL error'):
        await run_sql(toolset, 'SELECT fetch_weather(\'{"lat": "not-a-number", "lon": 1.0}\')')


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


async def test_agent_runs_sql_tool() -> None:
    """An agent wired with the toolset can call `run_sql` end to end."""
    toolset = SQLModeBuilder().register_tool(add_ints).build()
    responses = iter(
        [
            ModelResponse(parts=[ToolCallPart('run_sql', {'query': 'SELECT add_ints(40, 2) AS answer'})]),
            ModelResponse(parts=[TextPart('done')]),
        ]
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return next(responses)

    agent = Agent(FunctionModel(model_fn), toolsets=[toolset])
    result = await agent.run('compute the answer')
    assert result.output == 'done'
