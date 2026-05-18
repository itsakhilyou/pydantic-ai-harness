"""The `SQLModeToolset` -- exposes a single `run_sql` tool backed by locked-down DuckDB."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any

from anyio.from_thread import BlockingPortal
from anyio.to_thread import run_sync
from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from typing_extensions import TypedDict

from pydantic_ai_harness.sql_mode._duckdb import run_query

if TYPE_CHECKING:
    from pydantic_ai_harness.sql_mode._builder import _RegisteredTool  # pyright: ignore[reportPrivateUsage]

_RUN_SQL_TOOL_NAME = 'run_sql'


class _RunSqlArguments(TypedDict):
    query: Annotated[str, Field(description='The DuckDB SQL query to run.')]


_RUN_SQL_ADAPTER = TypeAdapter(_RunSqlArguments)
_RUN_SQL_JSON_SCHEMA = _RUN_SQL_ADAPTER.json_schema()
_RUN_SQL_ARGS_VALIDATOR: SchemaValidatorProt = _RUN_SQL_ADAPTER.validator  # pyright: ignore[reportAssignmentType]

_BASE_DESCRIPTION = """\
Run a SQL query against a fresh, locked-down, in-memory DuckDB database.

Write SQL to orchestrate tool calls: invoke the tool functions listed below from
inside the query, pass values between them, and join, filter, aggregate, and
transform the results -- all in a single query and a single round-trip.

Every call gets a brand-new in-memory database; nothing persists between calls.
The database is sandboxed -- no filesystem, no network, no extension loading -- \
but every built-in DuckDB function is available, including the JSON functions \
(`json_extract`, `->`, `->>`, `json_transform`, `unnest`, `array_agg`, `struct_pack`, and more).

A tool whose result type is shown as `JSON` returns a JSON value: read fields \
from it with `->`/`->>`, or feed it straight into another tool that takes a \
`JSON` argument. To call a tool once per item, pair it with `unnest`:

    SELECT my_tool(item) AS result FROM (SELECT unnest(['a', 'b', 'c']) AS item)

The result set is returned as `columns` (each with a name and type) and `rows`. \
Keep result sets small -- aggregate or `LIMIT` in SQL rather than returning raw rows.\
"""


def _render_tool(tool: _RegisteredTool) -> str:
    """Render one registered tool as a signature, docstring, and JSON schemas."""
    signature = ', '.join(f'{param.name} {param.duck_type}' for param in tool.params)
    lines = [f'{tool.name}({signature}) -> {tool.return_duck_type}']
    if tool.description:
        lines.extend(f'    {line}' for line in tool.description.splitlines())
    for param in tool.params:
        if param.json_schema is not None:
            lines.append(f'    `{param.name}` is JSON matching: {json.dumps(param.json_schema)}')
    if tool.return_json_schema is not None:
        lines.append(f'    returns JSON matching: {json.dumps(tool.return_json_schema)}')
    return '\n'.join(lines)


@dataclass
class SQLModeToolset(AbstractToolset[Any]):
    """Toolset exposing a single `run_sql` tool backed by a locked-down DuckDB database.

    Build one with `SQLModeBuilder`. The registered tools are presented to the
    model as DuckDB functions inside the `run_sql` tool description; they are not
    exposed as native tool calls. Each `run_sql` call runs against a fresh,
    sandboxed, in-memory database -- no filesystem, no network, no state shared
    between calls.
    """

    tools: tuple[_RegisteredTool, ...]
    """The tools registered with the builder, exposed as DuckDB functions."""

    max_rows: int = 1000
    """Maximum number of result rows returned to the model before truncation."""

    max_retries: int = 3
    """Retries allowed for `run_sql` when a query or tool call fails."""

    @property
    def id(self) -> str | None:
        """Toolset identifier (unused -- SQLMode contributes a single fixed tool)."""
        return None

    @cached_property
    def _description(self) -> str:
        """The `run_sql` description: base guidance plus the rendered tool listing."""
        listing = '\n\n'.join(_render_tool(tool) for tool in self.tools)
        return f'{_BASE_DESCRIPTION}\n\nTool functions available inside the query:\n\n{listing}'

    @cached_property
    def _tool_def(self) -> ToolDefinition:
        """The `ToolDefinition` for `run_sql`, built once from the registered tools."""
        return ToolDefinition(
            name=_RUN_SQL_TOOL_NAME,
            description=self._description,
            parameters_json_schema=_RUN_SQL_JSON_SCHEMA,
            metadata={'code_arg_name': 'query', 'code_arg_language': 'sql'},
        )

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Return the single `run_sql` tool."""
        return {
            _RUN_SQL_TOOL_NAME: ToolsetTool(
                toolset=self,
                tool_def=self._tool_def,
                max_retries=self.max_retries,
                args_validator=_RUN_SQL_ARGS_VALIDATOR,
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> Any:
        """Run the model's SQL query in a locked-down DuckDB database off the event loop.

        The query runs in a worker thread; async tools are bridged back to the
        running event loop through a `BlockingPortal`. Any error is captured
        inside the portal scope and re-raised afterwards so a `ModelRetry`
        reaches the caller directly, rather than wrapped in the portal task
        group's `ExceptionGroup`.
        """
        query: str = tool_args['query']
        result: dict[str, Any] = {}
        error: Exception | None = None
        async with BlockingPortal() as portal:
            try:
                result = await run_sync(run_query, self.tools, query, portal, self.max_rows)
            except Exception as exc:
                error = exc
        if error is not None:
            raise error
        return result
