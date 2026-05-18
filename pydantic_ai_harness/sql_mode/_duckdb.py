"""DuckDB-backed execution for SQLMode.

Each `run_sql` call gets a brand-new, locked-down, in-memory DuckDB connection:
the registered tools are exposed as DuckDB user-defined functions, the query
runs, and the connection is thrown away -- nothing persists between calls.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from collections.abc import Callable
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_core import to_jsonable_python

try:
    import duckdb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'duckdb is required for SQLMode. Install it with: pip install "pydantic-ai-harness[sql-mode]"'
    ) from e

if importlib.util.find_spec('numpy') is None:  # pragma: no cover
    raise ImportError(
        'numpy is required for SQLMode -- DuckDB needs it to register Python functions. '
        'Install it with: pip install "pydantic-ai-harness[sql-mode]"'
    )

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

    from pydantic_ai_harness.sql_mode._builder import _RegisteredTool  # pyright: ignore[reportPrivateUsage]


# Applied in order on every connection. `lock_configuration` MUST come last: it
# freezes every preceding setting so the model's SQL cannot turn them back on.
# See https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview
_LOCKDOWN_STATEMENTS: tuple[str, ...] = (
    'SET autoload_known_extensions = false',
    'SET autoinstall_known_extensions = false',
    'SET allow_community_extensions = false',
    'SET enable_external_access = false',
    'SET lock_configuration = true',
)


def _build_udf(tool: _RegisteredTool, portal: BlockingPortal) -> Callable[..., Any]:
    """Build the synchronous callable DuckDB invokes for `tool`.

    DuckDB calls this once per row, from a worker thread. JSON arguments are
    parsed and validated against the tool's pydantic types; the return value is
    dumped back to JSON. Async tools are run on the event loop via `portal`.
    """

    def udf(*raw_args: Any) -> Any:
        """Validate arguments, invoke the tool, and serialize its result for DuckDB."""
        kwargs = {
            param.name: (param.adapter.validate_json(raw) if param.is_json else param.adapter.validate_python(raw))
            for param, raw in zip(tool.params, raw_args)
        }
        if tool.is_async:
            result = portal.call(functools.partial(tool.fn, **kwargs))
        else:
            result = tool.fn(**kwargs)
        if tool.return_is_json:
            return tool.return_adapter.dump_json(result, warnings=False).decode()
        return result

    # DuckDB matches a UDF's arity via `inspect.signature`, so give the variadic
    # wrapper a fixed-arity signature matching the registered parameter types.
    udf.__signature__ = Signature(  # pyright: ignore[reportFunctionMemberAccess]
        [Parameter(param.name, Parameter.POSITIONAL_OR_KEYWORD) for param in tool.params]
    )
    return udf


def _format_result(cursor: duckdb.DuckDBPyConnection, max_rows: int) -> dict[str, Any]:
    """Shape a finished DuckDB result into a JSON-friendly dict of columns and rows."""
    columns = [{'name': str(col[0]), 'type': str(col[1])} for col in cursor.description]
    json_columns = {index for index, col in enumerate(columns) if col['type'].upper() == 'JSON'}

    all_rows = cursor.fetchall()
    truncated = len(all_rows) > max_rows
    visible_rows = all_rows[:max_rows] if truncated else all_rows

    records: list[dict[str, Any]] = []
    for row in visible_rows:
        record: dict[str, Any] = {}
        for index, col in enumerate(columns):
            value = row[index]
            if index in json_columns and isinstance(value, str):
                value = json.loads(value)
            record[col['name']] = value
        records.append(record)

    result: dict[str, Any] = {
        'columns': columns,
        'rows': to_jsonable_python(records, serialize_unknown=True),
        'row_count': len(records),
    }
    if truncated:
        result['truncated'] = True
        result['note'] = f'Showing the first {max_rows} of {len(all_rows)} rows.'
    return result


def run_query(
    tools: tuple[_RegisteredTool, ...],
    query: str,
    portal: BlockingPortal,
    max_rows: int,
) -> dict[str, Any]:
    """Run `query` against a fresh, locked-down, in-memory DuckDB database.

    Called in a worker thread. The connection is created, has `tools` registered
    as user-defined functions, is locked down, runs the query, and is closed.
    Raises `ModelRetry` on any SQL or tool error so the model can try again.
    """
    con = duckdb.connect(':memory:')
    try:
        con.execute('LOAD json')
        for tool in tools:
            con.create_function(  # pyright: ignore[reportUnknownMemberType]
                tool.name,
                _build_udf(tool, portal),
                [duckdb.dtype(param.duck_type) for param in tool.params],
                duckdb.dtype(tool.return_duck_type),
                side_effects=True,
            )
        for statement in _LOCKDOWN_STATEMENTS:
            con.execute(statement)
        try:
            cursor = con.execute(query)
            return _format_result(cursor, max_rows)
        except duckdb.Error as e:
            raise ModelRetry(f'SQL error: {e}') from e
    finally:
        con.close()
