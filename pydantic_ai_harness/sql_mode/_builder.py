"""The `SQLModeBuilder` -- register tool functions, then `.build()` a `SQLModeToolset`."""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

from pydantic import TypeAdapter

from pydantic_ai_harness.sql_mode._toolset import SQLModeToolset

# Python scalar types that map directly onto a native DuckDB column type. Every
# other annotation (models, TypedDicts, dicts, lists, unions, `Any`) becomes JSON.
_SCALAR_DUCK_TYPES: dict[Any, str] = {
    str: 'VARCHAR',
    bool: 'BOOLEAN',
    int: 'BIGINT',
    float: 'DOUBLE',
    bytes: 'BLOB',
}


@dataclass
class _Param:
    """A single tool parameter, resolved to its DuckDB type and pydantic validator."""

    name: str
    duck_type: str
    is_json: bool
    adapter: TypeAdapter[Any]
    json_schema: dict[str, Any] | None


@dataclass
class _RegisteredTool:
    """A tool function registered with `SQLModeBuilder`, ready to become a DuckDB UDF."""

    name: str
    fn: Callable[..., Any]
    is_async: bool
    description: str | None
    params: tuple[_Param, ...]
    return_is_json: bool
    return_duck_type: str
    return_adapter: TypeAdapter[Any]
    return_json_schema: dict[str, Any] | None


def _unwrap_optional(annotation: Any) -> Any:
    """Reduce `Optional[X]` / `X | None` to `X`; leave every other annotation unchanged."""
    if typing.get_origin(annotation) in (Union, types.UnionType):
        non_none = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _resolve_type(annotation: Any) -> tuple[str, bool, TypeAdapter[Any], dict[str, Any] | None]:
    """Resolve an annotation to `(duck_type, is_json, validator, json_schema)`.

    Scalar annotations get a native DuckDB type and no schema; everything else
    is carried as JSON, with the pydantic JSON Schema kept for the model to see.
    """
    adapter: TypeAdapter[Any] = TypeAdapter(annotation)
    duck_type = _SCALAR_DUCK_TYPES.get(_unwrap_optional(annotation))
    if duck_type is not None:
        return duck_type, False, adapter, None
    return 'JSON', True, adapter, adapter.json_schema()


class SQLModeBuilder:
    """Collects tool functions and builds a `SQLModeToolset`.

    Register sync or async functions with `register_tool`, then call `build` to
    get a toolset exposing a single `run_sql` tool. The model writes SQL that
    calls the registered tools as DuckDB functions.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import SQLModeBuilder

    def geocode(city: str) -> dict[str, float]:
        '''Look up a city's latitude and longitude.'''
        ...

    sql_mode = SQLModeBuilder().register_tool(geocode).build()
    agent = Agent('openai:gpt-5', toolsets=[sql_mode])
    ```
    """

    def __init__(self, *, max_rows: int = 1000, max_retries: int = 3) -> None:
        """Create an empty builder.

        Args:
            max_rows: Maximum number of result rows returned to the model; extra
                rows are dropped and the result is flagged as truncated.
            max_retries: Retries allowed for the `run_sql` tool when a query or
                tool call fails.
        """
        self._tools: list[_RegisteredTool] = []
        self._max_rows = max_rows
        self._max_retries = max_retries

    def register_tool(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> SQLModeBuilder:
        """Register a sync or async function as a tool callable from SQL.

        Parameters and the return value are introspected from the function's
        type hints. Scalar types (`str`, `int`, `float`, `bool`, `bytes`) map to
        native DuckDB column types; everything else is passed as JSON, validated
        against the function's pydantic types.

        Args:
            fn: The tool function. Must not use `*args`/`**kwargs`.
            name: SQL function name. Defaults to `fn.__name__`; must be a valid
                identifier.
            description: Overrides the function's docstring in the tool listing.

        Returns:
            The builder, so calls can be chained.
        """
        tool_name = name or getattr(fn, '__name__', '')
        if not tool_name.isidentifier():
            raise ValueError(f'SQLMode tool name {tool_name!r} is not a valid SQL identifier; pass `name=`.')
        if any(tool.name == tool_name for tool in self._tools):
            raise ValueError(f'A tool named {tool_name!r} is already registered.')

        hints = typing.get_type_hints(fn)
        params: list[_Param] = []
        for parameter in inspect.signature(fn).parameters.values():
            if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise ValueError(f'SQLMode tool {tool_name!r} cannot use *args/**kwargs parameters.')
            duck_type, is_json, adapter, schema = _resolve_type(hints.get(parameter.name, Any))
            params.append(
                _Param(name=parameter.name, duck_type=duck_type, is_json=is_json, adapter=adapter, json_schema=schema)
            )

        return_duck, return_is_json, return_adapter, return_schema = _resolve_type(hints.get('return', Any))
        self._tools.append(
            _RegisteredTool(
                name=tool_name,
                fn=fn,
                is_async=inspect.iscoroutinefunction(fn),
                description=description or inspect.getdoc(fn),
                params=tuple(params),
                return_is_json=return_is_json,
                return_duck_type=return_duck,
                return_adapter=return_adapter,
                return_json_schema=return_schema,
            )
        )
        return self

    def build(self) -> SQLModeToolset:
        """Build the `SQLModeToolset` from the registered tools.

        Raises:
            ValueError: If no tools have been registered.
        """
        if not self._tools:
            raise ValueError('Register at least one tool with `register_tool` before calling `build`.')
        return SQLModeToolset(tools=tuple(self._tools), max_rows=self._max_rows, max_retries=self._max_retries)
