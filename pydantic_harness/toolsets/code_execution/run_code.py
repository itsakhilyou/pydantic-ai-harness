"""Code execution toolset that runs LLM-generated Python in a Monty sandbox."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypedDict, cast

from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_monty import MontyRepl, MontyRuntimeError, MontySyntaxError, MontyTypingError
from typing_extensions import NotRequired


class _RunCodeArguments(TypedDict):
    code: Annotated[str, Field(description='The Python code to execute in the sandbox.')]
    restart: NotRequired[Annotated[bool, Field(description='Set to true to reset REPL state. When false (default), state is preserved between calls.')]]


_RUN_CODE_TOOL_NAME = 'run_code'
_RUN_CODE_ADAPTER = TypeAdapter(_RunCodeArguments)
_RUN_CODE_JSON_SCHEMA = _RUN_CODE_ADAPTER.json_schema()
_RUN_CODE_ARGS_VALIDATOR = _RUN_CODE_ADAPTER.validator

_RUN_CODE_DESCRIPTION = """\
Write and run Python code in a sandboxed environment.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes**: class definitions are not supported
- **No third-party libraries**: only the standard library modules listed below are available
- **Available modules**: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `datetime`, `os`, `pathlib`
- **No `import *`**: wildcard imports are not supported

State is preserved between calls (REPL-style). Set `restart: true` to reset state.

Returns: {
  "output": "Text printed to stdout via print() calls. Omitted if nothing was printed.",
  "result": "The value of the last expression, like a Python REPL. Omitted if the last statement is not an expression (e.g. assignment, function call with no return value)."
}\
"""

# TODO: Add Python function signatures for available external functions to the tool description,
# so the LLM knows what functions are available and their parameters.

# TODO: Sanitize tool names that aren't valid Python identifiers (e.g. MCP tools with
# hyphens/dots like `get-weather`, `api.call`) and map them back on dispatch.


@dataclass
class CodeExecutionToolset(WrapperToolset[AgentDepsT]):
    """Executes LLM-generated Python code in a Monty sandbox.

    Exposes a single `run_code` tool. If the wrapped toolset has tools, they are
    automatically available as callable functions inside the sandbox.
    """

    # init=False so `replace()` in `for_run` produces a fresh instance with _repl=None,
    # giving each agent run isolated REPL state. Lazy-initialized on first call_tool.
    _repl: MontyRepl | None = field(default=None, init=False, repr=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset instance with isolated REPL state for this agent run."""
        # `replace()` creates a new instance — _repl resets to None since it's init=False,
        # so concurrent agents sharing the same toolset don't leak state between runs.
        wrapped = await self.wrapped.for_run(ctx)
        return replace(self, wrapped=wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Update the wrapped toolset for this step while preserving REPL state."""
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        # replace() resets _repl to None since it's init=False. Without this,
        # the LLM could set x=1 in step 1, then get a NameError for x in step 2
        # just because the wrapped toolset changed between turns.
        new_self = replace(self, wrapped=new_wrapped)
        new_self._repl = self._repl
        return new_self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the `run_code` tool definition."""
        # Wrapped tools are not fetched here — only needed at call time to build external functions.
        # Fetching lazily in call_tool avoids staleness if the wrapped toolset changes between steps.
        return {
            _RUN_CODE_TOOL_NAME: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=_RUN_CODE_TOOL_NAME,
                    description=_RUN_CODE_DESCRIPTION,
                    parameters_json_schema=_RUN_CODE_JSON_SCHEMA,
                ),
                max_retries=3,
                args_validator=cast(SchemaValidatorProt, _RUN_CODE_ARGS_VALIDATOR),
            ),
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Execute Python code in the sandbox, dispatching any external function calls to wrapped tools."""
        code = tool_args['code']
        restart = tool_args.get('restart', False)

        if self._repl is None or restart:
            self._repl = MontyRepl()

        original_tools = await self.wrapped.get_tools(ctx)

        # Pass None instead of {} when there are no tools — Monty treats them differently.
        external_functions = {
            t_name: _make_tool_callable(t_name, t, ctx) for t_name, t in original_tools.items()
        } or None

        output_chunks: list[str] = []

        def _print_callback(_stream: str, text: str) -> None:
            output_chunks.append(text)

        try:
            result = await self._repl.feed_run_async(
                code=code, external_functions=external_functions, print_callback=_print_callback
            )
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in code:\n{e.display()}') from e
        except MontyTypingError as e:
            raise ModelRetry(f'Type error in code:\n{e.display()}') from e
        except MontyRuntimeError as e:
            raise ModelRetry(f'Runtime error:\n{e.display()}') from e

        # TODO: For stdio-based driver runtimes (e.g. Monty's current driver loop),
        # print output goes to stdout which is also used for JSON protocol parsing.
        # This will need a different capture strategy for those runtimes.
        response: dict[str, Any] = {}
        printed = ''.join(output_chunks)
        if printed:
            response['output'] = printed
        if result is not None:
            response['result'] = result
        return response


def _make_tool_callable(tool_name: str, original_tool: ToolsetTool[Any], ctx: RunContext[Any]) -> Callable[..., Any]:
    async def wrapper(**kwargs: Any) -> Any:
        return await original_tool.toolset.call_tool(tool_name, kwargs, ctx, original_tool)

    return wrapper
