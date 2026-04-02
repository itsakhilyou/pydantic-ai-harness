"""Tool output management capability.

Intercepts tool return values and truncates or summarizes large outputs
to prevent context window blowup. Uses the `after_tool_execute` hook so
that the original tool result is preserved in telemetry / trajectory logs,
while only the LLM sees the truncated version.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


class TruncationStrategy(str, Enum):
    """Strategy for truncating oversized tool output."""

    head = 'head'
    """Keep only the first characters."""

    tail = 'tail'
    """Keep only the last characters."""

    head_tail = 'head_tail'
    """Keep the first and last characters, eliding the middle."""


SummarizeFn = Callable[[str, str], str | Awaitable[str]]
"""A function `(tool_name, output) -> summarized_output`.

May be sync or async.
"""


def _head_tail_default_split(limit: int) -> tuple[int, int]:
    """Split a character limit into head and tail portions (60/40)."""
    head = int(limit * 0.6)
    tail = limit - head
    return head, tail


def _truncate(text: str, limit: int, strategy: TruncationStrategy) -> str:
    """Apply a truncation strategy to *text* that exceeds *limit* chars."""
    total = len(text)
    if total <= limit:
        return text

    if strategy is TruncationStrategy.head:
        kept = text[:limit]
        return f'{kept}\n\n[Truncated: showing first {limit:,} of {total:,} chars]'

    if strategy is TruncationStrategy.tail:
        kept = text[-limit:]
        return f'[Truncated: showing last {limit:,} of {total:,} chars]\n\n{kept}'

    # head_tail
    head_chars, tail_chars = _head_tail_default_split(limit)
    head_part = text[:head_chars]
    tail_part = text[-tail_chars:]
    omitted = total - head_chars - tail_chars
    return (
        f'{head_part}\n\n'
        f'[Truncated: {omitted:,} chars omitted from middle; showing first {head_chars:,} + last {tail_chars:,} of {total:,} chars]\n\n'
        f'{tail_part}'
    )


def _truncate_by_lines(text: str, limit: int, strategy: TruncationStrategy) -> str:
    """Apply a truncation strategy to *text* that exceeds *limit* lines."""
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if total <= limit:
        return text

    if strategy is TruncationStrategy.head:
        kept = ''.join(lines[:limit])
        return f'{kept}\n\n[Truncated: showing first {limit:,} of {total:,} lines]'

    if strategy is TruncationStrategy.tail:
        kept = ''.join(lines[-limit:])
        return f'[Truncated: showing last {limit:,} of {total:,} lines]\n\n{kept}'

    # head_tail
    head_lines, tail_lines = _head_tail_default_split(limit)
    head_part = ''.join(lines[:head_lines])
    tail_part = ''.join(lines[-tail_lines:])
    omitted = total - head_lines - tail_lines
    return (
        f'{head_part}\n\n'
        f'[Truncated: {omitted:,} lines omitted from middle; showing first {head_lines:,} + last {tail_lines:,} of {total:,} lines]\n\n'
        f'{tail_part}'
    )


def _is_binary(value: Any) -> bool:
    """Return True if *value* is binary data that should not be truncated."""
    return isinstance(value, (bytes, bytearray, memoryview))


def _stringify(value: Any) -> str:
    """Convert an arbitrary tool return value to a string for size measurement."""
    if isinstance(value, str):
        return value
    return str(value)


@dataclass
class ToolOutputManagement(AbstractCapability[AgentDepsT]):
    """Manage large tool outputs to prevent context window blowup.

    Intercepts tool return values via the `after_tool_execute` hook and
    truncates or summarizes them when they exceed a configurable character
    limit.  The original (full) result is preserved upstream (telemetry,
    `FunctionToolResultEvent.content`); only the value forwarded to the
    model is modified.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import ToolOutputManagement

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[
                ToolOutputManagement(max_output_chars=8000),
            ],
        )
        ```
    """

    max_output_chars: int = 10_000
    """Default character limit for tool outputs.  Outputs exceeding this
    are truncated according to `strategy`."""

    max_output_lines: int | None = None
    """Optional line-count limit for tool outputs.

    When set, output is also checked against this line limit.  If both
    `max_output_chars` and `max_output_lines` are set, the limit that
    triggers first wins.
    """

    strategy: TruncationStrategy = TruncationStrategy.head_tail
    """Default truncation strategy applied when output exceeds
    `max_output_chars`."""

    per_tool_limits: dict[str, int] = field(default_factory=lambda: {})
    """Per-tool character limits.  Keys are tool names; values override
    `max_output_chars` for that tool."""

    per_tool_line_limits: dict[str, int] = field(default_factory=lambda: {})
    """Per-tool line-count limits.  Keys are tool names; values override
    `max_output_lines` for that tool."""

    per_tool_strategies: dict[str, TruncationStrategy] = field(default_factory=lambda: {})
    """Per-tool truncation strategies.  Keys are tool names; values
    override `strategy` for that tool."""

    summarize_fn: SummarizeFn | None = None
    """Optional summarization function called *instead of* truncation.

    Receives `(tool_name, full_output_str)` and must return a
    (potentially shorter) string.  If the returned string still exceeds
    the limit, it is truncated as a safety net.

    May be sync or async.
    """

    spill_to_file: bool = False
    """When True, oversized output is written to a temporary file and
    the model receives a pointer to that file plus a truncated preview.
    """

    spill_dir: Path | None = None
    """Directory for spill files.  Defaults to the system temp directory
    when `spill_to_file` is True and this is None.
    """

    def _exceeds_limits(self, text: str, char_limit: int, line_limit: int | None) -> bool:
        """Return True if *text* exceeds either the char or line limit."""
        if len(text) > char_limit:
            return True
        if line_limit is not None and text.count('\n') + 1 > line_limit:
            return True
        return False

    def _apply_truncation(
        self, text: str, char_limit: int, line_limit: int | None, strategy: TruncationStrategy
    ) -> str:
        """Truncate *text* by whichever limit fires first (lines or chars)."""
        # Check which limit fires first
        lines_exceed = line_limit is not None and text.count('\n') + 1 > line_limit
        chars_exceed = len(text) > char_limit

        if lines_exceed and line_limit is not None:
            # If both exceed, apply line truncation first, then char truncation
            # if still needed; if only lines exceed, just truncate by lines.
            truncated = _truncate_by_lines(text, line_limit, strategy)
            if chars_exceed and len(truncated) > char_limit:
                return _truncate(truncated, char_limit, strategy)
            return truncated

        # Only chars exceed (or neither, but caller already checked)
        return _truncate(text, char_limit, strategy)

    def _spill(self, text: str, char_limit: int, line_limit: int | None, strategy: TruncationStrategy) -> str:
        """Write *text* to a temp file and return a pointer with a truncated preview."""
        total_chars = len(text)
        dir_ = self.spill_dir or Path(tempfile.gettempdir())
        dir_.mkdir(parents=True, exist_ok=True)

        fd, path_str = tempfile.mkstemp(suffix='.txt', dir=str(dir_), prefix='tool_output_')
        path = Path(path_str)
        # Close the fd opened by mkstemp and write via Path
        os.close(fd)
        path.write_text(text, encoding='utf-8')

        preview = self._apply_truncation(text, char_limit, line_limit, strategy)
        return f'[Full output ({total_chars:,} chars) saved to {path}]\n{preview}'

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Truncate or summarize the tool result if it exceeds the configured limit."""
        # Binary detection: skip truncation entirely
        if _is_binary(result):
            size = len(result) if isinstance(result, (bytes, bytearray)) else result.nbytes
            return f'[Binary data, {size:,} bytes]'

        text = _stringify(result)
        char_limit = self.per_tool_limits.get(call.tool_name, self.max_output_chars)
        line_limit = self.per_tool_line_limits.get(call.tool_name, self.max_output_lines)

        if not self._exceeds_limits(text, char_limit, line_limit):
            return result

        strategy = self.per_tool_strategies.get(call.tool_name, self.strategy)

        # Summarize path
        if self.summarize_fn is not None:
            summary = self.summarize_fn(call.tool_name, text)
            if isinstance(summary, Awaitable):
                summary = await summary
            assert isinstance(summary, str)
            # Safety net: if the summary itself is still too long, truncate it
            if self._exceeds_limits(summary, char_limit, line_limit):
                if self.spill_to_file:
                    return self._spill(summary, char_limit, line_limit, strategy)
                return self._apply_truncation(summary, char_limit, line_limit, strategy)
            return summary

        # Spill-to-file path
        if self.spill_to_file:
            return self._spill(text, char_limit, line_limit, strategy)

        # Truncation path
        return self._apply_truncation(text, char_limit, line_limit, strategy)
