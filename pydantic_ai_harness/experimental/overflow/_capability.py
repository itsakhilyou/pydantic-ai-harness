"""`OverflowingToolOutput` -- reduce oversized tool returns at production time."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.experimental.overflow._bands import (
    Action,
    Band,
    Passthrough,
    Spill,
    Summarize,
    Truncate,
)
from pydantic_ai_harness.experimental.overflow._payload import (
    is_binary,
    json_sketch,
    measure,
    strip_ansi,
    to_bytes,
    to_text,
    truncate_text,
)
from pydantic_ai_harness.experimental.overflow._store import LocalFileStore, OverflowStore

READ_TOOL_NAME = 'read_tool_result'
"""Name of the registered read-back tool. Its own returns are exempt from reduction."""

_DEFAULT_THRESHOLD = 10_000
"""Default band threshold (characters) -- below this, returns pass through untouched."""

_DEFAULT_SUMMARY_PROMPT = """\
The following output from the `{tool_name}` tool is too large to keep in full. Summarize it \
so the summary carries everything needed to keep working: concrete values, identifiers, \
errors, and structure. Respond ONLY with the summary, no preamble.

<output>
{output}
</output>\
"""


def _default_bands() -> list[Band]:
    """Lossless spill with a bounded truncation fallback: zero LLM cost, no silent drop."""
    return [Band(over=_DEFAULT_THRESHOLD, action=Spill(then=Truncate()))]


@dataclass
class _Payload:
    """Everything the reduction pipeline needs about one tool return."""

    value: object
    binary: bool
    text: str | None
    data: bytes
    original: Any
    was_tool_return: bool
    content: Any
    metadata: Any


@dataclass
class OverflowingToolOutput(AbstractCapability[AgentDepsT]):
    """Reduce oversized tool returns when they are produced, persisting the reduction.

    A tool can return a payload large enough to dominate the context window. Tool returns
    persist in history, so an oversized one is re-sent on every later request. This
    capability intercepts a return in `after_tool_execute`, reduces it once, and lets the
    reduced form persist -- it is not recomputed per request.

    Three reduction modes, freely combined through an ordered list of size `bands`:

    - `Truncate`: clamp to a character budget. Lossy, zero-cost.
    - `Spill`: persist the full payload, hand the model a `read_tool_result` handle plus a
      preview. Lossless.
    - `Summarize`: size-gated LLM summary. Inherits the run's model by default.

    The first band whose `over` threshold the measured size meets wins; smaller returns pass
    through. `per_tool` replaces the band list for named tools; `tool_filter` scopes which
    tools are touched at all. The default is `Spill(then=Truncate())`: lossless when a store
    accepts the write, a bounded truncation otherwise.

    `ModelRetry` and other errors never reach this hook (they are raised, not returned), so
    error payloads the model needs to recover are never spilled or summarized.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.experimental.overflow import (
            Band,
            OverflowingToolOutput,
            Spill,
            Summarize,
            Truncate,
        )

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[
                OverflowingToolOutput(
                    bands=[
                        Band(over=100_000, action=Spill()),
                        Band(over=20_000, action=Summarize()),
                        Band(over=5_000, action=Truncate()),
                    ],
                )
            ],
        )
        ```
    """

    bands: Sequence[Band] = field(default_factory=_default_bands)
    """Ordered size bands. The first band whose `over` threshold is met wins."""

    per_tool: Mapping[str, Sequence[Band]] = field(default_factory=dict[str, Sequence[Band]])
    """Per-tool band lists that replace `bands` for the named tools."""

    tool_filter: ToolSelector[AgentDepsT] = 'all'
    """Which tools this capability touches. Non-matching tools always pass through."""

    over_tokens: bool = False
    """Measure band thresholds in estimated tokens instead of characters."""

    tokenizer: Any = None
    """Optional `(str) -> int` tokenizer for `over_tokens`. Defaults to a ~4-char heuristic."""

    store: OverflowStore | None = None
    """Backend for spilled payloads. Defaults to a `LocalFileStore`."""

    strip_ansi: bool = False
    """Strip ANSI escape sequences from text returns before measuring and reducing."""

    summary_prompt: str = _DEFAULT_SUMMARY_PROMPT
    """Prompt template for `Summarize`. Must contain `{tool_name}` and `{output}`."""

    _store: OverflowStore = field(init=False, repr=False)
    _bands: list[Band] = field(init=False, repr=False)
    _per_tool: dict[str, list[Band]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = self.store if self.store is not None else LocalFileStore()
        self._bands = self._prepare_bands(self.bands)
        self._per_tool = {name: self._prepare_bands(bands) for name, bands in self.per_tool.items()}

    @staticmethod
    def _prepare_bands(bands: Sequence[Band]) -> list[Band]:
        """Validate thresholds and order bands largest-first so first-match means largest-fit."""
        for band in bands:
            if band.over < 0:
                raise ValueError('Band.over must be non-negative.')
        return sorted(bands, key=lambda b: b.over, reverse=True)

    # --- toolset ---

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Register the `read_tool_result` tool for reading spilled payloads on demand."""
        store = self._store

        async def read_tool_result(
            ctx: RunContext[AgentDepsT],
            handle: str,
            offset: int = 0,
            limit: int = 200,
            from_end: bool = False,
            pattern: str | None = None,
        ) -> str:
            """Read a slice of a spilled tool result.

            Args:
                ctx: The run context (supplied by the agent).
                handle: The handle from the overflowed tool return.
                offset: Number of matching lines to skip from the start (or end).
                limit: Maximum number of lines to return.
                from_end: Count `offset`/`limit` from the end of the result.
                pattern: Optional regular expression; only matching lines are returned.
            """
            return await _read_slice(store, handle, offset, limit, from_end, pattern)

        return FunctionToolset([read_tool_result])

    # --- reduction ---

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Reduce the tool result when it overflows the matching band's threshold."""
        if call.tool_name == READ_TOOL_NAME:
            return result
        if not await matches_tool_selector(self.tool_filter, ctx, tool_def):
            return result

        payload = self._build_payload(result)
        if payload is None:
            return result

        bands = self._per_tool.get(call.tool_name, self._bands)
        size = (
            len(payload.data)
            if payload.binary
            else measure(payload.text or '', over_tokens=self.over_tokens, tokenizer=self.tokenizer)
        )
        action = _select_action(bands, size)
        if action is None:
            return payload.original

        return await self._apply(ctx, call, action, payload)

    def _build_payload(self, result: Any) -> _Payload | None:
        """Unwrap a `ToolReturn`, skip error payloads, and pre-render text/bytes.

        Returns None when the result must not be reduced (an exception payload).
        """
        if isinstance(result, ToolReturn):
            value: object = result.return_value
            content = result.content
            metadata = result.metadata
            was_tool_return = True
        else:
            value = result
            content = None
            metadata = None
            was_tool_return = False

        if isinstance(value, BaseException):
            return None

        binary = is_binary(value)
        text: str | None = None
        if not binary:
            text = to_text(value)
            if self.strip_ansi:
                text = strip_ansi(text)
            data = text.encode('utf-8')
        else:
            data = to_bytes(value)

        return _Payload(
            value=value,
            binary=binary,
            text=text,
            data=data,
            original=result,
            was_tool_return=was_tool_return,
            content=content,
            metadata=metadata,
        )

    async def _apply(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Action,
        payload: _Payload,
    ) -> Any:
        """Apply one action, falling back to its `then` when the action cannot run."""
        if isinstance(action, Passthrough):
            return payload.original

        if isinstance(action, Truncate):
            if payload.binary:
                return await self._fallback(ctx, call, action.then, payload)
            assert payload.text is not None
            return self._rebuild(payload, truncate_text(payload.text, action.max_chars, action.strategy))

        if isinstance(action, Spill):
            return await self._spill(ctx, call, action, payload)

        return await self._summarize_action(ctx, call, action, payload)

    async def _fallback(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        then: Action | None,
        payload: _Payload,
    ) -> Any:
        """Run the fallback action, or return the original return when there is none."""
        if then is None:
            return payload.original
        return await self._apply(ctx, call, then, payload)

    async def _spill(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Spill,
        payload: _Payload,
    ) -> Any:
        key = _handle_key(ctx, call)
        try:
            handle = await self._store.write(key, payload.data)
        except Exception:
            return await self._fallback(ctx, call, action.then, payload)

        preview = _build_spill_preview(handle, payload, action.preview_chars, over_tokens=self.over_tokens)
        metadata = _merge_handle_metadata(payload.metadata, handle, len(payload.data))
        return ToolReturn(return_value=preview, content=payload.content, metadata=metadata)

    async def _summarize_action(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Summarize,
        payload: _Payload,
    ) -> Any:
        if payload.binary:
            return await self._fallback(ctx, call, action.then, payload)
        assert payload.text is not None
        try:
            summary = await self._summarize(ctx, call, action, payload.text)
        except Exception:
            return await self._fallback(ctx, call, action.then, payload)
        return self._rebuild(payload, summary)

    async def _summarize(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Summarize,
        text: str,
    ) -> str:
        """Generate the summary via a custom callable or the inherited-model agent."""
        if action.summarize is not None:
            outcome = action.summarize(call.tool_name, text)
            if isinstance(outcome, Awaitable):
                return await outcome
            return outcome

        from pydantic_ai import Agent

        model = action.model if action.model is not None else ctx.model
        prompt = self.summary_prompt.format(tool_name=call.tool_name, output=text)
        agent: Agent[None, str] = Agent(model, instructions='You summarize oversized tool output.')
        run = await agent.run(prompt, usage=ctx.usage)
        return run.output.strip()

    @staticmethod
    def _rebuild(payload: _Payload, new_text: str) -> Any:
        """Return the reduced text, preserving the `ToolReturn` envelope when there was one."""
        if payload.was_tool_return:
            return ToolReturn(return_value=new_text, content=payload.content, metadata=payload.metadata)
        return new_text


def _select_action(bands: Sequence[Band], size: int) -> Action | None:
    """Return the first (largest-threshold) band action whose threshold `size` meets."""
    for band in bands:
        if size >= band.over:
            return band.action
    return None


def _handle_key(ctx: RunContext[AgentDepsT], call: ToolCallPart) -> str:
    """Build a per-run, per-call, per-retry key so concurrent and retried calls never clash."""
    run_id = ctx.run_id or 'run'
    call_id = call.tool_call_id or 'call'
    return f'{run_id}/{call_id}.{ctx.retry}'


def _merge_handle_metadata(existing: Any, handle: str, byte_size: int) -> dict[str, Any]:
    """Stash the handle in `ToolReturn.metadata` (app-only, costs no model tokens)."""
    base: dict[str, Any] = {}
    if isinstance(existing, Mapping):
        base.update(_copy_mapping(existing))  # pyright: ignore[reportUnknownArgumentType]
    base['overflow_handle'] = handle
    base['overflow_bytes'] = byte_size
    return base


def _copy_mapping(source: Mapping[Any, Any]) -> dict[str, Any]:
    """Copy an arbitrary mapping with stringified keys (tool metadata is app-defined)."""
    return {str(key): source[key] for key in source}


def _build_spill_preview(handle: str, payload: _Payload, preview_chars: int, *, over_tokens: bool) -> str:
    """Compose the model-visible spill stand-in: marker, sketch, and a head/tail preview."""
    if payload.binary:
        size_desc = f'{len(payload.data):,} bytes (binary)'
        body = f'<{len(payload.data):,} bytes of binary data>'
        sketch = ''
    else:
        text = payload.text or ''
        unit = 'tokens' if over_tokens else 'chars'
        amount = measure(text, over_tokens=over_tokens, tokenizer=None) if over_tokens else len(text)
        size_desc = f'{amount:,} {unit}'
        body = _head_tail_preview(text, preview_chars)
        sketch = json_sketch(payload.value)

    header = (
        f'[Tool output too large ({size_desc}); stored to handle {handle!r}. '
        f'Read it with read_tool_result(handle={handle!r}, offset=0, limit=200, '
        f'from_end=False, pattern=None).]'
    )
    parts = [header]
    if sketch:
        parts.append(f'shape: {sketch}')
    parts.append(body)
    return '\n'.join(parts)


def _head_tail_preview(text: str, preview_chars: int) -> str:
    """Return a head+tail slice of `text` with a middle-elision marker."""
    if len(text) <= preview_chars:
        return text
    head_chars = preview_chars // 2
    tail_chars = preview_chars - head_chars
    omitted = len(text) - head_chars - tail_chars
    return f'{text[:head_chars]}\n...[{omitted:,} chars omitted]...\n{text[-tail_chars:]}'


async def _read_slice(
    store: OverflowStore,
    handle: str,
    offset: int,
    limit: int,
    from_end: bool,
    pattern: str | None,
) -> str:
    """Read, optionally grep, and slice a spilled payload for `read_tool_result`."""
    try:
        data = await store.read(handle)
    except OSError as exc:
        raise ModelRetry(f'No stored tool result for handle {handle!r}: {exc}.') from exc

    lines = data.decode('utf-8', errors='replace').splitlines()
    if pattern is not None:
        try:
            matcher = re.compile(pattern)
        except re.error as exc:
            raise ModelRetry(f'Invalid pattern {pattern!r}: {exc}.') from exc
        lines = [line for line in lines if matcher.search(line)]

    total = len(lines)
    if from_end:
        end = max(0, total - offset)
        window = lines[max(0, end - limit) : end]
    else:
        window = lines[offset : offset + limit]

    header = f'[handle {handle!r}: {total:,} line(s) available; showing {len(window)}]'
    return '\n'.join([header, *window])
