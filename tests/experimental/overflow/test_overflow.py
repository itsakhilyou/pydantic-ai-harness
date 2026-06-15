"""Tests for pydantic_ai_harness.experimental.overflow."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.experimental.overflow import (
    Band,
    LocalFileStore,
    OverflowingToolOutput,
    Passthrough,
    Spill,
    Summarize,
    Truncate,
    TruncationStrategy,
)
from pydantic_ai_harness.experimental.overflow._capability import (
    READ_TOOL_NAME,
    _build_spill_preview,
    _handle_key,
    _head_tail_preview,
    _merge_handle_metadata,
    _Payload,
    _read_slice,
    _select_action,
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
from pydantic_ai_harness.experimental.overflow._store import _safe_segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(*, run_id: str | None = 'run-1', retry: int = 0, usage: RunUsage | None = None) -> Any:
    """Build a minimal RunContext-like object for testing the hook directly."""

    @dataclasses.dataclass
    class _FakeModel:
        model_id: str = 'test-model'

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage
        run_id: str | None
        retry: int
        tool_call_id: str | None = 'call-1'
        model: Any = dataclasses.field(default_factory=_FakeModel)
        deps: None = None

    return _FakeCtx(usage=usage if usage is not None else RunUsage(), run_id=run_id, retry=retry)


def _call(tool_name: str = 'big_tool', tool_call_id: str = 'call-1') -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args='{}', tool_call_id=tool_call_id)


def _tool_def(name: str = 'big_tool') -> ToolDefinition:
    return ToolDefinition(name=name)


async def _run(cap: OverflowingToolOutput, result: Any, *, ctx: Any = None, tool_name: str = 'big_tool') -> Any:
    return await cap.after_tool_execute(
        ctx if ctx is not None else _make_ctx(),
        call=_call(tool_name),
        tool_def=_tool_def(tool_name),
        args={},
        result=result,
    )


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# ---------------------------------------------------------------------------
# _payload helpers
# ---------------------------------------------------------------------------


class TestPayloadHelpers:
    def test_strip_ansi(self):
        assert strip_ansi('\x1b[31mred\x1b[0m') == 'red'

    def test_is_binary(self):
        assert is_binary(b'x') is True
        assert is_binary(bytearray(b'x')) is True
        assert is_binary('x') is False

    def test_to_bytes_variants(self):
        assert to_bytes('hi') == b'hi'
        assert to_bytes(memoryview(b'mv')) == b'mv'
        assert to_bytes(bytearray(b'ba')) == b'ba'
        assert to_bytes({'a': 1}) == b'{"a":1}'

    def test_to_text_variants(self):
        assert to_text('hi') == 'hi'
        assert to_text({'a': 1}) == '{"a":1}'

    def test_measure_chars_and_tokens(self):
        assert measure('x' * 100, over_tokens=False, tokenizer=None) == 100
        assert measure('x' * 100, over_tokens=True, tokenizer=None) == 25
        assert measure('abcd', over_tokens=True, tokenizer=lambda s: len(s)) == 4

    def test_json_sketch_mapping(self):
        assert json_sketch({'a': 1, 'b': 'x'}) == "{'a': int, 'b': str}"

    def test_json_sketch_mapping_truncated(self):
        big = {f'k{i}': i for i in range(12)}
        sketch = json_sketch(big)
        assert sketch.endswith('... (12 keys)}')

    def test_json_sketch_sequence(self):
        assert json_sketch([1, 2, 3]) == '[3 items of int]'

    def test_json_sketch_empty_sequence(self):
        assert json_sketch([]) == '[0 items of empty]'

    def test_json_sketch_scalar(self):
        assert json_sketch(42) == ''
        assert json_sketch('plain') == ''

    def test_truncate_under_limit(self):
        assert truncate_text('short', 100, TruncationStrategy.head_tail) == 'short'

    def test_truncate_head(self):
        out = truncate_text('a' * 100, 10, TruncationStrategy.head)
        assert out.startswith('aaaaaaaaaa')
        assert 'showing first 10' in out

    def test_truncate_tail(self):
        out = truncate_text('a' * 100, 10, TruncationStrategy.tail)
        assert out.endswith('aaaaaaaaaa')
        assert 'showing last 10' in out

    def test_truncate_head_tail(self):
        out = truncate_text('a' * 100, 10, TruncationStrategy.head_tail)
        assert 'omitted from the middle' in out


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    def test_safe_segment(self):
        assert _safe_segment('a b!@#') == 'a_b_'
        assert _safe_segment('') == '_'
        assert _safe_segment('..') == '_'
        assert _safe_segment('.') == '_'
        assert _safe_segment('ok-1.2') == 'ok-1.2'

    def test_default_root(self):
        store = LocalFileStore()
        assert store._root.name == 'pyai_harness_overflow'

    async def test_write_read_roundtrip(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        handle = await store.write('run-1/call-1.0', b'payload')
        assert handle == 'run-1/call-1.0'
        assert await store.read(handle) == b'payload'

    async def test_empty_key(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        handle = await store.write('', b'data')
        assert await store.read(handle) == b'data'

    async def test_read_missing_raises(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        with pytest.raises(OSError):
            await store.read('nope/x.0')


# ---------------------------------------------------------------------------
# Capability construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_band_is_spill_then_truncate(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput()
        assert len(cap._bands) == 1
        action = cap._bands[0].action
        assert isinstance(action, Spill)
        assert isinstance(action.then, Truncate)

    def test_bands_sorted_descending(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=10, action=Truncate()), Band(over=100, action=Spill())]
        )
        assert [b.over for b in cap._bands] == [100, 10]

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match='non-negative'):
            OverflowingToolOutput(bands=[Band(over=-1, action=Passthrough())])

    def test_provided_store_used(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(store=store)
        assert cap._store is store

    def test_per_tool_prepared(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            per_tool={'read_file': [Band(over=5, action=Truncate())]}
        )
        assert 'read_file' in cap._per_tool


# ---------------------------------------------------------------------------
# Passthrough / filtering / guards
# ---------------------------------------------------------------------------


class TestPassthrough:
    async def test_read_tool_exempt(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=1, action=Truncate(max_chars=2))])
        out = await _run(cap, 'x' * 100, tool_name=READ_TOOL_NAME)
        assert out == 'x' * 100

    async def test_tool_filter_skips_unmatched(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=1, action=Truncate(max_chars=2))], tool_filter=['other']
        )
        out = await _run(cap, 'x' * 100)
        assert out == 'x' * 100

    async def test_callable_filter(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=1, action=Truncate(max_chars=2))],
            tool_filter=lambda ctx, td: td.name == 'big_tool',
        )
        out = await _run(cap, 'x' * 100)
        assert isinstance(out, str) and 'truncated' in out

    async def test_below_threshold_passthrough(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=1000, action=Truncate())])
        out = await _run(cap, 'small')
        assert out == 'small'

    async def test_exception_result_passthrough(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=1, action=Truncate(max_chars=2))])
        err = ValueError('boom')
        assert await _run(cap, err) is err


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    async def test_truncates_text(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=10, action=Truncate(max_chars=20, strategy=TruncationStrategy.head))]
        )
        out = await _run(cap, 'a' * 100)
        assert isinstance(out, str) and out.startswith('a' * 20)

    async def test_strip_ansi_applied(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=5, action=Truncate(max_chars=1000))], strip_ansi=True
        )
        out = await _run(cap, '\x1b[31m' + 'red text ' * 10 + '\x1b[0m')
        assert isinstance(out, str) and '\x1b[' not in out

    async def test_binary_truncate_falls_back_to_passthrough(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=1, action=Truncate())])
        data = b'\x00\x01' * 100
        assert await _run(cap, data) == data

    async def test_tool_return_envelope_preserved(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=10, action=Truncate(max_chars=20))])
        out = await _run(cap, ToolReturn(return_value='a' * 100, content='note', metadata={'k': 1}))
        assert isinstance(out, ToolReturn)
        assert out.content == 'note'
        assert out.metadata == {'k': 1}


# ---------------------------------------------------------------------------
# Spill
# ---------------------------------------------------------------------------


class TestSpill:
    async def test_spill_roundtrip(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=10, action=Spill(preview_chars=20))], store=store
        )
        text = 'line\n' * 1000
        out = await _run(cap, text)
        assert isinstance(out, ToolReturn)
        assert isinstance(out.return_value, str) and 'too large' in out.return_value
        handle = out.metadata['overflow_handle']
        assert handle == 'run-1/call-1.0'
        assert await store.read(handle) == text.encode('utf-8')

    async def test_spill_binary_verbatim(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=1, action=Spill())], store=store)
        data = b'\x00\xff' * 100
        out = await _run(cap, data)
        assert isinstance(out, ToolReturn)
        assert 'binary' in out.return_value  # type: ignore[operator]
        assert await store.read(out.metadata['overflow_handle']) == data

    async def test_spill_structured_includes_sketch(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=5, action=Spill())], store=store)
        out = await _run(cap, {'rows': list(range(1000)), 'ok': True})
        assert isinstance(out, ToolReturn)
        assert 'shape:' in out.return_value  # type: ignore[operator]

    async def test_spill_failure_falls_back_to_truncate(self):
        class _BrokenStore:
            async def write(self, key: str, data: bytes) -> str:
                raise OSError('disk full')

            async def read(self, handle: str) -> bytes:  # pragma: no cover - never reached
                raise FileNotFoundError(handle)

        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=10, action=Spill(then=Truncate(max_chars=15)))], store=_BrokenStore()
        )
        out = await _run(cap, 'a' * 100)
        assert isinstance(out, str) and 'truncated' in out

    async def test_spill_failure_no_fallback_returns_original(self):
        class _BrokenStore:
            async def write(self, key: str, data: bytes) -> str:
                raise OSError('disk full')

            async def read(self, handle: str) -> bytes:  # pragma: no cover - never reached
                raise FileNotFoundError(handle)

        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=10, action=Spill())], store=_BrokenStore()
        )
        out = await _run(cap, 'a' * 100)
        assert out == 'a' * 100

    async def test_handle_distinct_per_retry(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=5, action=Spill())], store=store)
        out0 = await _run(cap, 'a' * 100, ctx=_make_ctx(retry=0))
        out1 = await _run(cap, 'b' * 100, ctx=_make_ctx(retry=1))
        assert out0.metadata['overflow_handle'] != out1.metadata['overflow_handle']  # type: ignore[union-attr]

    async def test_spill_merges_existing_metadata(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=5, action=Spill())], store=store)
        out = await _run(cap, ToolReturn(return_value='a' * 100, metadata={'orig': True}))
        assert isinstance(out, ToolReturn)
        assert out.metadata['orig'] is True
        assert 'overflow_handle' in out.metadata


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    async def test_custom_sync_summarizer(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(summarize=lambda name, text: f'{name}:{len(text)}'))]
        )
        out = await _run(cap, 'x' * 100)
        assert out == 'big_tool:100'

    async def test_custom_async_summarizer(self):
        async def summ(name: str, text: str) -> str:
            return f'async:{len(text)}'

        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=5, action=Summarize(summarize=summ))])
        out = await _run(cap, 'x' * 100)
        assert out == 'async:100'

    async def test_inherited_model_summarize(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=5, action=Summarize())])
        mock_result = AsyncMock()
        mock_result.output = '  the summary  '
        with patch('pydantic_ai.Agent') as MockAgent:
            instance = AsyncMock()
            instance.run.return_value = mock_result
            MockAgent.return_value = instance
            out = await _run(cap, 'x' * 100)
        assert out == 'the summary'

    async def test_explicit_model_summarize(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(model='test:other'))]
        )
        mock_result = AsyncMock()
        mock_result.output = 'sum'
        with patch('pydantic_ai.Agent') as MockAgent:
            instance = AsyncMock()
            instance.run.return_value = mock_result
            MockAgent.return_value = instance
            out = await _run(cap, 'x' * 100)
        assert out == 'sum'

    async def test_binary_summarize_falls_back(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=1, action=Summarize(then=Passthrough()))]
        )
        data = b'\x00' * 100
        assert await _run(cap, data) == data

    async def test_summarize_failure_falls_back(self):
        def boom(name: str, text: str) -> str:
            raise RuntimeError('model down')

        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=5, action=Summarize(summarize=boom, then=Truncate(max_chars=10)))]
        )
        out = await _run(cap, 'a' * 100)
        assert isinstance(out, str) and 'truncated' in out


# ---------------------------------------------------------------------------
# Passthrough action + per-tool + band selection
# ---------------------------------------------------------------------------


class TestActionsAndSelection:
    async def test_passthrough_action(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(bands=[Band(over=1, action=Passthrough())])
        assert await _run(cap, 'x' * 100) == 'x' * 100

    async def test_per_tool_replaces_bands(self):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=1, action=Truncate(max_chars=5))],
            per_tool={'big_tool': [Band(over=100_000, action=Truncate())]},
        )
        # global band would truncate, but per_tool threshold is huge -> passthrough
        assert await _run(cap, 'x' * 100) == 'x' * 100

    def test_select_action_no_match(self):
        assert _select_action([Band(over=100, action=Passthrough())], 50) is None

    def test_select_action_first_match(self):
        bands = [Band(over=100, action=Spill()), Band(over=10, action=Truncate())]
        action = _select_action(bands, 50)
        assert isinstance(action, Truncate)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternals:
    def test_handle_key_defaults(self):
        ctx = _make_ctx(run_id=None, retry=2)
        ctx.tool_call_id = None
        key = _handle_key(ctx, ToolCallPart(tool_name='t', args='{}', tool_call_id=''))
        assert key == 'run/call.2'

    def test_merge_handle_metadata_non_mapping(self):
        meta = _merge_handle_metadata('not-a-mapping', 'h/1.0', 42)
        assert meta == {'overflow_handle': 'h/1.0', 'overflow_bytes': 42}

    def test_head_tail_preview_under(self):
        assert _head_tail_preview('short', 1000) == 'short'

    def test_head_tail_preview_over(self):
        out = _head_tail_preview('a' * 100, 10)
        assert 'omitted' in out

    def test_build_spill_preview_tokens_unit(self):
        payload = _Payload(
            value='x' * 100,
            binary=False,
            text='x' * 100,
            data=b'x' * 100,
            original='x' * 100,
            was_tool_return=False,
            content=None,
            metadata=None,
        )
        preview = _build_spill_preview('h/1.0', payload, 20, over_tokens=True)
        assert 'tokens' in preview


# ---------------------------------------------------------------------------
# read_tool_result / _read_slice
# ---------------------------------------------------------------------------


class TestReadBack:
    async def test_read_slice_basic(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'line {i}' for i in range(50)).encode('utf-8'))
        out = await _read_slice(store, 'h/1.0', offset=0, limit=3, from_end=False, pattern=None)
        assert 'line 0' in out and 'line 2' in out and 'line 3' not in out

    async def test_read_slice_from_end(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'line {i}' for i in range(50)).encode('utf-8'))
        out = await _read_slice(store, 'h/1.0', offset=0, limit=2, from_end=True, pattern=None)
        assert 'line 49' in out and 'line 48' in out

    async def test_read_slice_pattern(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'apple\nbanana\navocado\ncherry')
        out = await _read_slice(store, 'h/1.0', offset=0, limit=200, from_end=False, pattern='^a')
        assert 'apple' in out and 'avocado' in out and 'banana' not in out

    async def test_read_slice_missing_handle(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        with pytest.raises(ModelRetry, match='No stored tool result'):
            await _read_slice(store, 'missing/1.0', offset=0, limit=10, from_end=False, pattern=None)

    async def test_read_slice_bad_pattern(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='Invalid pattern'):
            await _read_slice(store, 'h/1.0', offset=0, limit=10, from_end=False, pattern='[')

    async def test_get_toolset_registers_read_tool(self, tmp_path: Path):
        store = LocalFileStore(base_dir=tmp_path)
        await store.write('h/1.0', b'hello\nworld')
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(store=store)
        toolset = cap.get_toolset()
        assert toolset is not None
        tool = toolset.tools[READ_TOOL_NAME]  # type: ignore[union-attr]
        out = await tool.function(_make_ctx(), 'h/1.0')  # type: ignore[attr-defined]
        assert 'hello' in out


# ---------------------------------------------------------------------------
# Agent-path integration
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    async def test_spill_persists_in_history(self, tmp_path: Path, anyio_backend: str):
        store = LocalFileStore(base_dir=tmp_path)
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=100, action=Spill(preview_chars=50))], store=store
        )
        agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

        @agent.tool_plain
        def big_tool() -> str:
            return 'data line\n' * 500

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        spilled = [p for p in returns if p.tool_name == 'big_tool']
        assert spilled
        part = spilled[0]
        assert isinstance(part.content, str) and 'too large' in part.content
        assert part.metadata is not None and 'overflow_handle' in part.metadata
        assert await store.read(part.metadata['overflow_handle']) == ('data line\n' * 500).encode('utf-8')

    async def test_small_output_untouched(self, tmp_path: Path, anyio_backend: str):
        cap: OverflowingToolOutput[None] = OverflowingToolOutput(
            bands=[Band(over=10_000, action=Spill())], store=LocalFileStore(base_dir=tmp_path)
        )
        agent = Agent(TestModel(call_tools=['small_tool']), capabilities=[cap])

        @agent.tool_plain
        def small_tool() -> str:
            return 'tiny'

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        small = [p for p in returns if p.tool_name == 'small_tool']
        assert small and small[0].content == 'tiny'
