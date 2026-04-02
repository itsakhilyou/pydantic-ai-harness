"""Tests for ToolOutputManagement capability."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from pydantic_harness.tool_output_management import (
    ToolOutputManagement,
    TruncationStrategy,
)

# Re-export private helpers for unit testing; pyright: ignore for private usage.
from pydantic_harness.tool_output_management import (
    _head_tail_default_split as head_tail_default_split,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_harness.tool_output_management import (
    _is_binary as is_binary,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_harness.tool_output_management import (
    _stringify as stringify,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_harness.tool_output_management import (
    _truncate as truncate,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_harness.tool_output_management import (
    _truncate_by_lines as truncate_by_lines,  # pyright: ignore[reportPrivateUsage]
)

CALL = ToolCallPart(tool_name='my_tool', args={})
TOOL_DEF = ToolDefinition(name='my_tool', description='test tool', parameters_json_schema={})


# ---------------------------------------------------------------------------
# Unit tests: stringify
# ---------------------------------------------------------------------------


class TestStringify:
    def test_string_passthrough(self) -> None:
        assert stringify('hello') == 'hello'

    def test_int(self) -> None:
        assert stringify(42) == '42'

    def test_dict(self) -> None:
        result = stringify({'key': 'value'})
        assert 'key' in result
        assert 'value' in result

    def test_list(self) -> None:
        assert stringify([1, 2, 3]) == '[1, 2, 3]'

    def test_none(self) -> None:
        assert stringify(None) == 'None'


# ---------------------------------------------------------------------------
# Unit tests: head_tail_default_split
# ---------------------------------------------------------------------------


class TestHeadTailSplit:
    def test_split_100(self) -> None:
        head, tail = head_tail_default_split(100)
        assert head == 60
        assert tail == 40

    def test_split_sums_to_limit(self) -> None:
        for limit in (1, 10, 77, 1000, 9999):
            head, tail = head_tail_default_split(limit)
            assert head + tail == limit


# ---------------------------------------------------------------------------
# Unit tests: truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_no_truncation_needed(self) -> None:
        text = 'short'
        assert truncate(text, 100, TruncationStrategy.head) == text

    def test_head_strategy(self) -> None:
        text = 'a' * 200
        result = truncate(text, 50, TruncationStrategy.head)
        assert result.startswith('a' * 50)
        assert '[Truncated: showing first 50 of 200 chars]' in result

    def test_tail_strategy(self) -> None:
        text = 'a' * 200
        result = truncate(text, 50, TruncationStrategy.tail)
        assert result.endswith('a' * 50)
        assert '[Truncated: showing last 50 of 200 chars]' in result

    def test_head_tail_strategy(self) -> None:
        text = 'H' * 100 + 'M' * 800 + 'T' * 100
        result = truncate(text, 100, TruncationStrategy.head_tail)
        # head=60 chars, tail=40 chars
        assert result.startswith('H' * 60)
        assert result.endswith('T' * 40)
        assert 'omitted from middle' in result
        assert '900' in result  # 1000 - 60 - 40 = 900 omitted

    def test_head_tail_exact_boundary(self) -> None:
        text = 'x' * 100
        # Exactly at limit -> no truncation
        assert truncate(text, 100, TruncationStrategy.head_tail) == text

    def test_comma_formatting(self) -> None:
        text = 'a' * 100_000
        result = truncate(text, 1000, TruncationStrategy.head)
        assert '100,000' in result


# ---------------------------------------------------------------------------
# Integration tests: ToolOutputManagement.after_tool_execute
# ---------------------------------------------------------------------------


class TestAfterToolExecute:
    @pytest.mark.anyio
    async def test_short_output_unchanged(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=100)
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='short output',
        )
        assert result == 'short output'

    @pytest.mark.anyio
    async def test_long_string_truncated(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=50)
        long_text = 'x' * 200
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=long_text,
        )
        assert isinstance(result, str)
        assert len(result) < len(long_text)
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_non_string_result_truncated(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=20)
        big_dict = {'key': 'v' * 200}
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=big_dict,
        )
        assert isinstance(result, str)
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_per_tool_limit(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=10_000,
            per_tool_limits={'special_tool': 20},
        )
        call = ToolCallPart(tool_name='special_tool', args={})
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=call,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 100,
        )
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_per_tool_limit_does_not_affect_others(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=10_000,
            per_tool_limits={'other_tool': 5},
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 100,  # under 10_000
        )
        assert result == 'a' * 100

    @pytest.mark.anyio
    async def test_per_tool_strategy(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            per_tool_strategies={'tail_tool': TruncationStrategy.tail},
        )
        call = ToolCallPart(tool_name='tail_tool', args={})
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=call,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 200,
        )
        assert 'showing last' in result

    @pytest.mark.anyio
    async def test_head_strategy_via_config(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            strategy=TruncationStrategy.head,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 200,
        )
        assert 'showing first' in result

    @pytest.mark.anyio
    async def test_sync_summarize_fn(self) -> None:
        def summarize(tool_name: str, output: str) -> str:
            return f'Summary of {tool_name}: {len(output)} chars'

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100,
            summarize_fn=summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='x' * 200,
        )
        assert result == 'Summary of my_tool: 200 chars'

    @pytest.mark.anyio
    async def test_async_summarize_fn(self) -> None:
        async def summarize(tool_name: str, output: str) -> str:
            return f'Async summary: {len(output)} chars'

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100,
            summarize_fn=summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='x' * 200,
        )
        assert result == 'Async summary: 200 chars'

    @pytest.mark.anyio
    async def test_summarize_fn_safety_net(self) -> None:
        """If summarize_fn returns something still too long, truncation kicks in."""

        def bad_summarize(tool_name: str, output: str) -> str:
            return output  # returns full output, still too long

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            summarize_fn=bad_summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='x' * 200,
        )
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_summarize_fn_not_called_under_limit(self) -> None:
        calls: list[str] = []

        def summarize(tool_name: str, output: str) -> str:
            calls.append(tool_name)
            return 'summarized'

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100,
            summarize_fn=summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='short',
        )
        assert result == 'short'
        assert calls == []

    @pytest.mark.anyio
    async def test_original_returned_when_exactly_at_limit(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=10)
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 10,
        )
        assert result == 'a' * 10


# ---------------------------------------------------------------------------
# Test public exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_import_from_package(self) -> None:
        from pydantic_harness import ToolOutputManagement, TruncationStrategy

        assert ToolOutputManagement is not None
        assert TruncationStrategy is not None


# ---------------------------------------------------------------------------
# Unit tests: is_binary
# ---------------------------------------------------------------------------


class TestIsBinary:
    def test_bytes(self) -> None:
        assert is_binary(b'\x00\x01\x02') is True

    def test_bytearray(self) -> None:
        assert is_binary(bytearray(b'\xff')) is True

    def test_memoryview(self) -> None:
        assert is_binary(memoryview(b'hello')) is True

    def test_str_not_binary(self) -> None:
        assert is_binary('hello') is False

    def test_int_not_binary(self) -> None:
        assert is_binary(42) is False

    def test_none_not_binary(self) -> None:
        assert is_binary(None) is False


# ---------------------------------------------------------------------------
# Unit tests: truncate_by_lines
# ---------------------------------------------------------------------------


class TestTruncateByLines:
    def test_no_truncation_needed(self) -> None:
        text = 'line1\nline2\nline3'
        assert truncate_by_lines(text, 5, TruncationStrategy.head) == text

    def test_head_strategy(self) -> None:
        text = '\n'.join(f'line{i}' for i in range(20))
        result = truncate_by_lines(text, 5, TruncationStrategy.head)
        assert result.startswith('line0\n')
        assert '[Truncated: showing first 5 of 20 lines]' in result

    def test_tail_strategy(self) -> None:
        text = '\n'.join(f'line{i}' for i in range(20))
        result = truncate_by_lines(text, 5, TruncationStrategy.tail)
        assert 'line19' in result
        assert '[Truncated: showing last 5 of 20 lines]' in result

    def test_head_tail_strategy(self) -> None:
        text = '\n'.join(f'line{i}' for i in range(100))
        result = truncate_by_lines(text, 10, TruncationStrategy.head_tail)
        # head=6 lines, tail=4 lines
        assert 'line0' in result
        assert 'line99' in result
        assert 'omitted from middle' in result
        assert '90' in result  # 100 - 6 - 4 = 90 omitted

    def test_exact_boundary(self) -> None:
        text = 'line1\nline2\nline3'
        assert truncate_by_lines(text, 3, TruncationStrategy.head) == text


# ---------------------------------------------------------------------------
# Integration tests: line-count limits
# ---------------------------------------------------------------------------


class TestLineCountLimits:
    @pytest.mark.anyio
    async def test_line_limit_triggers_before_char_limit(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100_000,
            max_output_lines=5,
            strategy=TruncationStrategy.head,
        )
        text = '\n'.join(f'line{i}' for i in range(20))
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=text,
        )
        assert 'showing first 5 of 20 lines' in result

    @pytest.mark.anyio
    async def test_char_limit_triggers_before_line_limit(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            max_output_lines=1000,
            strategy=TruncationStrategy.head,
        )
        text = 'x' * 200
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=text,
        )
        assert 'showing first 50 of 200 chars' in result

    @pytest.mark.anyio
    async def test_under_both_limits(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=1000,
            max_output_lines=10,
        )
        text = 'short\ntext'
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=text,
        )
        assert result == text

    @pytest.mark.anyio
    async def test_per_tool_line_limit(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100_000,
            max_output_lines=100,
            per_tool_line_limits={'my_tool': 3},
            strategy=TruncationStrategy.head,
        )
        text = '\n'.join(f'line{i}' for i in range(10))
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=text,
        )
        assert 'showing first 3 of 10 lines' in result

    @pytest.mark.anyio
    async def test_line_limit_only(self) -> None:
        """When max_output_lines is set but char limit is very high, line limit alone fires."""
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=1_000_000,
            max_output_lines=3,
            strategy=TruncationStrategy.tail,
        )
        text = '\n'.join(f'line{i}' for i in range(10))
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=text,
        )
        assert 'showing last 3 of 10 lines' in result


# ---------------------------------------------------------------------------
# Integration tests: spill-to-file
# ---------------------------------------------------------------------------


class TestSpillToFile:
    @pytest.mark.anyio
    async def test_spill_creates_file(self, tmp_path: Path) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            spill_to_file=True,
            spill_dir=tmp_path,
        )
        long_text = 'x' * 200
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=long_text,
        )
        assert isinstance(result, str)
        assert '[Full output (200 chars) saved to' in result
        assert 'Truncated' in result

        # Verify the file actually exists and contains the full output
        spill_files = list(tmp_path.glob('tool_output_*.txt'))
        assert len(spill_files) == 1
        assert spill_files[0].read_text(encoding='utf-8') == long_text

    @pytest.mark.anyio
    async def test_spill_default_dir(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            spill_to_file=True,
        )
        long_text = 'x' * 200
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=long_text,
        )
        assert '[Full output (200 chars) saved to' in result

    @pytest.mark.anyio
    async def test_spill_not_triggered_under_limit(self, tmp_path: Path) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=1000,
            spill_to_file=True,
            spill_dir=tmp_path,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='short',
        )
        assert result == 'short'
        assert list(tmp_path.glob('tool_output_*.txt')) == []

    @pytest.mark.anyio
    async def test_spill_with_summarize_fn_safety_net(self, tmp_path: Path) -> None:
        """When summarize_fn still produces oversized output, spill kicks in."""

        def bad_summarize(tool_name: str, output: str) -> str:
            return output  # returns full output, still too long

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            summarize_fn=bad_summarize,
            spill_to_file=True,
            spill_dir=tmp_path,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='x' * 200,
        )
        assert '[Full output' in result
        assert len(list(tmp_path.glob('tool_output_*.txt'))) == 1


# ---------------------------------------------------------------------------
# Integration tests: binary detection
# ---------------------------------------------------------------------------


class TestBinaryDetection:
    @pytest.mark.anyio
    async def test_bytes_result(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=50)
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=b'\x89PNG\r\n\x1a\n' + b'\x00' * 1000,
        )
        assert result == '[Binary data, 1,008 bytes]'

    @pytest.mark.anyio
    async def test_bytearray_result(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=50)
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=bytearray(b'\xff' * 256),
        )
        assert result == '[Binary data, 256 bytes]'

    @pytest.mark.anyio
    async def test_memoryview_result(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=50)
        data = b'hello world'
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=memoryview(data),
        )
        assert result == '[Binary data, 11 bytes]'

    @pytest.mark.anyio
    async def test_small_bytes_still_detected(self) -> None:
        """Binary detection applies regardless of size -- even small bytes are replaced."""
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=10_000)
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=b'hi',
        )
        assert result == '[Binary data, 2 bytes]'
