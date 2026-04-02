"""Tests for the FileSystem capability."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_harness.filesystem import FileSystem, format_lines

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Create a temporary root directory populated with test files."""
    (tmp_path / 'hello.txt').write_text('line one\nline two\nline three\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.py').write_text('print("hello")\n')
    (tmp_path / 'secret.env').write_text('API_KEY=abc123\n')
    return tmp_path


@pytest.fixture
def fs(tmp_root: Path) -> FileSystem:
    """A FileSystem capability rooted at the test directory."""
    return FileSystem(root_dir=tmp_root)


# ---------------------------------------------------------------------------
# format_lines
# ---------------------------------------------------------------------------


class TestFormatLines:
    def test_basic(self) -> None:
        result = format_lines('a\nb\nc\n', 0, 10)
        assert '     1\ta\n' in result
        assert '     2\tb\n' in result
        assert '     3\tc\n' in result

    def test_offset(self) -> None:
        result = format_lines('a\nb\nc\nd\n', 2, 1)
        assert '     3\tc\n' in result
        assert 'a' not in result.split('\t')[0]  # line 1 not present
        assert '1 more lines' in result

    def test_offset_past_end(self) -> None:
        with pytest.raises(ValueError, match='Offset 10 exceeds file length'):
            format_lines('a\n', 10, 1)

    def test_continuation_hint(self) -> None:
        result = format_lines('a\nb\nc\n', 0, 2)
        assert 'more lines' in result
        assert 'offset=2' in result

    def test_no_trailing_newline(self) -> None:
        result = format_lines('a', 0, 10)
        assert result.endswith('\n')
        assert '     1\ta' in result


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_existing(self, fs: FileSystem) -> None:
        result = fs.read_file('hello.txt')
        assert 'line one' in result
        assert 'line two' in result

    def test_read_with_offset(self, fs: FileSystem) -> None:
        result = fs.read_file('hello.txt', offset=1, limit=1)
        assert 'line two' in result
        assert 'line one' not in result

    def test_read_nested(self, fs: FileSystem) -> None:
        result = fs.read_file('sub/nested.py')
        assert 'print' in result

    def test_read_missing(self, fs: FileSystem) -> None:
        with pytest.raises(FileNotFoundError):
            fs.read_file('nonexistent.txt')

    def test_read_directory(self, fs: FileSystem) -> None:
        with pytest.raises(FileNotFoundError, match='is a directory'):
            fs.read_file('sub')

    def test_traversal_blocked(self, fs: FileSystem) -> None:
        with pytest.raises(PermissionError, match='outside the root'):
            fs.read_file('../../../etc/passwd')


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_new(self, fs: FileSystem, tmp_root: Path) -> None:
        result = fs.write_file('new.txt', 'hello world')
        assert 'Successfully wrote' in result
        assert (tmp_root / 'new.txt').read_text() == 'hello world'

    def test_write_creates_parents(self, fs: FileSystem, tmp_root: Path) -> None:
        fs.write_file('deep/nested/file.txt', 'content')
        assert (tmp_root / 'deep' / 'nested' / 'file.txt').read_text() == 'content'

    def test_write_overwrite(self, fs: FileSystem, tmp_root: Path) -> None:
        fs.write_file('hello.txt', 'overwritten')
        assert (tmp_root / 'hello.txt').read_text() == 'overwritten'


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_edit_single(self, fs: FileSystem, tmp_root: Path) -> None:
        result = fs.edit_file('hello.txt', 'line two', 'LINE TWO')
        assert '1 occurrence' in result
        assert 'LINE TWO' in (tmp_root / 'hello.txt').read_text()

    def test_edit_not_found(self, fs: FileSystem) -> None:
        with pytest.raises(ValueError, match='not found'):
            fs.edit_file('hello.txt', 'does not exist', 'replacement')

    def test_edit_ambiguous(self, fs: FileSystem, tmp_root: Path) -> None:
        (tmp_root / 'dup.txt').write_text('aaa\naaa\n')
        with pytest.raises(ValueError, match='2 times'):
            fs.edit_file('dup.txt', 'aaa', 'bbb')

    def test_edit_replace_all(self, fs: FileSystem, tmp_root: Path) -> None:
        (tmp_root / 'dup.txt').write_text('aaa\naaa\n')
        result = fs.edit_file('dup.txt', 'aaa', 'bbb', replace_all=True)
        assert '2 occurrence' in result
        assert (tmp_root / 'dup.txt').read_text() == 'bbb\nbbb\n'

    def test_edit_missing_file(self, fs: FileSystem) -> None:
        with pytest.raises(FileNotFoundError):
            fs.edit_file('nope.txt', 'a', 'b')


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    def test_list_root(self, fs: FileSystem) -> None:
        result = fs.list_directory()
        assert 'hello.txt' in result
        assert 'sub/' in result

    def test_list_subdir(self, fs: FileSystem) -> None:
        result = fs.list_directory('sub')
        assert 'nested.py' in result

    def test_list_nonexistent(self, fs: FileSystem) -> None:
        with pytest.raises(NotADirectoryError):
            fs.list_directory('nonexistent')

    def test_list_empty(self, fs: FileSystem, tmp_root: Path) -> None:
        (tmp_root / 'empty').mkdir()
        result = fs.list_directory('empty')
        assert result == '(empty directory)'


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


class TestSearchFiles:
    def test_search_match(self, fs: FileSystem) -> None:
        result = fs.search_files('line')
        assert 'hello.txt:1:line one' in result

    def test_search_regex(self, fs: FileSystem) -> None:
        result = fs.search_files(r'line\s+t')
        assert 'hello.txt' in result

    def test_search_no_match(self, fs: FileSystem) -> None:
        result = fs.search_files('zzzzz_nothing')
        assert result == 'No matches found.'

    def test_search_nested(self, fs: FileSystem) -> None:
        result = fs.search_files('print')
        assert 'sub/nested.py' in result

    def test_search_specific_file(self, fs: FileSystem) -> None:
        result = fs.search_files('line', path='hello.txt')
        assert 'hello.txt:1:line one' in result

    def test_search_skips_hidden(self, fs: FileSystem, tmp_root: Path) -> None:
        (tmp_root / '.hidden').mkdir()
        (tmp_root / '.hidden' / 'secret.txt').write_text('findme\n')
        result = fs.search_files('findme')
        assert result == 'No matches found.'

    def test_search_skips_binary(self, fs: FileSystem, tmp_root: Path) -> None:
        (tmp_root / 'binary.dat').write_bytes(b'findme\x00binary')
        result = fs.search_files('findme')
        assert 'binary.dat' not in result

    def test_search_skips_unreadable(self, fs: FileSystem, tmp_root: Path) -> None:
        target = tmp_root / 'unreadable.txt'
        target.write_text('findme\n')
        target.chmod(0o000)
        try:
            result = fs.search_files('findme')
            assert 'unreadable.txt' not in result
        finally:
            target.chmod(0o644)

    def test_search_truncation(self, tmp_root: Path) -> None:
        # Create enough matches to trigger truncation at 1000
        big = '\n'.join(f'match line {i}' for i in range(1100))
        (tmp_root / 'big.txt').write_text(big)
        fs = FileSystem(root_dir=tmp_root)
        result = fs.search_files('match')
        assert 'truncated' in result


# ---------------------------------------------------------------------------
# Path filtering (allowed_patterns / denied_patterns)
# ---------------------------------------------------------------------------


class TestPathFiltering:
    def test_denied_pattern(self, tmp_root: Path) -> None:
        fs = FileSystem(root_dir=tmp_root, denied_patterns=['*.env'])
        with pytest.raises(PermissionError, match='denied'):
            fs.read_file('secret.env')
        # Other files still accessible
        result = fs.read_file('hello.txt')
        assert 'line one' in result

    def test_allowed_pattern(self, tmp_root: Path) -> None:
        fs = FileSystem(root_dir=tmp_root, allowed_patterns=['*.txt'])
        result = fs.read_file('hello.txt')
        assert 'line one' in result
        with pytest.raises(PermissionError, match='does not match'):
            fs.read_file('secret.env')

    def test_denied_write(self, tmp_root: Path) -> None:
        fs = FileSystem(root_dir=tmp_root, denied_patterns=['*.env'])
        with pytest.raises(PermissionError, match='denied'):
            fs.write_file('new.env', 'bad')

    def test_denied_edit(self, tmp_root: Path) -> None:
        fs = FileSystem(root_dir=tmp_root, denied_patterns=['*.env'])
        with pytest.raises(PermissionError, match='denied'):
            fs.edit_file('secret.env', 'API_KEY', 'REDACTED')


# ---------------------------------------------------------------------------
# Toolset integration
# ---------------------------------------------------------------------------


class TestToolset:
    def test_get_toolset_returns_function_toolset(self, fs: FileSystem) -> None:
        from pydantic_ai.toolsets import FunctionToolset

        toolset = fs.get_toolset()
        assert isinstance(toolset, FunctionToolset)

    def test_toolset_has_expected_tools(self, fs: FileSystem) -> None:
        from pydantic_ai.toolsets import FunctionToolset

        toolset = fs.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        tool_names = set(toolset.tools.keys())
        assert tool_names == {'read_file', 'write_file', 'edit_file', 'list_directory', 'search_files'}

    def test_serialization_name(self) -> None:
        assert FileSystem.get_serialization_name() == 'FileSystem'
