"""Tests for the optional ripgrep backend of the FileSystem capability.

These exercise both the detection/resolution logic (which runs everywhere, via
monkeypatching) and the ripgrep search path itself (which is skipped when the
`ripgrep` extra or the `rg` binary is unavailable). The pure-Python fallback is
covered both here (via `use_ripgrep=False`) and by `test_filesystem.py`.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.filesystem import _ripgrep as rg
from pydantic_ai_harness.filesystem._toolset import FileSystemToolset

requires_ripgrep = pytest.mark.skipif(
    not rg.ripgrep_available(),
    reason="ripgrep backend unavailable (install the 'ripgrep' extra and an 'rg' binary)",
)


def _toolset(
    root: Path,
    *,
    use_ripgrep: bool,
    allowed_patterns: Sequence[str] = (),
    denied_patterns: Sequence[str] = (),
    max_search_results: int = 1000,
) -> FileSystemToolset[None]:
    return FileSystemToolset(
        root_dir=root,
        allowed_patterns=allowed_patterns,
        denied_patterns=denied_patterns,
        protected_patterns=(),
        max_read_lines=2000,
        max_search_results=max_search_results,
        max_find_results=1000,
        use_ripgrep=use_ripgrep,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small tree shared by the backend-parity tests."""
    (tmp_path / 'hello.txt').write_text('Hello, world!\nfindme here\n')
    (tmp_path / 'multi.txt').write_text('line1\nline2\nline3\nfindme\n')
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'mod.py').write_text('def go():\n    findme = 1\n    return findme\n')
    (tmp_path / 'pkg' / 'notes.md').write_text('# notes\nfindme in markdown\n')
    (tmp_path / '.hidden').write_text('findme secret\n')
    (tmp_path / 'binary.bin').write_bytes(b'findme\x00\x01\x02')
    return tmp_path


def _set_available(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    def fake_available() -> bool:
        return available

    monkeypatch.setattr(rg, 'ripgrep_available', fake_available)


class TestDetection:
    def test_available_requires_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_binary(cmd: str) -> str | None:
            return None

        monkeypatch.setattr(shutil, 'which', no_binary)
        assert rg.ripgrep_available() is False

    def test_available_when_binary_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def has_binary(cmd: str) -> str | None:
            return '/usr/bin/rg'

        monkeypatch.setattr(shutil, 'which', has_binary)
        assert rg.ripgrep_available() is True

    def test_resolve_false_disables(self) -> None:
        assert rg.resolve_ripgrep_enabled(False) is False

    def test_resolve_none_follows_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_available(monkeypatch, True)
        assert rg.resolve_ripgrep_enabled(None) is True
        _set_available(monkeypatch, False)
        assert rg.resolve_ripgrep_enabled(None) is False

    def test_resolve_true_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_available(monkeypatch, True)
        assert rg.resolve_ripgrep_enabled(True) is True

    def test_resolve_true_unavailable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin both halves of the message: the problem and the actionable fix.
        _set_available(monkeypatch, False)
        with pytest.raises(RuntimeError, match=r"no 'rg' binary.+Install the 'ripgrep' extra"):
            rg.resolve_ripgrep_enabled(True)


class TestCapabilityWiring:
    def test_default_is_none(self) -> None:
        assert FileSystem[None]().use_ripgrep is None

    def test_forwards_to_toolset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The capability must forward use_ripgrep to the toolset: forcing the
        # backend on while it is unavailable surfaces through get_toolset().
        _set_available(monkeypatch, False)
        with pytest.raises(RuntimeError, match="no 'rg' binary"):
            FileSystem[None](root_dir=tmp_path, use_ripgrep=True).get_toolset()


class _FakeProcess:
    def __init__(self, returncode: int, stdout: bytes = b'', stderr: bytes = b'') -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestRipgrepProtocol:
    """Pin the `rg` invocation and JSON/exit-code handling directly.

    The behavioral tests run real ripgrep, but the toolset falls back to pure
    Python whenever ripgrep fails, which would mask a broken `rg` command or
    parser. Faking the subprocess is the pragmatic boundary to assert the
    protocol the fallback otherwise hides.
    """

    async def test_builds_expected_command(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        async def fake(command: list[str], *, cwd: Path, check: bool) -> _FakeProcess:
            captured.update(command=command, cwd=cwd, check=check)
            return _FakeProcess(1)

        monkeypatch.setattr(rg, 'run_process', fake)
        result = await rg.ripgrep_file_matches(root=tmp_path, target=Path('sub'), pattern='foo')
        assert result == []
        assert captured['command'] == ['rg', '--json', '--no-ignore', '--sort=path', '-e', 'foo', '--', 'sub']
        assert captured['cwd'] == tmp_path
        assert captured['check'] is False

    async def test_parses_match_events(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        stdout = (
            b'{"type":"begin","data":{"path":{"text":"a.txt"}}}\n'
            b'{"type":"match","data":{"path":{"text":"a.txt"},"line_number":2}}\n'
            b'{"type":"match","data":{"path":{"text":"a.txt"},"line_number":5}}\n'
            b'{"type":"end","data":{"path":{"text":"a.txt"}}}\n'
            b'{"type":"summary","data":{}}\n'
        )

        async def fake(command: list[str], *, cwd: Path, check: bool) -> _FakeProcess:
            return _FakeProcess(0, stdout=stdout)

        monkeypatch.setattr(rg, 'run_process', fake)
        result = await rg.ripgrep_file_matches(root=tmp_path, target=Path('.'), pattern='x')
        assert result == [(Path('a.txt'), [2, 5])]

    async def test_raises_on_failure_with_stderr(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Invalid bytes also exercise the stderr decode's error handling.
        async def fake(command: list[str], *, cwd: Path, check: bool) -> _FakeProcess:
            return _FakeProcess(2, stderr=b'\xff regex parse error: boom')

        monkeypatch.setattr(rg, 'run_process', fake)
        with pytest.raises(rg.RipgrepError, match='regex parse error: boom'):
            await rg.ripgrep_file_matches(root=tmp_path, target=Path('.'), pattern='(')

    async def test_line_past_eof_is_dropped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # A file shrinking between ripgrep's scan and the re-read must not crash.
        (tmp_path / 'f.txt').write_text('only one line\n')
        _set_available(monkeypatch, True)

        async def fake_matches(*, root: Path, target: Path, pattern: str) -> list[tuple[Path, list[int]]]:
            return [(Path('f.txt'), [1, 99])]

        monkeypatch.setattr('pydantic_ai_harness.filesystem._toolset.ripgrep_file_matches', fake_matches)
        result = await _toolset(tmp_path, use_ripgrep=True).search_files('one')
        assert result == 'f.txt:1:only one line'


@requires_ripgrep
class TestRipgrepBackend:
    """ripgrep-specific behavior; equivalence to the fallback is in TestBackendParity."""

    async def test_output_format(self, tree: Path) -> None:
        result = await _toolset(tree, use_ripgrep=True).search_files('findme', path='multi.txt')
        assert result == 'multi.txt:4:findme'

    async def test_access_patterns_filter_results(self, tree: Path) -> None:
        result = await _toolset(tree, use_ripgrep=True, denied_patterns=['*.md']).search_files('findme')
        assert 'mod.py' in result
        assert 'notes.md' not in result

    async def test_truncation(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f'm{i}.txt').write_text('findme\n' * 50)
        result = await _toolset(tmp_path, use_ripgrep=True, max_search_results=30).search_files('findme')
        assert 'truncated at 30 matches' in result

    async def test_unsupported_pattern_falls_back(self, tree: Path) -> None:
        # A lookahead is valid for Python `re` but rejected by ripgrep's default
        # engine, so the toolset must fall back instead of failing or differing.
        rg_result = await _toolset(tree, use_ripgrep=True).search_files('(?=findme)')
        py_result = await _toolset(tree, use_ripgrep=False).search_files('(?=findme)')
        assert rg_result == py_result
        assert 'findme' in rg_result


@requires_ripgrep
class TestBackendParity:
    """The ripgrep path must produce the same output as the pure-Python path."""

    @pytest.mark.parametrize('pattern', ['findme', r'line\d', 'def ', 'world'])
    async def test_same_output(self, tree: Path, pattern: str) -> None:
        rg_result = await _toolset(tree, use_ripgrep=True).search_files(pattern)
        py_result = await _toolset(tree, use_ripgrep=False).search_files(pattern)
        assert rg_result == py_result

    async def test_same_output_with_include_glob(self, tree: Path) -> None:
        rg_result = await _toolset(tree, use_ripgrep=True).search_files('findme', include_glob='*.py')
        py_result = await _toolset(tree, use_ripgrep=False).search_files('findme', include_glob='*.py')
        assert rg_result == py_result

    async def test_same_output_in_subdir(self, tree: Path) -> None:
        rg_result = await _toolset(tree, use_ripgrep=True).search_files('findme', path='pkg')
        py_result = await _toolset(tree, use_ripgrep=False).search_files('findme', path='pkg')
        assert rg_result == py_result

    async def test_same_output_with_crlf(self, tmp_path: Path) -> None:
        (tmp_path / 'win.txt').write_bytes(b'alpha\r\nfindme\r\nbeta\r\n')
        rg_result = await _toolset(tmp_path, use_ripgrep=True).search_files('findme')
        py_result = await _toolset(tmp_path, use_ripgrep=False).search_files('findme')
        assert rg_result == py_result == 'win.txt:2:findme'

    async def test_binary_detection_differs_past_8kb(self, tmp_path: Path) -> None:
        # Documented limitation: the pure-Python walker samples the first 8 KB for
        # a NUL byte while ripgrep scans the whole file, so they differ on a file
        # whose only NUL byte is past 8 KB.
        (tmp_path / 'big.txt').write_bytes((b'findme\n' * 1300) + b'\x00' + b'findme\n')
        rg_result = await _toolset(tmp_path, use_ripgrep=True).search_files('findme')
        py_result = await _toolset(tmp_path, use_ripgrep=False).search_files('findme')
        assert rg_result == 'No matches found.'  # ripgrep treats it as binary
        assert 'big.txt:1:findme' in py_result  # pure Python searches it


class TestFallbackPath:
    """Force the pure-Python backend so it is covered even where ripgrep exists."""

    async def test_fallback_basic(self, tree: Path) -> None:
        result = await _toolset(tree, use_ripgrep=False).search_files('findme')
        assert 'hello.txt:2:findme here' in result

    async def test_fallback_skips_hidden_and_binary(self, tree: Path) -> None:
        result = await _toolset(tree, use_ripgrep=False).search_files('findme')
        assert '.hidden' not in result
        assert 'binary.bin' not in result

    async def test_fallback_single_file(self, tree: Path) -> None:
        result = await _toolset(tree, use_ripgrep=False).search_files('findme', path='multi.txt')
        assert result == 'multi.txt:4:findme'

    async def test_fallback_truncation(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f'm{i}.txt').write_text('findme\n' * 50)
        result = await _toolset(tmp_path, use_ripgrep=False, max_search_results=30).search_files('findme')
        assert 'truncated at 30 matches' in result

    async def test_fallback_no_matches_message(self, tree: Path) -> None:
        assert await _toolset(tree, use_ripgrep=False).search_files('ZZZNOPE') == 'No matches found.'

    async def test_fallback_truncates_at_exactly_the_limit(self, tmp_path: Path) -> None:
        # Truncation triggers when results reach the limit, not only past it.
        (tmp_path / 'f.txt').write_text('findme\n' * 5)
        result = await _toolset(tmp_path, use_ripgrep=False, max_search_results=5).search_files('findme')
        assert 'truncated at 5 matches' in result


@pytest.mark.parametrize('use_ripgrep', [False, pytest.param(True, marks=requires_ripgrep)])
class TestWalkDoesNotStopOnSkip:
    """A skipped entry sorting before a wanted match must not halt the walk."""

    async def test_skipped_entry_before_match(self, tmp_path: Path, use_ripgrep: bool) -> None:
        # These all sort before 'zzz.txt': a directory (not a file), a binary
        # file, and a denied file. None may turn the per-entry skip into a stop.
        (tmp_path / 'aaa_dir').mkdir()
        (tmp_path / 'aab.bin').write_bytes(b'findme\x00\x01')
        (tmp_path / 'aac_skip.txt').write_text('findme\n')
        (tmp_path / 'zzz.txt').write_text('findme\n')
        ts = _toolset(tmp_path, use_ripgrep=use_ripgrep, denied_patterns=['aac_skip.txt'])
        result = await ts.search_files('findme')
        assert 'zzz.txt' in result
        assert 'aac_skip.txt' not in result
        assert 'aab.bin' not in result
