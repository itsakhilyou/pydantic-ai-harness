"""Backend conformance suite: the contract every `AbstractEnvironment` must satisfy.

Each test takes the parametrized `environment` fixture and runs against every backend.
Files are seeded on the host `tmp_path` and exercised through `environment`; the two
point at the same directory (see `conftest.py`). Backend-specific behavior (symlink
resolution, POSIX permissions) lives in the per-backend test modules, not here.
"""

from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai_harness.environments import AbstractMatch
from pydantic_ai_harness.environments.abstract import AbstractEnvironment, ShellCommandResult
from pydantic_ai_harness.environments.exceptions import (
    EnvInvalidPatternError,
    EnvIsADirectoryError,
    EnvNotADirectoryError,
    EnvNotFoundError,
    EnvWriteError,
    PathEscapeError,
)


async def test_write_then_read_round_trips(environment: AbstractEnvironment) -> None:
    await environment.write_file('note.txt', b'hello')
    assert await environment.read_file('note.txt') == b'hello'


async def test_read_returns_raw_bytes(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'data.bin').write_bytes(b'\x00\xff\xfe')
    assert await environment.read_file('data.bin') == b'\x00\xff\xfe'


async def test_write_creates_missing_file(environment: AbstractEnvironment, tmp_path: Path) -> None:
    await environment.write_file('fresh.txt', b'new')
    assert (tmp_path / 'fresh.txt').read_bytes() == b'new'


async def test_read_missing_file_raises_not_found(environment: AbstractEnvironment) -> None:
    with pytest.raises(EnvNotFoundError):
        await environment.read_file('does-not-exist.txt')


async def test_read_directory_raises_is_a_directory(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'subdir').mkdir()
    with pytest.raises(EnvIsADirectoryError):
        await environment.read_file('subdir')


async def test_read_through_file_component_raises_not_a_directory(
    environment: AbstractEnvironment, tmp_path: Path
) -> None:
    # A path that treats a regular file as if it were a directory.
    (tmp_path / 'file.txt').write_bytes(b'x')
    with pytest.raises(EnvNotADirectoryError):
        await environment.read_file('file.txt/inner')


async def test_write_onto_directory_raises_write_error(environment: AbstractEnvironment, tmp_path: Path) -> None:
    # Writing bytes where a directory already exists is an I/O failure, not a model-fixable
    # path problem -> the generic write error, which the capability layer propagates.
    (tmp_path / 'adir').mkdir()
    with pytest.raises(EnvWriteError):
        await environment.write_file('adir', b'nope')


async def test_relative_escape_read_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.read_file('../escape.txt')


async def test_relative_escape_write_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.write_file('../escape.txt', b'nope')


async def test_ls_lists_entries_with_types(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'a.txt').write_bytes(b'x')
    (tmp_path / 'subdir').mkdir()
    listing = await environment.ls('.')
    assert {(f.name, f.is_directory) for f in listing} == {('a.txt', False), ('subdir', True)}


async def test_ls_missing_directory_raises_not_found(environment: AbstractEnvironment) -> None:
    with pytest.raises(EnvNotFoundError):
        await environment.ls('does-not-exist')


async def test_ls_on_a_file_raises_not_a_directory(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'file.txt').write_bytes(b'x')
    with pytest.raises(EnvNotADirectoryError):
        await environment.ls('file.txt')


async def test_ls_relative_escape_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.ls('..')


async def test_grep_finds_matches_recursively(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'top.txt').write_text('hello\nNEEDLE here\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'deep.txt').write_text('nothing\nalso NEEDLE\n')
    (tmp_path / 'sub' / 'miss.txt').write_text('no match here\n')

    matches = await environment.grep('.', 'NEEDLE')

    # The backend returns matches in filesystem walk order (unsorted) -- determinism is added at
    # the capability layer, not here -- so compare as a set, like the `ls` conformance test.
    assert {(m.path, m.lineno, m.line) for m in matches} == {
        ('top.txt', 2, 'NEEDLE here\n'),
        ('sub/deep.txt', 2, 'also NEEDLE\n'),
    }


async def test_grep_missing_file_raises_not_found(environment: AbstractEnvironment) -> None:
    with pytest.raises(EnvNotFoundError):
        await environment.grep('does-not-exist', 'NEEDLE')


async def test_grep_relative_escape_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.grep('..', 'NEEDLE')


async def test_grep_binary_file_skips_and_continues(environment: AbstractEnvironment, tmp_path: Path) -> None:
    # Create a directory with a binary file in it
    (tmp_path / 'dir').mkdir()
    (tmp_path / 'dir' / 'binary.bin').write_bytes(b'\x00\xff\xfe')

    # Create a file with a match
    (tmp_path / 'dir' / 'match.txt').write_text('NEEDLE')

    # Create a file with a non-match
    (tmp_path / 'dir' / 'non-match.txt').write_text('no match here')

    # Grep the directory

    matches = await environment.grep('dir', 'NEEDLE')
    assert matches == snapshot([AbstractMatch(path='dir/match.txt', line='NEEDLE', lineno=1)])


async def test_grep_single_file(environment: AbstractEnvironment, tmp_path: Path) -> None:
    # A file path (not a directory) exercises the is-a-file branch: search just that file.
    (tmp_path / 'only.txt').write_text('first\nNEEDLE on two\nthird\n')
    matches = await environment.grep('only.txt', 'NEEDLE')
    assert matches == snapshot([AbstractMatch(path='only.txt', line='NEEDLE on two\n', lineno=2)])


# --- grep regex subset: backend-portable dialect (every backend must agree) ---
#
# The contract on `AbstractEnvironment.grep` commits to a subset of regex that we promise
# every backend will match identically. These tests pin that subset; when DockerEnvironment
# (or any new backend) lands, the same matrix runs against it and divergence becomes a red
# CI signal, not a user-reported drift. Features outside this list are undefined-by-design.


@pytest.fixture
def grep_corpus(tmp_path: Path) -> Path:
    """Tree with one line of each shape the subset covers, so every test below uses the same input."""
    (tmp_path / 'src.py').write_text(
        'def main():\n'
        'class Widget:\n'
        'from typing import Any\n'
        'import os\n'
        'TODO: refactor\n'
        '# FIXME bug here\n'
        'value = 42\n'
        'name = "alpha"\n'
        'literal_dot.here\n'
    )
    return tmp_path


async def test_grep_subset_literal_match(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """Literal text -- the simplest regex; carries over from the old substring contract."""
    matches = await environment.grep('.', 'TODO')
    assert {m.lineno for m in matches} == {5}


async def test_grep_subset_word_class(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """`\\w+` -- the bread-and-butter "identifier" pattern models reach for first."""
    matches = await environment.grep('.', r'def \w+')
    assert {m.lineno for m in matches} == {1}


async def test_grep_subset_digit_class(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """`\\d+` -- the matching numeric class."""
    matches = await environment.grep('.', r'\d+')
    assert {m.lineno for m in matches} == {7}


async def test_grep_subset_anchor(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """`^` -- start-of-line anchor; lets the model say "imports at column 0," not anywhere."""
    matches = await environment.grep('.', r'^from \w+')
    assert {m.lineno for m in matches} == {3}


async def test_grep_subset_alternation(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """`|` -- "either of these"; common for TODO|FIXME-style sweeps."""
    matches = await environment.grep('.', r'TODO|FIXME')
    assert {m.lineno for m in matches} == {5, 6}


async def test_grep_subset_char_class(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """`[…]` -- explicit character class; the only way to express "any of these chars" in regex."""
    matches = await environment.grep('.', r'[A-Z]+')
    assert {m.lineno for m in matches} == {2, 3, 5, 6}  # Widget, Any, TODO, FIXME


async def test_grep_subset_escape_metacharacter(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """`\\.` -- escape; the model must be able to search for literal `.` without it being any-char."""
    matches = await environment.grep('.', r'literal_dot\.here')
    assert {m.lineno for m in matches} == {9}


async def test_grep_subset_group_and_quantifier(environment: AbstractEnvironment, grep_corpus: Path) -> None:
    """`(?:…)+` -- non-capturing group with a quantifier; needed for "one or more of <thing>"."""
    matches = await environment.grep('.', r'(?:import|from) \w+')
    assert {m.lineno for m in matches} == {3, 4}


async def test_grep_invalid_pattern_raises(environment: AbstractEnvironment, tmp_path: Path) -> None:
    """Malformed regex (here, an unmatched `[`) must raise EnvInvalidPatternError so the capability
    layer can route to ModelRetry -- the model gets a chance to fix it rather than silently see []."""
    (tmp_path / 'src.py').write_text('hello\n')
    with pytest.raises(EnvInvalidPatternError):
        await environment.grep('.', r'[unterminated')


async def test_grep_flag_looking_pattern_is_treated_as_data(environment: AbstractEnvironment, tmp_path: Path) -> None:
    """A model-supplied pattern that LOOKS like a CLI flag must NOT be interpreted as one.

    This is the load-bearing read-only safety test: if a backend's grep implementation passed
    the model's pattern positionally to a regex tool like ripgrep, the model could write
    `pattern='--pre=/bin/sh'` and trick the tool into executing an arbitrary preprocessor
    binary on every file -- effectively shell access through a "search" tool, in a sandbox
    that explicitly denied shell. Every backend must therefore pass the pattern as DATA, not
    as an argv element that could be re-parsed as a flag. Concretely on rg: use the form
    `--regexp=<pat>` (value bound in one argv element) and never `<pat>` positionally.

    We assert the pattern simply doesn't match anything -- it's regex, treated as the literal
    string `--pre=/bin/sh`, which doesn't appear in any file in the tree.
    """
    (tmp_path / 'src.py').write_text('def main():\n    pass\n')
    matches = await environment.grep('.', '--pre=/bin/sh')
    assert matches == []


async def test_glob_missing_directory_raises_not_found(environment: AbstractEnvironment) -> None:
    with pytest.raises(EnvNotFoundError):
        await environment.glob('does-not-exist', '*.py')


async def test_glob_on_a_file_raises_not_a_directory(environment: AbstractEnvironment, tmp_path: Path) -> None:
    # glob's `path` is the directory to search WITHIN; pointing it at a file is a model
    # argument error, surfaced as EnvNotADirectoryError -> ModelRetry (mirrors ls).
    (tmp_path / 'file.txt').write_bytes(b'x')
    with pytest.raises(EnvNotADirectoryError):
        await environment.glob('file.txt', '*.py')


async def test_glob_relative_escape_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.glob('..', '*.py')


async def test_glob_matches_recursively(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'top').mkdir()
    (tmp_path / 'top' / 'sub').mkdir()
    (tmp_path / 'top' / 'sub' / 'deep.py').write_text('NEEDLE')
    (tmp_path / 'top' / 'sub' / 'notes.txt').write_text('NO MATCH')

    matches = await environment.glob('.', '*.py')
    assert matches == snapshot(['top/sub/deep.py'])


async def test_glob_excludes_directories(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'inner.txt').write_text('x')
    assert await environment.glob('.', 'sub') == []


async def test_shell_captures_stdout(environment: AbstractEnvironment, tmp_path: Path) -> None:
    result = await environment.shell_command('echo "hello"')
    assert result == snapshot(ShellCommandResult(stdout=b'hello\n', stderr=b'', return_code=0, timed_out=False))


async def test_shell_non_zero_exit_is_not_an_error(environment: AbstractEnvironment, tmp_path: Path) -> None:
    result = await environment.shell_command('echo "hello" && exit 1')
    assert result == snapshot(ShellCommandResult(stdout=b'hello\n', stderr=b'', return_code=1, timed_out=False))


async def test_shell_captures_stderr_separately(environment: AbstractEnvironment, tmp_path: Path) -> None:
    result = await environment.shell_command('echo "hello" >&2')
    assert result == snapshot(ShellCommandResult(stdout=b'', stderr=b'hello\n', return_code=0, timed_out=False))


async def test_shell_is_shell_interpreted(environment: AbstractEnvironment, tmp_path: Path) -> None:
    result = await environment.shell_command('echo a && echo b')
    assert result == snapshot(ShellCommandResult(stdout=b'a\nb\n', stderr=b'', return_code=0, timed_out=False))


async def test_shell_runs_in_root(environment: AbstractEnvironment, tmp_path: Path) -> None:
    result = await environment.shell_command('pwd', timeout=1)
    assert result.stdout == f'{tmp_path.resolve()}\n'.encode()


async def test_shell_no_state_persists_between_calls(environment: AbstractEnvironment, tmp_path: Path) -> None:
    result = await environment.shell_command('export FOO=bar', timeout=1)
    assert result == snapshot(ShellCommandResult(stdout=b'', stderr=b'', return_code=0, timed_out=False))
    result = await environment.shell_command('echo $FOO', timeout=1)
    assert result == snapshot(ShellCommandResult(stdout=b'\n', stderr=b'', return_code=0, timed_out=False))


async def test_shell_timeout_sets_flag(environment: AbstractEnvironment, tmp_path: Path) -> None:
    result = await environment.shell_command('sleep 10', timeout=1)
    assert result == snapshot(ShellCommandResult(stdout=b'', stderr=b'', return_code=-15, timed_out=True))
