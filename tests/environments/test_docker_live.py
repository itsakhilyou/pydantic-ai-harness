"""Live `DockerEnvironment.shell_command` tests against a real Docker daemon.

These run end-to-end against `python:3.12-slim`. They skip automatically when no Docker
daemon is reachable so contributors without Docker installed still get a green suite; the
mocked lifecycle tests in `test_docker.py` cover the daemon-free decision tree.
"""

import asyncio
from collections.abc import AsyncIterator

import docker
import docker.errors
import pytest

from pydantic_ai_harness.environments.docker import DockerEnvironment
from pydantic_ai_harness.environments.exceptions import (
    EnvInvalidPatternError,
    EnvIsADirectoryError,
    EnvNotFoundError,
    EnvShellExecutionError,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _docker_available() -> bool:
    """`True` iff a Docker daemon is reachable. Used to skip the whole module otherwise."""
    try:
        docker.from_env().ping()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except (docker.errors.DockerException, OSError):
        return False
    return True


pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not _docker_available(), reason='no Docker daemon')]


@pytest.fixture
async def env() -> AsyncIterator[DockerEnvironment]:
    """Start `python:3.12-slim` for one test and remove it on teardown."""
    environment = DockerEnvironment(image='python:3.12-slim')
    async with environment:
        yield environment


async def test_write_then_read_round_trips_bytes(env: DockerEnvironment) -> None:
    """write_file then read_file recovers the exact bytes -- binary-safe."""
    data = bytes(range(256))  # every byte value, including NULs and high-bit
    await env.write_file('round-trip.bin', data)
    assert await env.read_file('round-trip.bin') == data


async def test_write_file_creates_parent_dirs(env: DockerEnvironment) -> None:
    """ABC: missing intermediate directories are created."""
    await env.write_file('deep/nested/path/file.txt', b'hello')
    assert await env.read_file('deep/nested/path/file.txt') == b'hello'


async def test_read_file_not_found(env: DockerEnvironment) -> None:
    with pytest.raises(EnvNotFoundError):
        await env.read_file('does-not-exist.txt')


async def test_read_file_on_directory_raises_is_a_directory(env: DockerEnvironment) -> None:
    await env.shell_command('mkdir -p /workspace/somedir')
    with pytest.raises(EnvIsADirectoryError):
        await env.read_file('somedir')


async def test_grep_raises_when_rg_missing(env: DockerEnvironment) -> None:
    """python:3.12-slim does not ship ripgrep; grep must raise a clear error pointing the
    user at the missing dependency, not silently fail."""
    with pytest.raises(EnvShellExecutionError, match='ripgrep'):
        await env.grep('.', 'anything')


@pytest.fixture
async def env_with_rg() -> AsyncIterator[DockerEnvironment]:
    """Same as `env` but with `rg` installed -- the cost of `apt-get install` is paid once
    per test; vendoring rg into the container is tracked separately so this is the interim path."""
    environment = DockerEnvironment(image='python:3.12-slim')
    async with environment:
        rg_install = await environment.shell_command('apt-get update -qq && apt-get install -y -qq ripgrep')
        assert rg_install.return_code == 0, f'apt-get install ripgrep failed: {rg_install.stderr!r}'
        yield environment


async def test_grep_finds_matches_and_returns_relative_paths(env_with_rg: DockerEnvironment) -> None:
    await env_with_rg.write_file('a.py', b"x = 'needle'\ny = 1\n")
    await env_with_rg.write_file('sub/b.py', b"z = 'needle'\n")
    matches = await env_with_rg.grep('.', 'needle')
    by_path = {m.path: m for m in matches}
    assert set(by_path) == {'a.py', 'sub/b.py'}
    assert by_path['a.py'].lineno == 1
    assert by_path['sub/b.py'].lineno == 1


async def test_grep_invalid_regex_raises_invalid_pattern(env_with_rg: DockerEnvironment) -> None:
    with pytest.raises(EnvInvalidPatternError):
        await env_with_rg.grep('.', '[unclosed')


async def test_glob_returns_matching_files_at_any_depth(env_with_rg: DockerEnvironment) -> None:
    await env_with_rg.write_file('a.py', b'')
    await env_with_rg.write_file('sub/b.py', b'')
    await env_with_rg.write_file('sub/c.txt', b'')
    result = sorted(await env_with_rg.glob('.', '*.py'))
    assert result == ['a.py', 'sub/b.py']


async def test_ls_returns_entries_with_type_info(env: DockerEnvironment) -> None:
    """`ls -1AF` parsing: dotfiles included, dirs flagged, symlink-to-dir classified as itself."""
    await env.write_file('visible.txt', b'')
    await env.write_file('.hidden', b'')
    await env.shell_command('mkdir -p /workspace/subdir && ln -sfn /workspace/subdir /workspace/link')
    entries = {f.name: f.is_directory for f in await env.ls('.')}
    assert entries.get('visible.txt') is False
    assert entries.get('.hidden') is False
    assert entries.get('subdir') is True
    # Symlink to a directory classifies as the symlink itself (`is_directory=False`), matching
    # LocalEnvironment's `is_dir(follow_symlinks=False)` semantics.
    assert entries.get('link') is False


async def test_echo_returns_stdout_and_zero_exit(env: DockerEnvironment) -> None:
    """Happy path: `echo hello` -> stdout=b'hello\\n', exit 0, timed_out False."""
    result = await env.shell_command('echo hello')
    assert result.stdout == b'hello\n'
    assert result.stderr == b''
    assert result.return_code == 0
    assert result.timed_out is False


async def test_non_zero_exit_is_returned_not_raised(env: DockerEnvironment) -> None:
    """The ABC requires non-zero exits to come back as a result, not an exception."""
    result = await env.shell_command('exit 7')
    assert result.return_code == 7
    assert result.timed_out is False


async def test_stderr_is_captured_separately(env: DockerEnvironment) -> None:
    """`demux=True` keeps stdout and stderr on separate channels."""
    result = await env.shell_command('echo out; echo err 1>&2')
    assert result.stdout == b'out\n'
    assert result.stderr == b'err\n'
    assert result.return_code == 0


async def test_timeout_kills_process_and_marks_timed_out(env: DockerEnvironment) -> None:
    """`sleep 5` with timeout=0.5 must return promptly with `timed_out=True`, not raise."""
    result = await env.shell_command('sleep 5', timeout=0.5)
    assert result.timed_out is True
    # SIGKILL via the kill path lands as exit 137 (128 + 9) in Docker's accounting.
    assert result.return_code == 137


async def test_cancel_kills_process_and_reraises(env: DockerEnvironment) -> None:
    """Cancelling the awaiting task must SIGKILL the in-container process and re-raise.

    Verifying the kill happened: a follow-up cheap command must return immediately rather
    than be blocked by the prior `sleep` still holding daemon resources.
    """
    task = asyncio.create_task(env.shell_command('sleep 5'))
    await asyncio.sleep(0.2)  # let exec_start spawn and exec_inspect populate the PID
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Container is still healthy; another command runs cleanly.
    result = await env.shell_command('echo alive')
    assert result.stdout == b'alive\n'
    assert result.return_code == 0
