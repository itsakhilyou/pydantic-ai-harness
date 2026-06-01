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
