"""DockerEnvironment lifecycle tests with the `docker` SDK faked out.

Covers the failure modes called out in `agent_docs/environment-lifecycle.md`
"Backend implementer's guide": mode validation, attach-mode setup/teardown as no-ops,
owned-mode setup/teardown success and failure paths, and the "already gone" idempotent
case. Runs without a real Docker daemon by monkeypatching `docker.from_env` to return
a fake client.

Real-daemon integration tests will land alongside the file I/O methods; the unit
coverage here is what proves the lifecycle decision tree, not the daemon plumbing.
"""

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from pydantic_ai_harness.environments._ripgrep import RG_EXIT_USAGE_OR_PATTERN
from pydantic_ai_harness.environments.exceptions import (
    EnvInvalidPatternError,
    EnvPermissionError,
    EnvReadError,
    EnvSetupError,
    EnvShellExecutionError,
    EnvWriteError,
)

try:
    import docker.errors

    from pydantic_ai_harness.environments import docker as docker_module
    from pydantic_ai_harness.environments.docker import (
        _EXIT_COMMAND_NOT_FOUND,  # pyright: ignore[reportPrivateUsage]
        _EXIT_IS_DIRECTORY,  # pyright: ignore[reportPrivateUsage]
        _EXIT_PERMISSION_DENIED,  # pyright: ignore[reportPrivateUsage]
        DockerEnvironment,
    )
except ImportError:  # pragma: no cover -- only hit on the slim (no-`docker`-extra) CI leg
    pytest.skip('docker extra not installed', allow_module_level=True)


@dataclass
class _ExecStep:
    """One scripted `exec` outcome for `_FakeClient`.

    A logical exec is one `exec_create` + `exec_start` + `exec_inspect`. `exit_code` is what
    `exec_inspect` reports (`None` simulates the daemon not recording an exit); `stdout`/`stderr`
    are what a demuxed `exec_start` returns; `drain` is the bytes the stdin half-close socket
    yields before EOF; `sendall_exc`, if set, makes the socket's `sendall` raise (the broken-pipe
    race where a pre-check exits before consuming stdin).
    """

    exit_code: int | None = 0
    stdout: bytes = b''
    stderr: bytes = b''
    drain: bytes = b''
    sendall_exc: BaseException | None = None


class _FakeRawSock:
    """Stand-in for the raw socket behind docker-py's `SocketIO` (`sock._sock`)."""

    def __init__(self, drain: bytes, sendall_exc: BaseException | None) -> None:
        self._chunks: list[bytes] = [drain, b''] if drain else [b'']
        self._sendall_exc = sendall_exc
        self.sent = b''

    def sendall(self, data: bytes) -> None:
        if self._sendall_exc is not None:
            raise self._sendall_exc
        self.sent += data

    def shutdown(self, how: int) -> None:
        pass

    def recv(self, _bufsize: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b''


class _FakeSocketIO:
    """Stand-in for the `SocketIO` returned by `exec_start(socket=True)`."""

    def __init__(self, drain: bytes, sendall_exc: BaseException | None) -> None:
        self._sock = _FakeRawSock(drain, sendall_exc)
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class _FakeClient:
    """In-memory stand-in for `docker.DockerClient`.

    Configurable per test: `run_result` is what `containers.run` returns (or an exception
    instance to raise); `get_result` is what `containers.get` returns (or an exception);
    `remove_exc` lets a test make `Container.remove` raise. Tracks calls so tests can
    assert "this method was never called" (the load-bearing rule for attach mode).
    """

    def __init__(self) -> None:
        self.run_result: Any = None
        self.get_result: Any = None
        self.remove_exc: BaseException | None = None
        self.calls: list[str] = []
        self.closed = False
        # Exit code returned by every internal `exec` (mkdir + tool probe). Default 0 so setup
        # succeeds; a test sets it non-zero to simulate the root-creation/probe failure path.
        self.exec_exit_code = 0
        # Optional script of per-exec outcomes for the file-I/O / shell methods. Each logical
        # exec pops the next step; when the queue is empty we fall back to `exec_exit_code` with
        # empty output (the lifecycle defaults, so existing tests are untouched). `exec_create_exc`
        # makes the next `exec_create` raise, simulating a daemon-side failure.
        self.exec_steps: deque[_ExecStep] = deque()
        self.exec_create_exc: BaseException | None = None
        self._current_step: _ExecStep | None = None

        client_self = self

        class _Containers:
            def run(self, image: str, command: list[str], **kwargs: Any) -> Any:
                client_self.calls.append(f'run({image!r}, kwargs={sorted(kwargs)!r})')
                result = client_self.run_result
                if isinstance(result, BaseException):
                    raise result
                return result

            def get(self, container_id: str) -> Any:
                client_self.calls.append(f'get({container_id!r})')
                result = client_self.get_result
                if isinstance(result, BaseException):
                    raise result
                return result

        class _API:
            """Minimal `client.api` surface: just enough to let `setup`'s internal mkdir + tool probe succeed."""

            def exec_create(self, container: str, cmd: list[str], **kwargs: Any) -> dict[str, str]:
                client_self.calls.append(f'exec_create({container!r}, {cmd!r})')
                if client_self.exec_create_exc is not None:
                    raise client_self.exec_create_exc
                client_self._current_step = client_self.exec_steps.popleft() if client_self.exec_steps else None
                return {'Id': 'fake-exec-id'}

            def exec_start(self, exec_id: str, **kwargs: Any) -> Any:
                step = client_self._current_step
                if kwargs.get('socket'):
                    return _FakeSocketIO(step.drain if step else b'', step.sendall_exc if step else None)
                return (step.stdout, step.stderr) if step is not None else (b'', b'')

            def exec_inspect(self, exec_id: str) -> dict[str, int | None]:
                step = client_self._current_step
                return {'ExitCode': step.exit_code if step is not None else client_self.exec_exit_code}

        self.containers = _Containers()
        self.api = _API()

    def close(self) -> None:
        self.closed = True


def _make_container(container_id: str, fake_client: _FakeClient) -> MagicMock:
    """Build a fake `Container` whose `.remove` raises if the client has a remove_exc set."""
    container = MagicMock()
    container.id = container_id

    def remove(force: bool) -> None:
        fake_client.calls.append(f'remove({container_id!r}, force={force})')
        if fake_client.remove_exc is not None:
            raise fake_client.remove_exc

    container.remove = remove
    return container


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeClient]:
    """Replace `docker.from_env` so tests get a `_FakeClient` instead of a real daemon."""
    client = _FakeClient()
    monkeypatch.setattr(docker_module.docker, 'from_env', lambda: client)  # pyright: ignore[reportPrivateImportUsage]
    yield client


# --- mode validation ---------------------------------------------------------


def test_neither_image_nor_container_raises() -> None:
    """`image` XOR `container` -- omitting both is a misconfiguration, not a sensible default."""
    with pytest.raises(ValueError, match='exactly one of `image`'):
        DockerEnvironment()


def test_both_image_and_container_raises() -> None:
    """Both is ambiguous: owned or attach? Refuse rather than pick silently."""
    with pytest.raises(ValueError, match='exactly one of `image`'):
        DockerEnvironment(image='python:3.12-slim', container='abc123')


def test_image_mode_starts_not_yet_started() -> None:
    """Owned mode: setup hasn't run, so `_started` is False and there's no bound container."""
    env = DockerEnvironment(image='python:3.12-slim')
    assert env._started is False  # pyright: ignore[reportPrivateUsage]
    assert env._container_id is None  # pyright: ignore[reportPrivateUsage]


def test_container_mode_binds_attached_id() -> None:
    """Attach mode binds the user's container id at construction; `_started` stays False so
    the base class's idempotency gate works normally."""
    env = DockerEnvironment(container='abc123')
    assert env._started is False  # pyright: ignore[reportPrivateUsage]
    assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]


# --- attach mode: setup/teardown branch to no-op on `self.image is None` -----


async def test_attach_mode_setup_runs_mkdir_and_probe(fake_docker: _FakeClient) -> None:
    """Attach mode runs `mkdir -p root` + the tool probe via exec_create against the user's
    container, but must not call `containers.run`/`get`/`remove`."""
    env = DockerEnvironment(container='abc123')
    await env.start()
    assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]
    assert env._started is True  # pyright: ignore[reportPrivateUsage]
    assert not any(c.startswith(('run(', 'get(', 'remove(')) for c in fake_docker.calls)


async def test_attach_mode_teardown_closes_client_only(fake_docker: _FakeClient) -> None:
    """Attach-mode `stop()` must close the SDK client but never touch the user's container."""
    env = DockerEnvironment(container='abc123')
    await env.start()
    fake_docker.calls.clear()
    await env.stop()
    assert not any(c.startswith(('run(', 'get(', 'remove(')) for c in fake_docker.calls)
    assert fake_docker.closed is True
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


# --- owned mode: setup success / failure --------------------------------------


async def test_setup_binds_container_id_on_run_success(fake_docker: _FakeClient) -> None:
    """Happy path: `containers.run` returns a Container; we bind its id."""
    fake_docker.run_result = _make_container('abc123', fake_docker)
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()

    assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]
    assert env._started is True  # pyright: ignore[reportPrivateUsage]


async def test_setup_raises_on_image_not_found(fake_docker: _FakeClient) -> None:
    """`ImageNotFound`: no container was created, nothing to clean up; `_started` stays False."""
    fake_docker.run_result = docker.errors.ImageNotFound('nosuchimage:latest not found')
    env = DockerEnvironment(image='nosuchimage:latest')

    with pytest.raises(RuntimeError, match='docker image not found'):
        await env.start()

    assert env._container_id is None  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_setup_raises_on_api_error(fake_docker: _FakeClient) -> None:
    """Generic `APIError` during run: surface as RuntimeError; do not mark started."""
    fake_docker.run_result = docker.errors.APIError('Cannot connect to the Docker daemon')
    env = DockerEnvironment(image='python:3.12-slim')

    with pytest.raises(RuntimeError, match='docker run failed'):
        await env.start()

    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_owned_setup_failure_removes_container(fake_docker: _FakeClient) -> None:
    """`containers.run` succeeds but the post-run `mkdir`/probe fails: setup must remove the
    container it created and close the client. Otherwise `_started` stays False, a later `stop()`
    no-ops, and the running container is leaked."""
    container = _make_container('abc123', fake_docker)
    fake_docker.run_result = container
    fake_docker.get_result = container
    fake_docker.exec_exit_code = 1  # `mkdir -p root` fails inside the container

    env = DockerEnvironment(image='python:3.12-slim')
    with pytest.raises(EnvSetupError):
        await env.start()

    assert any(c.startswith('remove(') for c in fake_docker.calls)
    assert fake_docker.closed is True
    assert env._started is False  # pyright: ignore[reportPrivateUsage]
    assert env._container_id is None  # pyright: ignore[reportPrivateUsage]


# --- owned mode: teardown success / "already gone" / daemon failure ----------


async def test_teardown_removes_owned_container(fake_docker: _FakeClient) -> None:
    """Happy path: `containers.get` + `container.remove(force=True)`; flag clears."""
    container = _make_container('abc123', fake_docker)
    fake_docker.run_result = container
    fake_docker.get_result = container
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()
    await env.stop()

    assert env._container_id is None  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]
    assert fake_docker.closed is True
    assert "remove('abc123', force=True)" in fake_docker.calls


async def test_teardown_swallows_not_found_from_get(fake_docker: _FakeClient) -> None:
    """`containers.get` raises `NotFound` (container removed out-of-band): swallow and finish."""
    fake_docker.run_result = _make_container('abc123', fake_docker)
    fake_docker.get_result = docker.errors.NotFound('No such container: abc123')
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()
    await env.stop()

    assert env._container_id is None  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_teardown_swallows_not_found_from_remove(fake_docker: _FakeClient) -> None:
    """`container.remove` raises `NotFound` (race with daemon GC): swallow and finish."""
    container = _make_container('abc123', fake_docker)
    fake_docker.run_result = container
    fake_docker.get_result = container
    fake_docker.remove_exc = docker.errors.NotFound('No such container: abc123')
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()
    await env.stop()

    assert env._container_id is None  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_teardown_propagates_api_error(fake_docker: _FakeClient) -> None:
    """`APIError` during remove (not NotFound): propagate; `_started` stays True so a retry is possible."""
    container = _make_container('abc123', fake_docker)
    fake_docker.run_result = container
    fake_docker.get_result = container
    fake_docker.remove_exc = docker.errors.APIError('Cannot connect to the Docker daemon')
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()
    with pytest.raises(docker.errors.APIError, match='Cannot connect'):
        await env.stop()

    assert env._started is True  # pyright: ignore[reportPrivateUsage]


# --- async context manager round trip ----------------------------------------


async def test_async_with_round_trip_in_owned_mode(fake_docker: _FakeClient) -> None:
    """`async with` flows through start/stop with both SDK calls in the expected order."""
    container = _make_container('abc123', fake_docker)
    fake_docker.run_result = container
    fake_docker.get_result = container

    env = DockerEnvironment(image='python:3.12-slim')
    async with env as bound:
        assert bound is env
        assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]
    assert env._container_id is None  # pyright: ignore[reportPrivateUsage]


# --- file-I/O + shell error mapping ------------------------------------------
#
# These drive the error branches that a real daemon can't easily (or ever) produce: a missing
# tool, an exec called before start(), a daemon `APIError`, a daemon that records no exit code,
# and the per-method sentinel-exit mappings. The fake client scripts each exec's outcome, so the
# decision tree is covered deterministically without a daemon (the live suite proves real
# behavior on top of this).


async def _started_env(fake: _FakeClient) -> DockerEnvironment:
    """Start an owned-mode env against the fake (setup uses the success defaults)."""
    container = _make_container('abc123', fake)
    fake.run_result = container
    fake.get_result = container
    env = DockerEnvironment(image='python:3.12-slim')
    await env.start()
    return env


async def test_setup_raises_when_required_tool_missing(fake_docker: _FakeClient) -> None:
    """mkdir succeeds but the tool probe reports a missing binary -> EnvSetupError naming it."""
    container = _make_container('abc123', fake_docker)
    fake_docker.run_result = container
    fake_docker.get_result = container
    fake_docker.exec_steps = deque([_ExecStep(exit_code=0), _ExecStep(exit_code=1, stdout=b'rg')])

    env = DockerEnvironment(image='python:3.12-slim')
    with pytest.raises(EnvSetupError, match="missing required tool 'rg'"):
        await env.start()


async def test_exec_before_start_raises_runtime_error() -> None:
    """A file op before start() has no client/container bound -> RuntimeError, not a crash."""
    env = DockerEnvironment(image='python:3.12-slim')
    with pytest.raises(RuntimeError, match='not started'):
        await env.read_file('x')


async def test_exec_create_api_error_becomes_shell_execution_error(fake_docker: _FakeClient) -> None:
    """A daemon `APIError` on exec_create surfaces as EnvShellExecutionError, not a raw APIError."""
    env = await _started_env(fake_docker)
    fake_docker.exec_create_exc = docker.errors.APIError('daemon gone')
    with pytest.raises(EnvShellExecutionError, match='exec_create failed'):
        await env.read_file('x')


async def test_exec_missing_exit_code_raises(fake_docker: _FakeClient) -> None:
    """`exec_inspect` with no ExitCode means inconsistent daemon state -> EnvShellExecutionError."""
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=None)])
    with pytest.raises(EnvShellExecutionError, match='no exit code'):
        await env.read_file('x')


async def test_read_file_permission_denied(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=_EXIT_PERMISSION_DENIED)])
    with pytest.raises(EnvPermissionError, match='not readable'):
        await env.read_file('x')


async def test_read_file_generic_error_is_read_error(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=1, stderr=b'io error')])
    with pytest.raises(EnvReadError, match='io error'):
        await env.read_file('x')


async def test_write_file_mkdir_failure_is_write_error(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=1, stderr=b'no space')])
    with pytest.raises(EnvWriteError, match='mkdir'):
        await env.write_file('d/x', b'data')


async def test_write_file_permission_denied(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=0), _ExecStep(exit_code=_EXIT_PERMISSION_DENIED)])
    with pytest.raises(EnvPermissionError, match='not writable'):
        await env.write_file('x', b'data')


async def test_write_file_generic_error_is_write_error(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    # The write goes through stdin; its diagnostic text comes back on the drained stream, which
    # `_exec_with_stdin` returns as stderr -- so the failure message is scripted via `drain`.
    fake_docker.exec_steps = deque([_ExecStep(exit_code=0), _ExecStep(exit_code=1, drain=b'disk error')])
    with pytest.raises(EnvWriteError, match='disk error'):
        await env.write_file('x', b'data')


async def test_write_file_drains_socket_output(fake_docker: _FakeClient) -> None:
    """`_exec_with_stdin` drains the stream after EOF; a non-empty chunk exercises the drain loop."""
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=0), _ExecStep(exit_code=0, drain=b'noise')])
    await env.write_file('x', b'data')  # success despite drained bytes


async def test_write_file_survives_broken_pipe_on_stdin(fake_docker: _FakeClient) -> None:
    """A pre-check that exits before reading stdin closes the remote end; the resulting broken
    pipe must be swallowed so the real cause (target is a directory) surfaces as EnvWriteError."""
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque(
        [_ExecStep(exit_code=0), _ExecStep(exit_code=_EXIT_IS_DIRECTORY, sendall_exc=BrokenPipeError())]
    )
    with pytest.raises(EnvWriteError, match='existing directory'):
        await env.write_file('x', b'data' * 100_000)


async def test_ls_permission_denied(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=_EXIT_PERMISSION_DENIED)])
    with pytest.raises(EnvPermissionError, match='not listable'):
        await env.ls('.')


async def test_ls_generic_error_is_read_error(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=1, stderr=b'ls boom')])
    with pytest.raises(EnvReadError, match='ls boom'):
        await env.ls('.')


async def test_ls_skips_blank_lines(fake_docker: _FakeClient) -> None:
    """A stray blank line in `ls` output is skipped, not turned into an empty-named entry."""
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=0, stdout=b'a\n\nsub/\n')])
    entries = {f.name: f.is_directory for f in await env.ls('.')}
    assert entries == {'a': False, 'sub': True}


async def test_glob_ripgrep_missing_is_shell_execution_error(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=0), _ExecStep(exit_code=_EXIT_COMMAND_NOT_FOUND)])
    with pytest.raises(EnvShellExecutionError, match='ripgrep'):
        await env.glob('.', '*.py')


async def test_glob_invalid_pattern_is_model_fixable(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque(
        [_ExecStep(exit_code=0), _ExecStep(exit_code=RG_EXIT_USAGE_OR_PATTERN, stderr=b'bad glob')]
    )
    with pytest.raises(EnvInvalidPatternError, match='bad glob'):
        await env.glob('.', '[')


async def test_glob_walk_error_is_read_error(fake_docker: _FakeClient) -> None:
    """rg exit 2 WITH stdout = a partial walk that hit an I/O error -> EnvReadError, not pattern."""
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque(
        [_ExecStep(exit_code=0), _ExecStep(exit_code=RG_EXIT_USAGE_OR_PATTERN, stdout=b'partial', stderr=b'walk error')]
    )
    with pytest.raises(EnvReadError, match='walk error'):
        await env.glob('.', '*.py')


async def test_shell_command_before_start_raises_runtime_error() -> None:
    env = DockerEnvironment(image='python:3.12-slim')
    with pytest.raises(RuntimeError, match='not started'):
        await env.shell_command('echo hi')


async def test_shell_command_exec_create_api_error(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_create_exc = docker.errors.APIError('daemon gone')
    with pytest.raises(EnvShellExecutionError, match='exec_create failed'):
        await env.shell_command('echo hi')


async def test_shell_command_missing_exit_code_raises(fake_docker: _FakeClient) -> None:
    env = await _started_env(fake_docker)
    fake_docker.exec_steps = deque([_ExecStep(exit_code=None)])
    with pytest.raises(EnvShellExecutionError, match='no exit code'):
        await env.shell_command('echo hi')
