"""DockerEnvironment lifecycle tests with the `docker` SDK faked out.

Covers the failure modes called out in `agent_docs/environment-lifecycle.md`
"Backend implementer's guide": mode validation, attach-mode setup/teardown as no-ops,
owned-mode setup/teardown success and failure paths, and the "already gone" idempotent
case. Runs without a real Docker daemon by monkeypatching `docker.from_env` to return
a fake client.

Real-daemon integration tests will land alongside the file I/O methods; the unit
coverage here is what proves the lifecycle decision tree, not the daemon plumbing.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import docker.errors
import pytest

from pydantic_ai_harness.environments import docker as docker_module
from pydantic_ai_harness.environments.docker import DockerEnvironment
from pydantic_ai_harness.environments.exceptions import EnvSetupError


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
                return {'Id': 'fake-exec-id'}

            def exec_start(self, exec_id: str, **kwargs: Any) -> tuple[bytes, bytes]:
                return (b'', b'')

            def exec_inspect(self, exec_id: str) -> dict[str, int]:
                return {'ExitCode': client_self.exec_exit_code}

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
