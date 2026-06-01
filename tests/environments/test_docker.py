"""DockerEnvironment lifecycle tests with the `docker` CLI faked out.

These cover the failure modes called out in `agent_docs/environment-lifecycle.md`
"Backend implementer's guide" -- mode validation, attach-mode pre-seeding, owned-mode
setup/teardown success and failure paths, and the "already gone" idempotent case. They
run without a real Docker daemon by monkeypatching the module-level `_run_docker`
helper to return canned `(returncode, stdout, stderr)` tuples.

Real-daemon integration tests will land alongside the file I/O methods; the unit
coverage here is what proves the lifecycle decision tree, not the daemon plumbing.
"""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from pydantic_ai_harness.environments import docker as docker_module
from pydantic_ai_harness.environments.docker import DockerEnvironment, _run_docker


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[int, bytes, bytes]]]:
    """Queue of canned `_run_docker` responses; each call to docker pops the next one.

    Pattern matches the monkeypatch style in test_local.py: replace the boundary helper
    rather than `asyncio.create_subprocess_exec` directly, since the boundary is the
    surface DockerEnvironment depends on.
    """
    responses: list[tuple[int, bytes, bytes]] = []
    calls: list[tuple[str, ...]] = []

    async def fake_run_docker(*args: str, timeout: float) -> tuple[int, bytes, bytes]:
        calls.append(args)
        if not responses:  # pragma: no cover
            raise AssertionError(f'unexpected docker call: {args!r}')
        return responses.pop(0)

    monkeypatch.setattr(docker_module, '_run_docker', fake_run_docker)
    yield responses
    # `calls` is exposed via closure if a test needs it; for now we just assert on responses.


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
    assert env._container_id == ''  # pyright: ignore[reportPrivateUsage]


def test_container_mode_binds_attached_id() -> None:
    """Attach mode binds the user's container id at construction; `_started` stays False so
    the base class's idempotency gate works normally. The harness never touches a container
    it didn't create because `setup`/`teardown` branch on `self.image is None`, not because
    of a pre-seeded flag."""
    env = DockerEnvironment(container='abc123')
    assert env._started is False  # pyright: ignore[reportPrivateUsage]
    assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]


# --- attach mode: setup/teardown branch to no-op on `self.image is None` -----


async def test_attach_mode_setup_runs_no_docker(fake_docker: list[tuple[int, bytes, bytes]]) -> None:
    """`start()` in attach mode calls `setup`, which returns immediately without running docker.
    The flag flips so subsequent calls are idempotent; the user's container is never touched."""
    env = DockerEnvironment(container='abc123')
    await env.start()  # would raise via fake_docker if any docker call were made
    assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]
    assert env._started is True  # pyright: ignore[reportPrivateUsage]


async def test_attach_mode_teardown_runs_no_docker(fake_docker: list[tuple[int, bytes, bytes]]) -> None:
    """`stop()` in attach mode must NEVER run `docker rm` on the user's container -- that is
    the load-bearing safety rule. We pin it by asserting zero docker calls across a full
    start/stop round trip; if the assertion fails, the harness killed a container it didn't own."""
    env = DockerEnvironment(container='abc123')
    await env.start()
    await env.stop()
    # Zero responses consumed = zero docker calls made = the user's container is intact.
    assert fake_docker == [], 'attach mode must not invoke docker; user owns lifecycle'
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


# --- owned mode: setup success / failure --------------------------------------


async def test_setup_binds_container_id_on_docker_run_success(
    fake_docker: list[tuple[int, bytes, bytes]],
) -> None:
    """Happy path: `docker run -d <image>` returns the container id on stdout; we bind it."""
    fake_docker.append((0, b'abc123\n', b''))
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()

    assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]
    assert env._started is True  # pyright: ignore[reportPrivateUsage]


async def test_setup_raises_with_stderr_when_docker_run_fails(
    fake_docker: list[tuple[int, bytes, bytes]],
) -> None:
    """`docker run` exits nonzero (bad image, network issue): surface stderr; do NOT mark started.

    No container was created, so there's nothing to clean up; the next `start()` will retry.
    This is the base class's failure policy (`_started` stays False on `setup` raise).
    """
    fake_docker.append((125, b'', b'Unable to find image nosuchimage:latest'))
    env = DockerEnvironment(image='nosuchimage:latest')

    with pytest.raises(RuntimeError, match='Unable to find image'):
        await env.start()

    assert env._container_id == ''  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


# --- owned mode: teardown success / "already gone" / daemon failure ----------


async def test_teardown_runs_docker_rm_on_owned_container(
    fake_docker: list[tuple[int, bytes, bytes]],
) -> None:
    """Happy path: `docker rm -f <id>` succeeds; we clear `_container_id` and the flag."""
    fake_docker.append((0, b'abc123\n', b''))  # docker run
    fake_docker.append((0, b'abc123\n', b''))  # docker rm
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()
    await env.stop()

    assert env._container_id == ''  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_teardown_swallows_no_such_container(
    fake_docker: list[tuple[int, bytes, bytes]],
) -> None:
    """ "Already gone" is the outcome teardown wanted; swallow rather than surface as an error.

    Triggered in real life by: container OOM-killed, manually removed, daemon restarted, or
    a previous teardown that succeeded on the daemon but lost its response. The base class's
    idempotency then becomes meaningful: a second `stop()` after a daemon hiccup must succeed.
    """
    fake_docker.append((0, b'abc123\n', b''))  # docker run
    fake_docker.append((1, b'', b'Error: No such container: abc123'))  # docker rm: already gone
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()
    await env.stop()  # must not raise

    assert env._container_id == ''  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_teardown_propagates_real_daemon_error(
    fake_docker: list[tuple[int, bytes, bytes]],
) -> None:
    """Daemon failure (not "no such container"): propagate. Leaves `_started=True` per the
    base contract so the caller can retry or escalate -- the OpenHands antipattern we
    deliberately avoid is blanket-suppressing this case."""
    fake_docker.append((0, b'abc123\n', b''))  # docker run
    fake_docker.append((1, b'', b'Cannot connect to the Docker daemon'))  # docker rm: daemon dead
    env = DockerEnvironment(image='python:3.12-slim')

    await env.start()
    with pytest.raises(RuntimeError, match='Cannot connect to the Docker daemon'):
        await env.stop()

    assert env._started is True  # pyright: ignore[reportPrivateUsage]


# --- async context manager round trip ----------------------------------------


async def test_async_with_round_trip_in_owned_mode(
    fake_docker: list[tuple[int, bytes, bytes]],
) -> None:
    """`async with` flows through start/stop with both docker calls in the expected order."""
    fake_docker.append((0, b'abc123\n', b''))  # docker run
    fake_docker.append((0, b'abc123\n', b''))  # docker rm

    env = DockerEnvironment(image='python:3.12-slim')
    async with env as bound:
        assert bound is env
        assert env._container_id == 'abc123'  # pyright: ignore[reportPrivateUsage]
    assert env._container_id == ''  # pyright: ignore[reportPrivateUsage]


# --- _run_docker: subprocess plumbing (success + timeout) --------------------


class _FakeProc:
    """Stand-in for an `asyncio.subprocess.Process` driven by canned values.

    Just enough surface to satisfy `_run_docker`: `communicate` returns the pre-set
    output and the `returncode` attribute is set; `kill` + `wait` exist for the timeout
    path so the helper can clean up its client subprocess.
    """

    def __init__(self, returncode: int, stdout: bytes, stderr: bytes, *, hang: bool = False) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            # Sleep longer than any test timeout so `wait_for` always trips first.
            await asyncio.sleep(60)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


async def test_run_docker_returns_captured_subprocess_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: `_run_docker` returns the subprocess's exit code, stdout, and stderr."""
    proc = _FakeProc(returncode=0, stdout=b'cid\n', stderr=b'')

    async def fake_exec(*args: str, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_exec)

    rc, stdout, stderr = await _run_docker('run', '-d', 'img', timeout=1.0)
    assert (rc, stdout, stderr) == (0, b'cid\n', b'')


async def test_run_docker_timeout_kills_subprocess_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the daemon hangs past `timeout`, the helper kills the client subprocess and
    propagates `asyncio.TimeoutError` -- the bounded-await guarantee from the implementer's
    guide. Without this, a hung daemon would freeze every `setup` / `teardown`."""
    proc = _FakeProc(returncode=0, stdout=b'', stderr=b'', hang=True)

    async def fake_exec(*args: str, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_exec)

    with pytest.raises(asyncio.TimeoutError):
        await _run_docker('rm', '-f', 'cid', timeout=0.05)
    assert proc.killed is True
