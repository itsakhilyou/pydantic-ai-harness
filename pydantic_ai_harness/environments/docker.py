"""Docker-backed execution environment.

Lifecycle (`setup`/`teardown`) is implemented via the `docker` Python SDK; tool methods
(`read_file`, `write_file`, `ls`, `grep`, `glob`, `shell_command`) raise
`NotImplementedError` for now and land in follow-up chunks.

Failure-handling patterns come from prior art -- see
`agent_docs/environment-lifecycle.md` "Backend implementer's guide":

- Container id is bound from the `Container` object returned by `containers.run`,
  before any other await -- the orphan window is zero lines long.
- Every blocking SDK call runs under `asyncio.wait_for(asyncio.to_thread(...))` so a
  hung daemon can't hang the agent.
- `teardown` swallows only `docker.errors.NotFound` (the idempotent "already gone"
  case). Every other SDK error (`APIError`, connection, permission) propagates,
  leaving `_started=True` so a retry is possible.
"""

import asyncio
import contextlib
import shlex
import uuid
from dataclasses import dataclass, field

import docker
import docker.errors
from docker import DockerClient

from .abstract import AbstractEnvironment, AbstractFile, AbstractMatch, ShellCommandResult
from .exceptions import EnvShellExecutionError


@dataclass(kw_only=True)
class DockerEnvironment(AbstractEnvironment):
    """Docker-backed environment with lifecycle implemented; tool methods coming next.

    Two modes determined at construction:

    - **Owned** (`image=...`): we create the container in `setup` and remove it in
      `teardown`. The container's lifetime is bound to the env's start/stop cycle.
    - **Attach** (`container=...`): we use a container someone else created.
      `setup`/`teardown` are no-ops; the harness never starts or stops a container it
      didn't create.

    Exactly one of `image` / `container` must be set; passing both or neither raises.
    """

    image: str | None = None
    """Docker image to run in *owned* mode. Mutually exclusive with `container`.

    Intentionally no default: the user chooses the image (and with it the Python version,
    OS, and tooling). Defaulting to e.g. `python:3.12-slim` would silently force an
    opinion that may not match the user's host.
    """

    container: str | None = None
    """Existing container id/name to attach to in *attach* mode. Mutually exclusive with `image`.

    Used for sharing a container managed out-of-band (CI fixture, devcontainer, sidecar).
    The harness will not start or stop a container it didn't create.
    """

    root: str = '/workspace'
    """Container path used as the agent's working root.

    Populated by the agent via `write_file`; we deliberately do not bind-mount the host
    cwd, which is brittle on macOS Docker Desktop and unsafe for remote daemons.
    """

    startup_timeout: float = 10.0
    """Absolute timeout for `setup` (`containers.run`), in seconds."""

    teardown_timeout: float = 5.0
    """Absolute timeout for `teardown` (`container.remove(force=True)`), in seconds."""

    _container_id: str | None = field(init=False, default=None)
    """The container id we own (or that was attached); empty when no container is bound."""

    _client: DockerClient | None = field(init=False, default=None)
    """The Docker SDK client; created in `setup`, closed in `teardown`. `None` in attach mode."""

    def __post_init__(self) -> None:
        """Validate mode (image XOR container) and bind the attach-mode container id."""
        if (self.image is None) == (self.container is None):
            raise ValueError(
                'DockerEnvironment requires exactly one of `image` (owned mode) or '
                '`container` (attach mode); got '
                f'image={self.image!r}, container={self.container!r}.'
            )
        if self.container is not None:
            self._container_id = self.container

    async def setup(self) -> None:
        """`containers.run` in owned mode; no-op in attach mode."""
        if self.image is None:
            return

        client = await asyncio.to_thread(docker.from_env)
        self._client = client
        try:
            # `containers.run(detach=True)` returns a Container as soon as the container is
            # created. We bind `_container_id` from its `.id` before any other await -- the
            # acquire-then-protect line.
            container = await asyncio.wait_for(
                asyncio.to_thread(client.containers.run, self.image, ['sleep', 'infinity'], detach=True),
                timeout=self.startup_timeout,
            )
        except docker.errors.ImageNotFound as exc:
            raise RuntimeError(f'docker image not found: {exc.explanation}') from exc
        except docker.errors.APIError as exc:
            raise RuntimeError(f'docker run failed: {exc.explanation}') from exc
        self._container_id = container.id

    async def teardown(self) -> None:
        """`container.remove(force=True)` in owned mode; no-op in attach mode.

        Swallows only `NotFound` (the idempotent already-gone case). Every other SDK
        error propagates so the caller can retry or escalate; the base class keeps
        `_started=True` on raise, so the resource is not silently forgotten.
        """
        if self.image is None:
            return
        client = self._client
        if client is None:  # pragma: no cover -- setup never completed
            return

        container_id = self._container_id
        if container_id is None:  # pragma: no cover -- setup never bound an id
            return
        try:
            container = await asyncio.to_thread(client.containers.get, container_id)
            await asyncio.wait_for(
                asyncio.to_thread(container.remove, force=True),
                timeout=self.teardown_timeout,
            )
        except docker.errors.NotFound:
            # Container is already gone: the state we wanted. Idempotent.
            pass
        self._container_id = None
        await asyncio.to_thread(client.close)
        self._client = None

    async def read_file(self, path: str) -> bytes:
        """Coming next chunk."""
        raise NotImplementedError  # pragma: no cover

    async def write_file(self, path: str, data: bytes) -> None:
        """Coming next chunk."""
        raise NotImplementedError  # pragma: no cover

    async def ls(self, path: str) -> list[AbstractFile]:
        """Coming next chunk."""
        raise NotImplementedError  # pragma: no cover

    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:
        """Coming next chunk."""
        raise NotImplementedError  # pragma: no cover

    async def glob(self, path: str, pattern: str) -> list[str]:
        """Coming next chunk."""
        raise NotImplementedError  # pragma: no cover

    async def shell_command(self, command: str, timeout: float | None = None) -> ShellCommandResult:
        """Run `command` in the container and return stdout, stderr, and the exit code.

        Cancellation/timeout strategy: a launcher records its own PID into a per-exec pidfile
        before `exec`-replacing itself with the user's shell, so the recorded PID *is* the user's
        process. On timeout/cancel a second `docker exec` runs `kill -KILL "$(cat <pidfile>)"`
        inside the container -- the daemon performs the signal in the container's PID namespace,
        which works on Linux, Docker Desktop, Colima, Podman-VM, and remote daemons alike.
        Host-side `os.killpg` is unreachable across the VM boundary on macOS, so we never use it.
        """
        # Guard via locals so pyright can narrow `client` and `container_id` to non-None
        # for the rest of the method (attribute narrowing doesn't survive across awaits).
        client = self._client
        container_id = self._container_id
        if client is None or container_id is None:
            raise RuntimeError('environment not started; call start() first')

        # Per-exec pidfile so concurrent `shell_command` calls don't race on the same path.
        pidfile = f'/tmp/.harness-exec-{uuid.uuid4().hex}.pid'
        # `exec` replaces the outer shell with the user's `sh -c <command>` while keeping
        # the same PID, so the PID we wrote into the pidfile is the actual command's PID.
        launcher = f'echo $$ > {pidfile}; exec /bin/sh -c {shlex.quote(command)}'

        # Upstream typeshed gap: in `docker-stubs/api/exec_api.pyi`, `exec_create`'s required
        # positional args (`container`, `cmd`) are missing annotations while every optional
        # kwarg is typed. `exec_start` and `exec_inspect` in the same stub are fully typed.
        exec_create = client.api.exec_create  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        try:
            exec_info = await asyncio.to_thread(exec_create, container_id, cmd=['/bin/sh', '-c', launcher])  # pyright: ignore[reportUnknownArgumentType]
        except docker.errors.APIError as exc:
            raise EnvShellExecutionError(f'docker exec_create failed: {exc.explanation}') from exc
        exec_id = exec_info['Id']

        exec_start = client.api.exec_start
        exec_task = asyncio.create_task(asyncio.to_thread(exec_start, exec_id, demux=True))

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(asyncio.shield(exec_task), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill_remote_exec(client, container_id, pidfile)
            stdout_bytes, stderr_bytes = await exec_task
            timed_out = True
        except asyncio.CancelledError:
            await self._kill_remote_exec(client, container_id, pidfile)
            with contextlib.suppress(BaseException):
                await exec_task
            raise

        info = await asyncio.to_thread(client.api.exec_inspect, exec_id)
        return ShellCommandResult(
            stdout=stdout_bytes or b'',
            stderr=stderr_bytes or b'',
            return_code=info['ExitCode'],
            timed_out=timed_out,
        )

    async def _kill_remote_exec(self, client: DockerClient, container_id: str, pidfile: str) -> None:
        """Send SIGKILL to the in-container process recorded in `pidfile` via a second exec.

        Tolerant by design: a missing pidfile or a PID that has already exited collapses to
        `kill 0`, which the trailing `|| true` swallows. Errors during cleanup must not mask
        the original timeout or cancellation that triggered this.
        """
        kill_cmd = f'kill -KILL "$(cat {pidfile} 2>/dev/null || echo 0)" 2>/dev/null || true'
        exec_create = client.api.exec_create  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        with contextlib.suppress(docker.errors.APIError):
            kill_info = await asyncio.to_thread(exec_create, container_id, cmd=['/bin/sh', '-c', kill_cmd])  # pyright: ignore[reportUnknownArgumentType]
            await asyncio.to_thread(client.api.exec_start, kill_info['Id'])
