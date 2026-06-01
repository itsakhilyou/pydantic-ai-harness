"""Docker-backed execution environment.

Lifecycle (`setup`/`teardown`) is implemented; tool methods (`read_file`, `write_file`,
`ls`, `grep`, `glob`, `shell_command`) raise `NotImplementedError` for now and land in
follow-up chunks. The lifecycle alone is meaningful: it lets `ExecutionEnv` wire and
manage the container across an agent run with the correct ownership semantics, so the
rest of the backend can land incrementally without touching the contract.

Failure-handling patterns adopted here come from prior art -- see
`agent_docs/environment-lifecycle.md` "Backend implementer's guide" for the bug
reports that justify each one. Summary:

- Container id is bound to `self._container_id` at line 1 of `setup`, before any
  flaky step. Cleanup-on-failure uses that id; the orphan window is zero lines long.
- Every `await` has an absolute timeout (`startup_timeout`, `teardown_timeout`).
- `teardown` swallows only "no such container" (the idempotent / already-gone case);
  any other Docker failure propagates, leaving `_started=True` so a retry is possible.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field

from .abstract import AbstractEnvironment, AbstractFile, AbstractMatch, ShellCommandResult


class _DockerNotFound(Exception):
    """`docker` reported the container as not present.

    Specifically distinguishes the *benign* idempotent outcome ("we asked to remove
    something that's already gone -- the state we wanted") from every other Docker
    failure (daemon unreachable, permission denied, networking). Used by `teardown` to
    decide what to swallow; everything else propagates.
    """


async def _run_docker(*args: str, timeout: float) -> tuple[int, bytes, bytes]:
    """Run `docker <args>` with an absolute timeout, returning (returncode, stdout, stderr).

    Subprocess-level errors (docker CLI missing, spawn failure) propagate as `OSError`; the
    caller wraps them. Timeout enforced here per the implementer's guide: no internal await
    is unbounded.
    """
    proc = await asyncio.create_subprocess_exec(
        'docker',
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # The daemon is hung or the operation is genuinely slow. Kill the client so we don't
        # leak a pending coroutine; the in-daemon work may continue but at least we return.
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    return proc.returncode or 0, stdout, stderr


@dataclass(kw_only=True)
class DockerEnvironment(AbstractEnvironment):
    """Docker-backed environment with lifecycle implemented; tool methods coming next.

    Two modes determined at construction:

    - **Owned** (`image=...`): we create the container in `setup` and remove it in
      `teardown`. The container's lifetime is bound to the env's start/stop cycle.
    - **Attach** (`container=...`): we use a container someone else created. `_started`
      is pre-seeded to `True` so the base class's `start`/`stop` short-circuit and
      we never run `docker run` or `docker rm` on a resource we don't own.

    Exactly one of `image` / `container` must be set; passing both or neither raises.
    """

    image: str | None = None
    """Docker image to run in *owned* mode. Mutually exclusive with `container`.

    Intentionally no default: matching the dmontagu/Douwe call in #4393, the user
    chooses the image (and with it, the Python version, OS, and tooling). Defaulting
    to e.g. `python:3.12-slim` would silently force an opinion that may not match the
    user's host.
    """

    container: str | None = None
    """Existing container id/name to attach to in *attach* mode. Mutually exclusive with `image`.

    Used for sharing a container managed out-of-band (CI fixture, devcontainer,
    sidecar). The harness will not start or stop a container it didn't create.
    """

    root: str = '/workspace'
    """Container path used as the agent's working root.

    Default `/workspace` matches dmontagu's #4393. The agent populates this directory
    via `write_file`; we deliberately do not bind-mount the host cwd, which is brittle
    on macOS Docker Desktop (`docker-prior-art.md:170-171`) and unsafe for remote daemons.
    """

    startup_timeout: float = 10.0
    """Absolute timeout for `setup`: `docker run` + any readiness check, in seconds.

    Bounded per the implementer's guide so a hung daemon can't hang the agent. Tuned
    short because `docker run -d` returns as soon as the container is created; longer
    waits are signal, not patience.
    """

    teardown_timeout: float = 5.0
    """Absolute timeout for `teardown`: `docker rm -f`, in seconds.

    Bounded for the same reason; a hung `docker rm` blocks the agent's `wrap_run`
    exit, which means a hung agent.
    """

    _container_id: str = field(init=False, default='')
    """The container id we own (or that was attached); empty when no container is bound."""

    def __post_init__(self) -> None:
        """Validate mode (image XOR container) and bind the attach-mode container id.

        Attach mode does **not** pre-seed `_started`. Instead, `setup` and `teardown`
        branch on `self.image is None`: in attach mode they are no-ops. This keeps both
        modes on the same idempotency gate (the base class's `_started` flag) and makes
        it physically impossible for the harness to run `docker run` or `docker rm` on
        a container the user owns -- the work paths simply do not execute.
        """
        if (self.image is None) == (self.container is None):
            raise ValueError(
                'DockerEnvironment requires exactly one of `image` (owned mode) or '
                '`container` (attach mode); got '
                f'image={self.image!r}, container={self.container!r}.'
            )
        if self.container is not None:
            self._container_id = self.container

    async def setup(self) -> None:
        """`docker run -d` in owned mode; no-op in attach mode (we don't own the container)."""
        if self.image is None:
            # Attach mode: the container exists outside our control; nothing to allocate.
            # `_container_id` was set in `__post_init__` from `self.container`.
            return

        # PATTERN 1 (acquire-then-protect): bind the id at line 1 -- before any other
        # await -- so a failure in any later step has a handle to clean up. `docker run -d`
        # prints the container id on stdout; this is the only line where the orphan window
        # can open. When we add readiness probes later, they go in a try/except whose cleanup
        # path force-removes `self._container_id` and re-raises.
        returncode, stdout, stderr = await _run_docker(
            'run', '-d', self.image, 'sleep', 'infinity', timeout=self.startup_timeout
        )
        if returncode != 0:
            # `docker run` itself failed: no container was created. Nothing to clean up.
            raise RuntimeError(f'docker run failed (exit {returncode}): {stderr.decode(errors="replace").strip()}')
        self._container_id = stdout.decode().strip()

    async def teardown(self) -> None:
        """`docker rm -f` in owned mode; no-op in attach mode (we don't kill what we didn't create)."""
        if self.image is None:
            # Attach mode: explicit guard, not a side effect of a flag. The harness never
            # touches a container the user owns -- this is the load-bearing rule and it lives
            # here in the work path, not as an emergent property of `_started` bookkeeping.
            return

        # PATTERN 3: swallow only the idempotent benign case ("already gone" -- the outcome
        # we wanted). Daemon errors propagate so the base class keeps `_started=True` and
        # the caller can retry. The OpenHands antipattern we explicitly avoid is blanket
        # `except APIError: pass`, which hides real cleanup failures from the operator.
        try:
            await self._docker_rm_force(self._container_id)
        except _DockerNotFound:
            pass
        self._container_id = ''

    async def _docker_rm_force(self, container_id: str) -> None:
        """Run `docker rm -f <id>`, raising `_DockerNotFound` for the idempotent case only.

        Helper exists so `teardown` and any future cleanup-on-failure path use the same
        "already gone" recognition logic in one place.
        """
        returncode, _, stderr = await _run_docker('rm', '-f', container_id, timeout=self.teardown_timeout)
        if returncode == 0:
            return
        # docker exits 1 with "Error: No such container: ..." on stderr when the id is gone.
        # Anything else (daemon down, permission, network) is real and must surface.
        if b'No such container' in stderr:
            raise _DockerNotFound(stderr.decode(errors='replace').strip())
        raise RuntimeError(f'docker rm failed (exit {returncode}): {stderr.decode(errors="replace").strip()}')

    async def read_file(self, path: str) -> bytes:
        """Coming next chunk: `docker cp` byte-accurate read."""
        raise NotImplementedError  # pragma: no cover

    async def write_file(self, path: str, data: bytes) -> None:
        """Coming next chunk: `docker cp` byte-accurate write."""
        raise NotImplementedError  # pragma: no cover

    async def ls(self, path: str) -> list[AbstractFile]:
        """Coming next chunk: `docker exec ls -A1p`."""
        raise NotImplementedError  # pragma: no cover

    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:
        """Coming next chunk: `docker exec rg --json`."""
        raise NotImplementedError  # pragma: no cover

    async def glob(self, path: str, pattern: str) -> list[str]:
        """Coming next chunk: `docker exec find` with pattern translation."""
        raise NotImplementedError  # pragma: no cover

    async def shell_command(self, command: str, timeout: float | None = None) -> ShellCommandResult:
        """Owned by the user: `docker exec` + in-container `timeout -k` + pgid kill on the host backstop."""
        raise NotImplementedError  # pragma: no cover
