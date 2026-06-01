"""Docker-backed execution environment.

Lifecycle (`setup`/`teardown`) and all tool methods (`read_file`, `write_file`, `ls`,
`grep`, `glob`, `shell_command`) are implemented via the `docker` Python SDK, running
each operation inside the container with `docker exec`.

Failure-handling patterns come from prior art -- see
`agent_docs/environment-lifecycle.md` "Backend implementer's guide":

- Container id is bound from the `Container` object returned by `containers.run`,
  before any other await -- the orphan window is zero lines long.
- Every blocking SDK call runs under `asyncio.wait_for(_run_blocking(...))` (a
  module-level `ThreadPoolExecutor`) so a hung daemon can't hang the agent and
  repeated timeouts don't leak threads.
- `teardown` swallows only `docker.errors.NotFound` (the idempotent "already gone"
  case). Every other SDK error (`APIError`, connection, permission) propagates,
  leaving `_started=True` so a retry is possible.
"""

import asyncio
import contextlib
import posixpath
import shlex
import socket
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import ParamSpec, TypeVar

try:
    import docker
    import docker.errors
    from docker import DockerClient
except ImportError as _import_error:  # pragma: no cover -- only hit when the `docker` extra is absent
    raise ImportError(
        'Please install the `docker` package to use `DockerEnvironment`, '
        'you can use the `docker` optional group — `pip install "pydantic-ai-harness[docker]"`'
    ) from _import_error

from ._ripgrep import RG_EXIT_USAGE_OR_PATTERN, parse_ripgrep_json
from .abstract import AbstractEnvironment, AbstractFile, AbstractMatch, ShellCommandResult
from .exceptions import (
    EnvInvalidPatternError,
    EnvIsADirectoryError,
    EnvNotADirectoryError,
    EnvNotFoundError,
    EnvPermissionError,
    EnvReadError,
    EnvSetupError,
    EnvShellExecutionError,
    EnvWriteError,
    PathEscapeError,
)

# Label keys for containers we create in owned mode. `prune_orphans` filters by the session
# label; the created-at label lets the reaper honor an `older_than` cutoff.
_SESSION_LABEL = 'com.pydantic.harness.session'
_CREATED_LABEL = 'com.pydantic.harness.created'

# Tools the harness assumes are present in the image. Probed once at `setup` so a broken
# image fails immediately with a useful name, rather than per-method with cryptic exit codes.
_REQUIRED_TOOLS = ('sh', 'cat', 'mkdir', 'kill', 'tar')

_P = ParamSpec('_P')
_T = TypeVar('_T')

# Shared bounded executor so a hung daemon or repeated timeouts don't leak one thread per
# blocking docker call. The default `asyncio.to_thread` executor is unbounded and grows under
# timeout; a fixed pool caps the worst case.
_DOCKER_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix='harness-docker')


# Sentinel exit codes returned by our internal pre-check shell scripts. Picked above 64 to
# stay clear of POSIX's reserved range (0 success, 1 generic error, 2 usage, 126 not-executable,
# 127 command-not-found, 128+N signal-N). When the script body has its own exit code we also
# care about (cat, ls, rg), it shares the same numeric space -- the shell pre-checks run before
# the body and exit early on mismatch, so the body's exit only surfaces on the happy path.
_EXIT_NOT_FOUND = 71
_EXIT_IS_DIRECTORY = 72
_EXIT_PERMISSION_DENIED = 73
_EXIT_NOT_A_DIRECTORY = 74

# POSIX-mandated exit code from `sh` / `docker exec` when the requested binary is not found.
# Used to detect "rg not installed in image" without parsing stderr text.
_EXIT_COMMAND_NOT_FOUND = 127


async def _run_blocking(fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """`asyncio.to_thread` equivalent that uses the module-level docker executor.

    `ParamSpec` preserves `fn`'s signature so overloaded SDK methods (e.g. `exec_start`)
    bind to the correct overload and the return type carries through unchanged.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_DOCKER_EXECUTOR, lambda: fn(*args, **kwargs))


@dataclass(kw_only=True)
class DockerEnvironment(AbstractEnvironment):
    """Docker-backed environment: lifecycle plus all tool methods run inside the container.

    Two modes determined at construction:

    - **Owned** (`image=...`): we create the container in `setup` and remove it in
      `teardown`. The container's lifetime is bound to the env's start/stop cycle.
    - **Attach** (`container=...`): we use a container someone else created. `setup` opens
      an SDK client and runs `mkdir -p root` + the tool probe against the user's container;
      `teardown` closes the SDK client. The harness never starts or stops a container it
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

    Reaping caveat: owned mode runs the container with an init (`init=True`) so killed
    descendants are reaped. Attach mode can't change the user's PID 1 -- if it isn't an
    init (e.g. `sleep infinity`), processes killed on `shell_command` timeout linger as
    zombies. Attach to a container started with `--init` to avoid this.
    """

    root: str = '/workspace'
    """Container path used as the agent's working root.

    Populated by the agent via `write_file`; we deliberately do not bind-mount the host
    cwd, which is brittle on macOS Docker Desktop and unsafe for remote daemons.
    """

    environment: dict[str, str] | None = None
    """Environment variables to set in the container.

    Forwarded to `containers.run(environment=...)`. `None` defers to docker-py's default
    (inherit from the image's `ENV`). Pass an empty dict to start with no extras.
    """

    user: str | None = None
    """User to run the container as.

    Forwarded to `containers.run(user=...)`. Accepts `'name'`, `'uid'`, or `'uid:gid'`.
    `None` uses the image's `USER` directive (often root). Setting a non-root user may
    make writes outside `root` fail; the harness does not chown `root` for you.
    """

    volumes: dict[str, dict[str, str]] | None = None
    """Bind/volume mounts in docker-py's documented shape.

    `{host_path_or_volume_name: {'bind': container_path, 'mode': 'rw' | 'ro'}}`. Forwarded
    to `containers.run(volumes=...)`; we inherit docker-py's schema rather than invent one.
    Malformed values raise at `setup` from docker-py itself.
    """

    startup_timeout: float = 10.0
    """Absolute timeout for `setup` (`containers.run`), in seconds."""

    teardown_timeout: float = 5.0
    """Absolute timeout for `teardown` (`container.remove(force=True)`), in seconds."""

    _container_id: str | None = field(init=False, default=None)
    """The container id we own (or that was attached); empty when no container is bound."""

    _client: DockerClient | None = field(init=False, default=None)
    """The Docker SDK client; created in `setup`, closed in `teardown` (both modes)."""

    _session_id: str = field(init=False, default_factory=lambda: uuid.uuid4().hex)
    """Per-instance UUID stamped on every owned container as a label so `prune_orphans`
    can find leaked containers after a crash. Generated at construction so the value is
    stable across `setup`/`teardown` cycles on the same instance."""

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
        """Owned mode: `containers.run` with labels + user kwargs.

        Attach mode: bind the SDK client to the user-supplied container. Both modes then `mkdir -p root` and run
        the required-tools probe so tool methods can assume a working environment.
        """
        client = await _run_blocking(docker.from_env)
        self._client = client
        # Everything past this point can fail after we've already acquired a resource: in owned
        # mode `containers.run` may succeed and then `mkdir`/the tool probe fail (non-root `user`
        # can't create `root`, image missing a required tool). `start()` only flips `_started`
        # once `setup()` returns, so a later `stop()` would no-op and leak the running container
        # plus the open client. Tear down whatever we built before propagating. `teardown()`
        # already does the mode-correct thing: remove the owned container + close the client, or
        # (attach mode) close the client only without touching the user's container.
        try:
            if self.image is not None:
                labels = {
                    _SESSION_LABEL: self._session_id,
                    _CREATED_LABEL: datetime.now(timezone.utc).isoformat(),
                }
                try:
                    # `containers.run(detach=True)` returns a Container as soon as the container is
                    # created. We bind `_container_id` from its `.id` before any other await -- the
                    # acquire-then-protect line.
                    container = await asyncio.wait_for(
                        _run_blocking(
                            client.containers.run,
                            self.image,
                            ['sleep', 'infinity'],
                            detach=True,
                            # `init=True` runs tini as PID 1, which reaps orphaned zombies. Our
                            # `sleep infinity` PID 1 never calls `wait()`, so a timed-out command
                            # that backgrounds a child (`sleep 30 & wait`) leaves the killed child
                            # as an unreaped zombie that accumulates over the container's lifetime.
                            init=True,
                            environment=self.environment,
                            user=self.user,
                            volumes=self.volumes,
                            labels=labels,
                        ),
                        timeout=self.startup_timeout,
                    )
                except docker.errors.ImageNotFound as exc:
                    raise RuntimeError(f'docker image not found: {exc.explanation}') from exc
                except docker.errors.APIError as exc:
                    raise RuntimeError(f'docker run failed: {exc.explanation}') from exc
                self._container_id = container.id

            # Both modes: ensure `root` exists and required tools are present. Attach mode runs
            # these against a user-owned container -- `mkdir -p` is a no-op if the dir already
            # exists, and the probe gives a clear failure if the user attached an image we can't drive.
            mkdir_exit, _o, mkdir_err = await self._exec(['mkdir', '-p', '--', self.root])
            if mkdir_exit != 0:
                raise EnvSetupError(
                    f'failed to create environment root {self.root!r} in container: '
                    f'{mkdir_err.decode(errors="replace").strip()}'
                )

            probe = (
                'for t in '
                + ' '.join(_REQUIRED_TOOLS)
                + '; do command -v "$t" >/dev/null 2>&1 || { printf %s "$t"; exit 1; }; done'
            )
            probe_exit, probe_out, _e = await self._exec(['sh', '-c', probe])
            if probe_exit != 0:
                missing = probe_out.decode(errors='replace').strip() or '<unknown>'
                raise EnvSetupError(
                    f'container is missing required tool {missing!r}; DockerEnvironment requires {_REQUIRED_TOOLS} in PATH'
                )
        except BaseException:
            # Best-effort cleanup; never let a teardown error mask the original setup failure.
            with contextlib.suppress(Exception):
                await self.teardown()
            raise

    async def teardown(self) -> None:
        """Close the SDK client, removing the container too in owned mode.

        Owned mode runs `container.remove(force=True)` then closes the client. Attach mode
        closes the client only; the container is user-owned and must outlive us. Swallows
        only `NotFound` on remove (the idempotent already-gone case). Every other SDK
        error propagates so the caller can retry or escalate; the base class keeps
        `_started=True` on raise, so the resource is not silently forgotten.
        """
        client = self._client
        if client is None:  # pragma: no cover -- setup never completed
            return
        try:
            if self.image is not None:
                container_id = self._container_id
                if container_id is None:  # pragma: no cover -- setup never bound an id
                    return
                try:
                    container = await _run_blocking(client.containers.get, container_id)
                    await asyncio.wait_for(
                        _run_blocking(container.remove, force=True),
                        timeout=self.teardown_timeout,
                    )
                except docker.errors.NotFound:
                    # Container is already gone: the state we wanted. Idempotent.
                    pass
                self._container_id = None
        finally:
            await _run_blocking(client.close)
            self._client = None

    def _resolve(self, path: str) -> str:
        """Resolve `path` against `root` and reject anything that escapes.

        `PurePosixPath` because container paths are POSIX regardless of host. `posixpath.normpath`
        collapses `..` and `.` syntactically -- without it, `is_relative_to` is purely lexical
        and `/workspace/..` is "inside" `/workspace`. Same advisory jail caveat as
        LocalEnvironment: jail is bypassable via the shell tool, symlinks, or a TOCTOU race --
        the container is the real boundary, the jail just keeps the model from accidentally
        addressing absolute paths.
        """
        normalized = posixpath.normpath(str(PurePosixPath(self.root, path)))
        resolved = PurePosixPath(normalized)
        root = PurePosixPath(self.root)
        if not resolved.is_relative_to(root):
            raise PathEscapeError(f'{path!r} resolves outside the environment root {self.root!r}')
        return str(resolved)

    async def _exec(self, argv: list[str], stdin: bytes | None = None) -> tuple[int, bytes, bytes]:
        """Run `argv` in the container, optionally piping `stdin`, return (exit, stdout, stderr).

        When `stdin` is provided the protocol is `exec_create(stdin=True)`, `exec_start(socket=True)`,
        write bytes onto the underlying raw socket, half-close with `shutdown(SHUT_WR)` to signal
        EOF, drain, then `exec_inspect` for the exit code. The `_sock` private-attribute reach is
        the only path docker-py offers for half-close; the pyright ignores annotate stub gaps,
        not unsafe access.
        """
        client = self._client
        container_id = self._container_id
        if client is None or container_id is None:
            raise RuntimeError('environment not started; call start() first')

        exec_create = client.api.exec_create  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        try:
            exec_info = await _run_blocking(
                exec_create,  # pyright: ignore[reportUnknownArgumentType]
                container_id,
                cmd=argv,
                stdin=stdin is not None,
            )
        except docker.errors.APIError as exc:
            raise EnvShellExecutionError(f'docker exec_create failed: {exc.explanation}') from exc
        exec_id = exec_info['Id']

        if stdin is None:
            stdout_bytes, stderr_bytes = await _run_blocking(client.api.exec_start, exec_id, demux=True)
        else:
            stdout_bytes, stderr_bytes = await self._exec_with_stdin(client, exec_id, stdin)

        info = await _run_blocking(client.api.exec_inspect, exec_id)
        exit_code = info['ExitCode']
        if exit_code is None:
            raise EnvShellExecutionError(f'docker reported no exit code for internal exec {exec_id}')
        return exit_code, stdout_bytes or b'', stderr_bytes or b''

    async def _exec_with_stdin(self, client: DockerClient, exec_id: str, data: bytes) -> tuple[bytes, bytes]:
        """Pipe `data` into a started exec via the half-close pattern, return drained output.

        `exec_start(socket=True)` returns docker-py's `SocketIO` wrapper exposing a `_sock`
        private attr -- the only way to call `shutdown(SHUT_WR)`, which is required to signal
        EOF so `cat > path` flushes and exits. After EOF we drain the demuxed stream until the
        socket closes; for `cat > path` the drain is normally empty (errors go to stderr and
        are surfaced via the `exec_inspect` exit code).
        """
        sock = await _run_blocking(client.api.exec_start, exec_id, socket=True, demux=True)
        raw = sock._sock  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        drained: list[bytes] = []
        try:
            # A pre-check in the target script can exit before it reaches `cat`, closing the
            # remote read end; for a large `data` the `sendall` (or a mid-drain `recv`) then
            # raises a broken-pipe/reset socket error. That's not a transport failure to surface
            # raw -- the exec's non-zero exit code, which the caller reads via `exec_inspect`,
            # already encodes the real cause (e.g. target is a directory). Suppress the socket
            # error and fall through so that clean signal wins.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                await _run_blocking(raw.sendall, data)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                await _run_blocking(raw.shutdown, socket.SHUT_WR)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                while True:
                    chunk: bytes = await _run_blocking(raw.recv, 65536)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                    if not chunk:
                        break
                    drained.append(chunk)
        finally:
            sock.close()  # pyright: ignore[reportUnknownMemberType]
        # `cat > path` doesn't emit on stdout; anything in `drained` is the multiplexed stream
        # which we'd parse for stderr only if we needed to distinguish. The exit code from
        # `exec_inspect` already tells us success/failure; we surface the drained bytes as stderr
        # for the error path's diagnostic text.
        return b'', b''.join(drained)

    async def read_file(self, path: str) -> bytes:
        """Read `path` via `cat` inside the container; pre-validate so errors map cleanly.

        A small inline sh script pre-checks existence/type/readability and exits with sentinel
        codes so we can raise the right ABC exception without scraping stderr text.
        """
        # Why `cat`, not `get_archive`: get_archive mis-behaves on volume-driver-mounted dirs.
        # Reference: openai/openai-agents-python `docker.py:663-666`.
        resolved = self._resolve(path)
        # The pre-check exits with sentinel codes (see module-top constants) so we map errors
        # without scraping stderr text. `cat`'s own exit is 0 for success or 1 for I/O error;
        # the pre-checks run before cat so the common cases become structured signals.
        # Parent-component check first: if any immediate parent exists as a non-directory the
        # path is fundamentally addressing through-a-file -- the ABC's NotADirectory case.
        # (Deep-nesting `a/b/c/x` where `a` is a file collapses to NotFound today; expand to
        # an ancestor walk if a test ever needs it.)
        script = (
            f'[ -e "$(dirname -- "$1")" ] && [ ! -d "$(dirname -- "$1")" ] && exit {_EXIT_NOT_A_DIRECTORY};'
            f' [ -e "$1" ] || exit {_EXIT_NOT_FOUND};'
            f' [ -d "$1" ] && exit {_EXIT_IS_DIRECTORY};'
            f' [ -r "$1" ] || exit {_EXIT_PERMISSION_DENIED};'
            ' cat -- "$1"'
        )
        exit_code, stdout, stderr = await self._exec(['/bin/sh', '-c', script, 'sh', resolved])
        if exit_code == 0:
            return stdout
        if exit_code == _EXIT_NOT_FOUND:
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}')
        if exit_code == _EXIT_IS_DIRECTORY:
            raise EnvIsADirectoryError(f'{path!r} is a directory in the environment root {self.root!r}')
        if exit_code == _EXIT_PERMISSION_DENIED:
            raise EnvPermissionError(f'{path!r} is not readable by the environment root {self.root!r}')
        if exit_code == _EXIT_NOT_A_DIRECTORY:
            raise EnvNotADirectoryError(
                f'{path!r} contains a non-directory component in the environment root {self.root!r}'
            )
        raise EnvReadError(
            f'{path!r} could not be read in the environment root {self.root!r}: {stderr.decode(errors="replace").strip()}'
        )

    async def write_file(self, path: str, data: bytes) -> None:
        """Write `data` to `path` via a streamed `cat > path` exec.

        A pre-check script encodes the write-specific cases (target is a directory, parent not
        writable) into sentinel exit codes so we don't have to parse stderr text.
        """
        # Why streamed stdin instead of `put_archive`: put_archive re-triggers mount-setup on
        # volume-driver-plugin mounts and is a known bug class.
        # References: openai/openai-agents-python `docker.py:663-666` (the workaround),
        # openai-agents-python issue #3093 (symlink-in-tar safety, related).
        resolved = self._resolve(path)
        parent = str(PurePosixPath(resolved).parent)

        # Create the parent dir; any mkdir failure becomes EnvWriteError. Write semantics treat
        # a non-dir component as a write failure rather than reporting NotADirectory separately.
        mkdir_exit, _mkdir_out, mkdir_err = await self._exec(['mkdir', '-p', '--', parent])
        if mkdir_exit != 0:
            raise EnvWriteError(
                f'{path!r} could not be written in the environment root {self.root!r}: '
                f'mkdir {parent!r} failed: {mkdir_err.decode(errors="replace").strip()}'
            )

        # Pre-check encodes write-specific cases into sentinel exit codes (see module-top
        # constants). cat's own non-zero exit means the redirect failed at runtime; `set -e`
        # so a failed redirect surfaces non-zero rather than silently truncating.
        script = (
            f'[ -d "$1" ] && exit {_EXIT_IS_DIRECTORY};'
            f' [ -w "$(dirname -- "$1")" ] || exit {_EXIT_PERMISSION_DENIED};'
            ' set -e; cat > "$1"'
        )
        exit_code, _stdout, stderr = await self._exec(
            ['/bin/sh', '-c', script, 'sh', resolved],
            stdin=data,
        )
        if exit_code == 0:
            return
        if exit_code == _EXIT_IS_DIRECTORY:
            raise EnvWriteError(f'{path!r} is an existing directory in the environment root {self.root!r}')
        if exit_code == _EXIT_PERMISSION_DENIED:
            raise EnvPermissionError(f'{path!r} is not writable by the environment root {self.root!r}')
        raise EnvWriteError(
            f'{path!r} could not be written in the environment root {self.root!r}: '
            f'{stderr.decode(errors="replace").strip()}'
        )

    async def ls(self, path: str) -> list[AbstractFile]:
        """List directory contents via `ls -1Ap -- <path>` and read the directory suffix.

        `-A` includes dotfiles (excludes `.`/`..`); `-1` one entry per line; `-p` appends a
        trailing `/` to directories and nothing else. `/` is the one byte that can never appear
        in a filename, so the suffix is unambiguous -- unlike `-F`, whose `@*=|` indicators
        collide with real filenames ending in those characters (a regular file named `data@`
        would be misread). `-p` does not append `/` to a symlink that points at a directory
        (verified on GNU coreutils and busybox), so symlinks classify as themselves and match
        LocalEnvironment's `is_dir(follow_symlinks=False)` semantics exactly.
        """
        resolved = self._resolve(path)
        # Pre-check before `ls` so failures are structured signals (see module-top constants)
        # rather than parsed from `ls`'s stderr text, which differs across coreutils/busybox.
        script = (
            f'[ -e "$1" ] || exit {_EXIT_NOT_FOUND};'
            f' [ -d "$1" ] || exit {_EXIT_NOT_A_DIRECTORY};'
            f' [ -r "$1" ] || exit {_EXIT_PERMISSION_DENIED};'
            ' ls -1Ap -- "$1"'
        )
        exit_code, stdout, stderr = await self._exec(['/bin/sh', '-c', script, 'sh', resolved])
        if exit_code == _EXIT_NOT_FOUND:
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}')
        if exit_code == _EXIT_PERMISSION_DENIED:
            raise EnvPermissionError(f'{path!r} is not listable by the environment root {self.root!r}')
        if exit_code == _EXIT_NOT_A_DIRECTORY:
            raise EnvNotADirectoryError(f'{path!r} is not a directory in the environment root {self.root!r}')
        if exit_code != 0:
            raise EnvReadError(
                f'{path!r} could not be listed in the environment root {self.root!r}: '
                f'{stderr.decode(errors="replace").strip()}'
            )

        entries: list[AbstractFile] = []
        for raw_line in stdout.splitlines():
            line = raw_line.decode()
            if not line:
                continue
            # `-p` only ever appends `/`, and only to directories; a trailing `/` is the sole
            # indicator and can't be part of a real filename. Everything else (files, symlinks,
            # executables, sockets, FIFOs) is is_directory=False with its name kept verbatim.
            if line.endswith('/'):
                entries.append(AbstractFile(name=line[:-1], is_directory=True))
            else:
                entries.append(AbstractFile(name=line, is_directory=False))
        return entries

    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:
        """Search `path` for `pattern` by invoking `rg` inside the container.

        Requires `rg` (ripgrep) to be installed in the container image. Exit 127 from the exec
        indicates the binary is missing; vendoring `rg` automatically into the container is
        tracked separately. Argv mirrors the host helper so the dialect contract is the same
        engine on both backends: `--regexp=` keeps the pattern inside one argv element,
        `--no-config`/`--one-file-system` defend against hostile env / mount escapes, `--` ends
        flags so a path starting with `-` cannot be parsed as one.
        """
        target = self._resolve(path)
        # Pre-check existence so a missing path raises NotFound rather than getting buried in
        # rg's stderr / exit 2 (which we'd otherwise have to scrape).
        probe_exit, _o, _e = await self._exec(['/bin/sh', '-c', f'[ -e "$1" ] || exit {_EXIT_NOT_FOUND}', 'sh', target])
        if probe_exit == _EXIT_NOT_FOUND:
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}')
        argv = [
            'rg',
            '--json',
            '--no-config',
            '--one-file-system',
            f'--regexp={pattern}',
            '--',
            target,
        ]
        exit_code, stdout, stderr = await self._exec(argv)
        if exit_code == _EXIT_COMMAND_NOT_FOUND:
            raise EnvShellExecutionError(
                'ripgrep (`rg`) is not installed in the container image; install it (e.g. '
                '`apt-get install -y ripgrep`) or use an image that already provides it'
            )
        # Same exit-code discrimination as the host helper: empty stdout + exit 2 means the
        # pattern did not compile (model-fixable); exit 2 with output means rg walked but hit
        # a per-file error which we surface as the parsed partial result. Exit 1 is "no
        # matches" -- normal empty result.
        if exit_code == RG_EXIT_USAGE_OR_PATTERN and not stdout:
            raise EnvInvalidPatternError(f'invalid regex {pattern!r}: {stderr.decode(errors="replace").strip()}')
        return parse_ripgrep_json(stdout, PurePosixPath(self.root))

    async def glob(self, path: str, pattern: str) -> list[str]:
        """List files under `path` matching glob `pattern` via `rg --files -g` in the container.

        Single engine (rg's `globset` crate) across host and container backends -- the dialect
        contract verified by the conformance suite is engine-level, not OS-level. Dotfile
        policy follows rg: hidden directories are not descended into; top-level dotfile leaves
        matched by the pattern are returned.
        """
        target = self._resolve(path)
        # Pre-check existence + type: missing path -> NotFound, file path -> NotADirectory.
        # Mirrors the ls pre-check; same sentinels.
        probe_script = f'[ -e "$1" ] || exit {_EXIT_NOT_FOUND}; [ -d "$1" ] || exit {_EXIT_NOT_A_DIRECTORY}'
        probe_exit, _o, _e = await self._exec(['/bin/sh', '-c', probe_script, 'sh', target])
        if probe_exit == _EXIT_NOT_FOUND:
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}')
        if probe_exit == _EXIT_NOT_A_DIRECTORY:
            raise EnvNotADirectoryError(f'{path!r} is not a directory in the environment root {self.root!r}')
        argv = [
            'rg',
            '--files',
            '--no-ignore',
            '--no-config',
            '--one-file-system',
            f'--glob={pattern}',
            '--',
            target,
        ]
        exit_code, stdout, stderr = await self._exec(argv)
        if exit_code == _EXIT_COMMAND_NOT_FOUND:
            raise EnvShellExecutionError(
                'ripgrep (`rg`) is not installed in the container image; install it (e.g. '
                '`apt-get install -y ripgrep`) or use an image that already provides it'
            )
        if exit_code == RG_EXIT_USAGE_OR_PATTERN:
            msg = stderr.decode(errors='replace').strip()
            if not stdout:
                raise EnvInvalidPatternError(f'invalid glob {pattern!r}: {msg}')
            raise EnvReadError(f'ripgrep failed for {path!r} in the environment root {self.root!r}: {msg}')
        root = PurePosixPath(self.root)
        return [str(PurePosixPath(line.decode()).relative_to(root)) for line in stdout.splitlines() if line]

    async def shell_command(self, command: str, timeout: float | None = None) -> ShellCommandResult:
        """Run `command` in the container and return stdout, stderr, and the exit code.

        Cancellation/timeout strategy: a launcher records its own PID into a per-exec pidfile
        before `exec`-replacing itself with the user's shell, so the recorded PID *is* the user's
        process. On timeout/cancel a second `docker exec` kills that PID *and all its descendants*
        (a `/proc`-based tree walk -- see `_kill_remote_exec`) inside the container, so a command
        that backgrounds children can't leave them running. The daemon performs the signal in the
        container's PID namespace, which works on Linux, Docker Desktop, Colima, Podman-VM, and
        remote daemons alike. Host-side `os.killpg` is unreachable across the VM boundary on macOS,
        so we never use it.
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
            exec_info = await _run_blocking(
                exec_create,  # pyright: ignore[reportUnknownArgumentType]
                container_id,
                cmd=['/bin/sh', '-c', launcher],
                workdir=self.root,
            )
        except docker.errors.APIError as exc:
            raise EnvShellExecutionError(f'docker exec_create failed: {exc.explanation}') from exc
        exec_id = exec_info['Id']

        exec_start = client.api.exec_start
        exec_task = asyncio.create_task(_run_blocking(exec_start, exec_id, demux=True))

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(asyncio.shield(exec_task), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill_remote_exec(client, container_id, pidfile)
            # No secondary timeout on this await: the kill delivers SIGKILL, which is uncatchable,
            # so once it lands the remote process dies and `exec_start` returns. The only way this
            # blocks is if the kill exec itself never ran (daemon paused/down) -- but that same
            # failure would already have wedged this exec's I/O, so a wrapper timeout would just
            # trade one hang for an orphaned task with no result to return.
            stdout_bytes, stderr_bytes = await exec_task
            timed_out = True
        except asyncio.CancelledError:
            await self._kill_remote_exec(client, container_id, pidfile)
            with contextlib.suppress(BaseException):
                await exec_task
            raise

        info = await _run_blocking(client.api.exec_inspect, exec_id)
        exit_code = info['ExitCode']
        if exit_code is None:
            # `ExitCode is None` means the daemon hasn't recorded the exit yet -- the stream
            # closed but the exec is still flagged Running. Treat as transport failure rather
            # than papering over it with a 0 or -1; the caller has no reliable result to use.
            raise EnvShellExecutionError(
                f'docker reported no exit code for exec {exec_id}; daemon state is inconsistent'
            )
        return ShellCommandResult(
            stdout=stdout_bytes or b'',
            stderr=stderr_bytes or b'',
            return_code=exit_code,
            timed_out=timed_out,
        )

    async def _kill_remote_exec(self, client: DockerClient, container_id: str, pidfile: str) -> None:
        """SIGKILL the recorded launcher PID *and all its descendants* via a second exec.

        The contract promises a timeout/cancel kills the whole process tree, not just the launcher
        shell -- a command like `sleep 30 & wait`, or a test runner that forks workers, must not
        leave children alive in the container. We can't `os.killpg` from the host (the container is
        a separate PID namespace, behind a VM on macOS), so we walk the tree *inside* the container
        using only POSIX `sh` and `/proc/<pid>/task/<tid>/children` (kernel >= 3.5, present in every
        container) -- no `ps`/`pkill`/`setsid` dependency. Children are collected *before* the
        parent is killed (post-order) so reparenting to PID 1 can't orphan them.

        Tolerant by design: a missing pidfile or an already-exited PID collapses to a no-op, and a
        cleanup error must never mask the original timeout or cancellation that triggered this.
        """
        kill_tree = (
            'kill_tree() { '
            'for c in $(cat /proc/"$1"/task/*/children 2>/dev/null); do kill_tree "$c"; done; '
            'kill -KILL "$1" 2>/dev/null; '
            '}; '
            f'p=$(cat {pidfile} 2>/dev/null || echo 0); [ "$p" = 0 ] || kill_tree "$p"; true'
        )
        exec_create = client.api.exec_create  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        with contextlib.suppress(docker.errors.APIError):
            kill_info = await _run_blocking(exec_create, container_id, cmd=['/bin/sh', '-c', kill_tree])  # pyright: ignore[reportUnknownArgumentType]
            await _run_blocking(client.api.exec_start, kill_info['Id'])
