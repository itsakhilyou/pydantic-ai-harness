"""Skeleton Docker-backed execution environment.

This is a structural skeleton, not a working backend. It exists so the
`ExecutionEnv` capability can wire `environment='docker'` and accept a
`DockerEnvironment(...)` instance for configuration, validating that the
public API design holds before Slice 5 implements the real backend.

Every method raises `NotImplementedError`; instantiate at your own risk.
See `agent_docs/docker-prior-art.md` for the planned implementation
(docker CLI via `create_subprocess_exec`, `docker cp` for byte-accurate
file transfer, in-container `timeout -k` for tree-kill).
"""

from dataclasses import dataclass

from .abstract import AbstractEnvironment, AbstractFile, AbstractMatch, ShellCommandResult


@dataclass(kw_only=True)
class DockerEnvironment(AbstractEnvironment):
    """Docker-backed environment -- SKELETON, not yet implemented (Slice 5).

    Exists to dogfood the `ExecutionEnv` API: lets callers write
    `ExecutionEnv(environment='docker')` for the zero-config default and
    `ExecutionEnv(environment=DockerEnvironment(image='python:3.12-slim', ...))`
    when they want to configure the backend. Every tool call raises
    `NotImplementedError` until Slice 5.
    """

    root: str = '/workspace'
    """Container path used as the agent's working root. Defaults to `/workspace`."""

    image: str = 'python:3.12-slim'
    """Docker image to run. Configuration knob exists to prove the API; not used yet."""

    def _not_implemented(self) -> NotImplementedError:
        return NotImplementedError(
            'DockerEnvironment is a skeleton; the real backend lands in Slice 5. See agent_docs/docker-prior-art.md.'
        )

    async def read_file(self, path: str) -> bytes:
        raise self._not_implemented()

    async def write_file(self, path: str, data: bytes) -> None:
        raise self._not_implemented()

    async def ls(self, path: str) -> list[AbstractFile]:
        raise self._not_implemented()

    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:
        raise self._not_implemented()

    async def glob(self, path: str, pattern: str) -> list[str]:
        raise self._not_implemented()

    async def shell_command(self, command: str, timeout: float | None = None) -> ShellCommandResult:
        raise self._not_implemented()
