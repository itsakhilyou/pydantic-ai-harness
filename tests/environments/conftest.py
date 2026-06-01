"""Shared fixtures for environment backend tests.

The `environment` fixture is parametrized over every backend that implements
`AbstractEnvironment`. Conformance tests in `test_conformance.py` use it to verify the
ABC contract end-to-end on each backend.

Conformance tests seed files via `env.write_file()` (the contract-defined entry point)
rather than touching the host filesystem directly -- that's what lets the same test body
run unchanged against `LocalEnvironment` (host FS), `DockerEnvironment` (container FS),
and any future backend. The `seed_file` / `seed_dir` helpers below are the only setup
primitives tests need.

The `docker` param skips when no daemon is reachable so contributors without Docker
installed still get a green local-only run; CI is expected to provide a daemon.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import docker
import docker.errors
import pytest

from pydantic_ai_harness.environments.abstract import AbstractEnvironment
from pydantic_ai_harness.environments.docker import DockerEnvironment
from pydantic_ai_harness.environments.local import LocalEnvironment


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _docker_available() -> bool:
    """`True` iff a Docker daemon is reachable. Used to skip the docker param locally."""
    try:
        docker.from_env().ping()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except (docker.errors.DockerException, OSError):  # pragma: no cover -- only hit without a daemon; CI provides one
        return False
    return True


@pytest.fixture(params=['local', 'docker'])
async def environment(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[AbstractEnvironment]:
    """A started backend, parametrized over every environment implementation.

    `local` roots at `tmp_path`; `docker` runs `python:3.12-slim` with `rg` installed at
    fixture startup. The docker param skips when no daemon is reachable.
    """
    if request.param == 'local':
        env: AbstractEnvironment = LocalEnvironment(root=str(tmp_path))
        async with env:
            yield env
    elif request.param == 'docker':
        if not _docker_available():  # pragma: no cover -- only hit without a daemon; CI provides one
            pytest.skip('no Docker daemon')
        docker_env = DockerEnvironment(image='python:3.12-slim')
        async with docker_env:
            # `python:3.12-slim` doesn't ship `rg`; install before yielding so grep/glob
            # conformance can run. Vendoring rg into the container is tracked separately;
            # this is the interim path.
            rg_install = await docker_env.shell_command('apt-get update -qq && apt-get install -y -qq ripgrep')
            assert rg_install.return_code == 0, f'apt-get install ripgrep failed: {rg_install.stderr!r}'
            yield docker_env
    else:  # pragma: no cover -- unreachable per `params`
        raise AssertionError(f'unknown environment backend {request.param!r}')


async def seed_file(env: AbstractEnvironment, relpath: str, content: bytes) -> None:
    """Write `content` to `relpath` via the env's own `write_file` contract.

    Replaces the older host-FS pattern `(tmp_path / 'name').write_bytes(...)` with one
    that's backend-agnostic: the conformance suite seeds through the contract instead of
    around it, so the same setup works on any backend without bind-mounting host paths.
    """
    await env.write_file(relpath, content)


async def seed_dir(env: AbstractEnvironment, relpath: str) -> None:
    """Create an empty directory at `relpath` via `env.write_file`'s parent-creation behavior.

    Writes a hidden marker (`.keep`) inside so the directory exists without the env needing
    a separate mkdir primitive. The marker is invisible to top-level `ls` and irrelevant to
    `grep` (empty file). For tests that seed content inside a directory, write that content
    directly -- `write_file` creates intermediate dirs.
    """
    await env.write_file(f'{relpath}/.keep', b'')
