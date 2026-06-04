"""Tests for the LocalStackContainer Docker context manager.

These drive the public `async with` lifecycle and observe the `docker` commands
it actually runs; readiness polling goes through a real local HTTP endpoint.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

from pydantic_ai_harness.experimental.localstack import LocalStackContainer, LocalStackError

from ._http_server import HttpResponse, http_server, unused_tcp_port

_AUTH_ENV_NAMES = ('LOCALSTACK_AUTH_TOKEN', 'LOCALSTACK_API_KEY')


@contextmanager
def _localstack_env(values: Mapping[str, str] | None = None) -> Generator[None]:
    saved = {name: os.environ.get(name) for name in _AUTH_ENV_NAMES}
    try:
        for name in _AUTH_ENV_NAMES:
            os.environ.pop(name, None)
        if values is not None:
            os.environ.update(values)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _docker_stub(tmp_path: Path) -> tuple[str, Path]:
    """A fake `docker` CLI that logs every invocation; `run` also prints a container id."""
    log = tmp_path / 'docker.log'
    stub = tmp_path / 'docker'
    stub.write_text(
        '#!/bin/sh\n'
        f'echo "$@" >> {log}\n'
        'if [ "$1" = run ]; then\n'
        f'  printf "env SERVICES=%s LOCALSTACK_AUTH_TOKEN=%s LOCALSTACK_API_KEY=%s\\n" '
        f'"$SERVICES" "$LOCALSTACK_AUTH_TOKEN" "$LOCALSTACK_API_KEY" >> {log}\n'
        '  echo container-abc123\n'
        'fi\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(stub), log


def _failing_docker_stub(tmp_path: Path) -> str:
    """A fake `docker` CLI whose `run` fails with a message on stderr."""
    stub = tmp_path / 'docker-fail'
    stub.write_text('#!/bin/sh\necho "boom: port in use" >&2\nexit 1\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(stub)


class TestProperties:
    def test_endpoint_url(self, tmp_path: Path) -> None:
        docker, _ = _docker_stub(tmp_path)
        assert (
            LocalStackContainer(docker_path=docker, host_port=4599).endpoint_url
            == 'http://localhost.localstack.cloud:4599'
        )

    def test_container_id_is_none_before_start(self, tmp_path: Path) -> None:
        docker, _ = _docker_stub(tmp_path)
        assert LocalStackContainer(docker_path=docker).container_id is None


class TestEnvironmentContext:
    def test_restores_existing_auth_token(self) -> None:
        original_environment = dict(os.environ)
        os.environ['LOCALSTACK_AUTH_TOKEN'] = 'original-token'
        try:
            with _localstack_env({'LOCALSTACK_AUTH_TOKEN': 'temporary-token'}):
                assert os.environ['LOCALSTACK_AUTH_TOKEN'] == 'temporary-token'
            assert os.environ['LOCALSTACK_AUTH_TOKEN'] == 'original-token'
        finally:
            os.environ.clear()
            os.environ.update(original_environment)


class TestLifecycle:
    async def test_starts_waits_then_stops(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)

        with _localstack_env(), http_server([HttpResponse(503), HttpResponse(200)]) as health:
            async with LocalStackContainer(docker_path=docker, host_port=health.port, poll_interval=0.01) as container:
                assert container.container_id == 'container-abc123'
                assert container.endpoint_url == f'http://localhost.localstack.cloud:{health.port}'

        assert health.paths == ['/_localstack/health', '/_localstack/health']
        assert container.container_id is None
        log_text = log.read_text()
        assert f'run -d --rm -p 127.0.0.1:{health.port}:4566 localstack/localstack' in log_text
        assert 'stop container-abc123' in log_text

    async def test_run_command_includes_env_and_name(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with _localstack_env(), http_server([HttpResponse(200)]) as health:
            async with LocalStackContainer(
                docker_path=docker,
                host_port=health.port,
                image='img',
                container_name='ls',
                environment={'SERVICES': 's3'},
                poll_interval=0.01,
            ):
                pass
        log_text = log.read_text()
        assert f'run -d --rm -p 127.0.0.1:{health.port}:4566 -e SERVICES --name ls img' in log_text
        assert 'env SERVICES=s3 LOCALSTACK_AUTH_TOKEN= LOCALSTACK_API_KEY=' in log_text

    async def test_mounts_service_ports_and_docker_socket(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        docker_socket = tmp_path / 'docker.sock'
        docker_socket.touch()

        with _localstack_env(), http_server([HttpResponse(200)]) as health:
            async with LocalStackContainer(
                docker_path=docker,
                host_port=health.port,
                host_address='localhost',
                service_port_range='4510-4559',
                mount_docker_socket=True,
                docker_socket_path=str(docker_socket),
            ):
                pass

        log_text = log.read_text()
        assert 'localhost:4510-4559:4510-4559' in log_text
        assert f'{docker_socket}:/var/run/docker.sock' in log_text

    async def test_missing_docker_socket_fails_before_start(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        missing_socket = tmp_path / 'missing.sock'
        with pytest.raises(LocalStackError, match='Docker socket'):
            async with LocalStackContainer(
                docker_path=docker,
                mount_docker_socket=True,
                docker_socket_path=str(missing_socket),
            ):
                pass  # pragma: no cover
        assert not log.exists()

    async def test_forwards_auth_token_from_environment_without_putting_value_in_argv(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with _localstack_env({'LOCALSTACK_AUTH_TOKEN': 'auth-token'}), http_server([HttpResponse(200)]) as health:
            async with LocalStackContainer(docker_path=docker, host_port=health.port, poll_interval=0.01):
                pass
        argv_line, env_line, *_ = log.read_text().splitlines()
        assert '-e LOCALSTACK_AUTH_TOKEN ' in f'{argv_line} '
        assert 'auth-token' not in argv_line
        assert env_line == 'env SERVICES= LOCALSTACK_AUTH_TOKEN=auth-token LOCALSTACK_API_KEY='

    async def test_forwards_legacy_api_key_when_auth_token_is_absent(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with _localstack_env({'LOCALSTACK_API_KEY': 'legacy-key'}), http_server([HttpResponse(200)]) as health:
            async with LocalStackContainer(docker_path=docker, host_port=health.port, poll_interval=0.01):
                pass
        argv_line, env_line, *_ = log.read_text().splitlines()
        assert '-e LOCALSTACK_API_KEY ' in f'{argv_line} '
        assert 'legacy-key' not in argv_line
        assert env_line == 'env SERVICES= LOCALSTACK_AUTH_TOKEN= LOCALSTACK_API_KEY=legacy-key'

    async def test_explicit_auth_environment_takes_precedence(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with _localstack_env({'LOCALSTACK_AUTH_TOKEN': 'process-token'}), http_server([HttpResponse(200)]) as health:
            async with LocalStackContainer(
                docker_path=docker,
                host_port=health.port,
                environment={'LOCALSTACK_AUTH_TOKEN': 'explicit-token'},
                poll_interval=0.01,
            ):
                pass
        argv_line, env_line, *_ = log.read_text().splitlines()
        assert '-e LOCALSTACK_AUTH_TOKEN ' in f'{argv_line} '
        assert 'process-token' not in argv_line
        assert 'explicit-token' not in argv_line
        assert env_line == 'env SERVICES= LOCALSTACK_AUTH_TOKEN=explicit-token LOCALSTACK_API_KEY='

    async def test_readiness_timeout_stops_container(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        port = unused_tcp_port()
        with pytest.raises(LocalStackError, match='did not become ready within 0.05s'):
            async with LocalStackContainer(
                docker_path=docker, host_port=port, startup_timeout=0.05, poll_interval=0.01
            ):
                pass  # pragma: no cover
        assert 'stop container-abc123' in log.read_text()

    async def test_docker_not_found(self) -> None:
        with pytest.raises(LocalStackError, match='Docker CLI .* not found'):
            async with LocalStackContainer(docker_path='/no/such/docker'):
                pass  # pragma: no cover

    async def test_start_failure_surfaces_stderr(self, tmp_path: Path) -> None:
        with pytest.raises(LocalStackError, match='Failed to start LocalStack container: boom: port in use'):
            async with LocalStackContainer(docker_path=_failing_docker_stub(tmp_path)):
                pass  # pragma: no cover

    async def test_exit_without_start_is_safe(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        await LocalStackContainer(docker_path=docker).__aexit__(None, None, None)
        assert not log.exists()

    async def test_stop_never_raises_if_docker_vanishes(self, tmp_path: Path) -> None:
        # Cleanup must not raise even if the docker binary is gone by exit time.
        container = LocalStackContainer(docker_path='/no/such/docker')
        container._container_id = 'orphan'
        await container.__aexit__(None, None, None)
        assert container.container_id is None
