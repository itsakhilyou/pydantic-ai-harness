"""Lifecycle behavior on `AbstractEnvironment`: idempotent start/stop and `async with`.

These tests cover the base class's contract -- the gate, the `_started` flag, the
context-manager adapter -- not any backend's resource. A tiny `_CountingEnvironment`
subclass counts `setup`/`teardown` invocations so we can prove the gate runs them
at most once per start/stop cycle. `LocalEnvironment` is exercised separately to prove
the default no-op implementations work for a backend that holds no resource.
"""

from dataclasses import dataclass

import pytest

from pydantic_ai_harness.environments.abstract import (
    AbstractEnvironment,
    AbstractFile,
    AbstractMatch,
    ShellCommandResult,
)
from pydantic_ai_harness.environments.local import LocalEnvironment


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass(kw_only=True)
class _CountingEnvironment(AbstractEnvironment):
    """Records how many times `setup` / `teardown` actually run.

    The abstract methods are stubs because lifecycle is the only thing under test -- the
    ABC requires them but they're never called.
    """

    setup_calls: int = 0
    teardown_calls: int = 0

    async def setup(self) -> None:
        self.setup_calls += 1

    async def teardown(self) -> None:
        self.teardown_calls += 1

    async def read_file(self, path: str) -> bytes:  # pragma: no cover
        return b''

    async def write_file(self, path: str, data: bytes) -> None:  # pragma: no cover
        return None

    async def ls(self, path: str) -> list[AbstractFile]:  # pragma: no cover
        return []

    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:  # pragma: no cover
        return []

    async def glob(self, path: str, pattern: str) -> list[str]:  # pragma: no cover
        return []

    async def shell_command(self, command: str, timeout: float | None = None) -> ShellCommandResult:  # pragma: no cover
        return ShellCommandResult(stdout=b'', stderr=b'', return_code=0, timed_out=False)


async def test_start_runs_setup_once_and_flips_flag() -> None:
    env = _CountingEnvironment(root='/x')
    assert env._started is False  # pyright: ignore[reportPrivateUsage]

    await env.start()
    assert env.setup_calls == 1
    assert env._started is True  # pyright: ignore[reportPrivateUsage]


async def test_start_is_idempotent_second_call_is_noop() -> None:
    """Two consecutive starts must run `setup` exactly once -- this is what lets an outer
    `async with` and an inner `wrap_run` both call `start` without double-allocating."""
    env = _CountingEnvironment(root='/x')
    await env.start()
    await env.start()
    assert env.setup_calls == 1


async def test_stop_is_noop_when_not_started() -> None:
    """Calling stop on a never-started env must not invoke `teardown` -- the inner scope of an
    attach-mode env (`_started=True` pre-seeded but no setup ran) relies on this asymmetry."""
    env = _CountingEnvironment(root='/x')
    await env.stop()
    assert env.teardown_calls == 0


async def test_stop_runs_teardown_once_and_flips_flag() -> None:
    env = _CountingEnvironment(root='/x')
    await env.start()
    await env.stop()
    assert env.teardown_calls == 1
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_stop_is_idempotent_second_call_is_noop() -> None:
    env = _CountingEnvironment(root='/x')
    await env.start()
    await env.stop()
    await env.stop()
    assert env.teardown_calls == 1


async def test_start_leaves_flag_false_when_setup_raises() -> None:
    """A failed allocation must not mark the env as started -- otherwise stop() would try to
    tear down a resource that never existed."""

    class _Boom(Exception):
        pass

    @dataclass(kw_only=True)
    class _FailingSetup(_CountingEnvironment):
        async def setup(self) -> None:
            raise _Boom

    env = _FailingSetup(root='/x')
    with pytest.raises(_Boom):
        await env.start()
    assert env._started is False  # pyright: ignore[reportPrivateUsage]


async def test_stop_leaves_flag_true_when_teardown_raises() -> None:
    """A failed teardown must not silently mark the env stopped -- the caller is told the
    resource may still exist and a retry of stop() will be attempted (not skipped)."""

    class _Boom(Exception):
        pass

    @dataclass(kw_only=True)
    class _FailingTeardown(_CountingEnvironment):
        async def teardown(self) -> None:
            raise _Boom

    env = _FailingTeardown(root='/x')
    await env.start()
    with pytest.raises(_Boom):
        await env.stop()
    assert env._started is True  # pyright: ignore[reportPrivateUsage]


async def test_aenter_aexit_pair_with_start_stop() -> None:
    """`async with env:` calls start on entry, stop on exit, and yields the env itself."""
    env = _CountingEnvironment(root='/x')
    async with env as bound:
        assert bound is env
        assert env.setup_calls == 1
        assert env.teardown_calls == 0
    assert env.teardown_calls == 1


async def test_aexit_runs_stop_on_exception() -> None:
    """If the `async with` body raises, `__aexit__` must still run -- container/process cleanup
    is the entire reason we chose context-manager + wrap_run over before_run/after_run."""

    class _Boom(Exception):
        pass

    env = _CountingEnvironment(root='/x')
    with pytest.raises(_Boom):
        async with env:
            raise _Boom
    assert env.teardown_calls == 1


async def test_local_environment_lifecycle_is_no_op(tmp_path: object) -> None:
    """LocalEnvironment holds no resource; start/stop inherit the base no-op `setup`/`teardown`.
    Proven by the flag flipping correctly across a context-manager round trip on a real
    LocalEnvironment instance -- the same path users hit when they do `async with env:`."""
    env = LocalEnvironment(root=str(tmp_path))
    async with env:
        assert env._started is True  # pyright: ignore[reportPrivateUsage]
    assert env._started is False  # pyright: ignore[reportPrivateUsage]
