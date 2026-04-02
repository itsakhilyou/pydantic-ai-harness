"""Tests for the Shell capability."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pydantic_harness.shell import Shell

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cwd(tmp_path: Path) -> Path:
    """Create a temporary working directory."""
    (tmp_path / 'greeting.txt').write_text('hello world\n')
    return tmp_path


@pytest.fixture
def sh(tmp_cwd: Path) -> Shell:
    """A Shell capability rooted at the test directory."""
    return Shell(cwd=tmp_cwd)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        sh = Shell()
        assert sh.default_timeout == 30.0
        assert sh.max_output_chars == 10_000

    def test_cannot_mix_allow_deny(self) -> None:
        with pytest.raises(ValueError, match='not both'):
            Shell(allowed_commands=['ls'], denied_commands=['rm'])


# ---------------------------------------------------------------------------
# Command validation
# ---------------------------------------------------------------------------


class TestCommandValidation:
    def test_denied_command(self) -> None:
        sh = Shell(denied_commands=['rm'])
        with pytest.raises(PermissionError, match='denied'):
            sh.check_command('rm -rf /')

    def test_allowed_command(self) -> None:
        sh = Shell(allowed_commands=['echo', 'cat'])
        sh.check_command('echo hello')  # should not raise
        with pytest.raises(PermissionError, match='not in the allowed'):
            sh.check_command('rm -rf /')

    def test_no_restrictions(self) -> None:
        sh = Shell()
        sh.check_command('anything goes')  # should not raise

    def test_malformed_command(self) -> None:
        sh = Shell(denied_commands=['rm'])
        # Unterminated quote: shlex.split raises ValueError.
        # The capability falls through and lets the shell handle it.
        sh.check_command("echo 'unterminated")  # should not raise

    def test_empty_command(self) -> None:
        sh = Shell(allowed_commands=['echo'])
        sh.check_command('')  # empty string should not raise


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_short_output(self) -> None:
        sh = Shell(max_output_chars=100)
        assert sh.truncate('short') == 'short'

    def test_long_output(self) -> None:
        sh = Shell(max_output_chars=10)
        result = sh.truncate('x' * 50)
        assert len(result.splitlines()[0]) == 10
        assert 'truncated' in result


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


class TestRunCommand:
    @pytest.mark.anyio
    async def test_echo(self, sh: Shell) -> None:
        result = await sh.run_command('echo hello')
        assert 'hello' in result

    @pytest.mark.anyio
    async def test_exit_code(self, sh: Shell) -> None:
        result = await sh.run_command('exit 1')
        assert 'exit code: 1' in result

    @pytest.mark.anyio
    async def test_timeout(self) -> None:
        sh = Shell()
        result = await sh.run_command('sleep 10', timeout_seconds=0.1)
        assert 'timed out' in result.lower()

    @pytest.mark.anyio
    async def test_cwd(self, sh: Shell) -> None:
        result = await sh.run_command('cat greeting.txt')
        assert 'hello world' in result

    @pytest.mark.anyio
    async def test_truncated_output(self, tmp_cwd: Path) -> None:
        sh = Shell(cwd=tmp_cwd, max_output_chars=20)
        result = await sh.run_command(f'{sys.executable} -c "print(\'x\' * 100)"')
        assert 'truncated' in result

    @pytest.mark.anyio
    async def test_denied_command_async(self) -> None:
        sh = Shell(denied_commands=['rm'])
        with pytest.raises(PermissionError, match='denied'):
            await sh.run_command('rm -rf /')

    @pytest.mark.anyio
    async def test_allowed_command_async(self) -> None:
        sh = Shell(allowed_commands=['echo'])
        result = await sh.run_command('echo works')
        assert 'works' in result
        with pytest.raises(PermissionError, match='not in the allowed'):
            await sh.run_command('cat /etc/passwd')


# ---------------------------------------------------------------------------
# Toolset integration
# ---------------------------------------------------------------------------


class TestToolset:
    def test_get_toolset_returns_function_toolset(self, sh: Shell) -> None:
        from pydantic_ai.toolsets import FunctionToolset

        toolset = sh.get_toolset()
        assert isinstance(toolset, FunctionToolset)

    def test_toolset_has_run_command(self, sh: Shell) -> None:
        from pydantic_ai.toolsets import FunctionToolset

        toolset = sh.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        assert set(toolset.tools.keys()) == {'run_command'}

    def test_serialization_name(self) -> None:
        assert Shell.get_serialization_name() == 'Shell'
