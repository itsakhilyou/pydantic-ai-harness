"""Shell capability: gives agents configurable command execution.

Provides a ``run_command`` tool with timeout support, output truncation,
and optional command allow/deny lists.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset


@dataclass
class Shell(AbstractCapability[Any]):
    """Capability that provides shell command execution.

    Commands are executed in a subprocess rooted at ``cwd``.  An optional
    allow-list (``allowed_commands``) or deny-list (``denied_commands``)
    restricts which executables may be invoked.  Output is truncated to
    ``max_output_chars`` to keep model context manageable.

    When ``persist_cwd`` is ``True``, the shell tracks ``cd`` commands and
    adjusts the working directory for subsequent calls, simulating a
    persistent shell session.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness.shell import Shell

        agent = Agent('openai:gpt-4o', capabilities=[Shell(cwd='.')])
    """

    cwd: str | Path = '.'
    """Working directory for command execution."""

    allowed_commands: list[str] = field(default_factory=lambda: list[str]())
    """If non-empty, only these command names may be executed (allowlist)."""

    denied_commands: list[str] = field(default_factory=lambda: list[str]())
    """These command names are always rejected (denylist)."""

    default_timeout: float = 30.0
    """Default timeout in seconds for command execution."""

    max_output_chars: int = 10_000
    """Maximum characters of output returned to the model."""

    persist_cwd: bool = False
    """If ``True``, track ``cd`` commands and adjust the working directory for subsequent calls."""

    def __post_init__(self) -> None:
        """Resolve the working directory and validate configuration."""
        self._cwd = Path(self.cwd).resolve()
        if self.allowed_commands and self.denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_command(self, command: str) -> None:
        """Validate *command* against allow/deny lists.

        Args:
            command: The shell command string to validate.

        Raises:
            PermissionError: If the command is blocked by the allow/deny lists.
        """
        try:
            tokens = shlex.split(command)
        except ValueError:
            # If shlex can't parse it, fall through and let the shell handle it
            return
        if not tokens:
            return
        executable = tokens[0]

        if self.denied_commands and executable in self.denied_commands:
            raise PermissionError(f'Command {executable!r} is denied.')
        if self.allowed_commands and executable not in self.allowed_commands:
            raise PermissionError(f'Command {executable!r} is not in the allowed list.')

    def truncate(self, text: str) -> str:
        """Truncate *text* to ``max_output_chars``.

        Args:
            text: The text to truncate.

        Returns:
            The original text if within limits, otherwise truncated with a notice.
        """
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + f'\n... [output truncated at {self.max_output_chars} characters]'

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _extract_cd_target(self, command: str) -> str | None:
        """Extract the target directory from a ``cd`` command.

        Returns ``None`` if *command* is not a ``cd`` invocation.
        """
        stripped = command.strip()
        # Match: `cd <path>`, `cd <path> && ...`, `cd <path>;...`
        m = re.match(r'^cd\s+(.+?)(?:\s*[;&|]|$)', stripped)
        if m is None:
            return None
        target = m.group(1).strip()
        # Strip surrounding quotes
        if len(target) >= 2 and target[0] in ('"', "'") and target[-1] == target[0]:
            target = target[1:-1]
        return target

    def _update_cwd(self, command: str) -> None:
        """Update ``_cwd`` if *command* contains a ``cd`` and the target exists."""
        target = self._extract_cd_target(command)
        if target is None:
            return
        if target == '~':
            new_cwd = Path.home()
        elif target.startswith('~'):
            new_cwd = Path.home() / target[2:]  # skip "~/"
        else:
            new_cwd = (self._cwd / target).resolve()
        if new_cwd.is_dir():
            self._cwd = new_cwd

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Execute a shell command and return its output.

        Stdout and stderr are captured separately and labeled in the output.
        When ``persist_cwd`` is enabled, ``cd`` commands update the working
        directory for subsequent calls.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait. Defaults to ``default_timeout``.

        Returns:
            Labeled stdout/stderr output, with exit code appended on non-zero exit.
        """
        self.check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout

        proc = await anyio.open_process(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._cwd,
        )
        try:
            assert proc.stdout is not None
            assert proc.stderr is not None
            stdout_chunks: list[bytes] = []
            stderr_chunks: list[bytes] = []

            async def _read_stdout() -> None:
                assert proc.stdout is not None
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)

            async def _read_stderr() -> None:
                assert proc.stderr is not None
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)

            with anyio.fail_after(timeout):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_read_stdout)
                    tg.start_soon(_read_stderr)
                await proc.wait()
        except TimeoutError:
            proc.kill()
            with anyio.CancelScope(shield=True):
                await proc.wait()
            return f'[Command timed out after {timeout} seconds]'
        finally:
            await proc.aclose()

        stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
        stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')

        parts: list[str] = []
        if stdout:
            parts.append(f'[stdout]\n{stdout}')
        if stderr:
            parts.append(f'[stderr]\n{stderr}')
        output = '\n'.join(parts) if parts else ''

        output = self.truncate(output)
        exit_code = proc.returncode if proc.returncode is not None else 0

        if self.persist_cwd and exit_code == 0:
            self._update_cwd(command)

        if exit_code != 0:
            return f'{output}\n[exit code: {exit_code}]'
        return output

    # ------------------------------------------------------------------
    # Capability interface
    # ------------------------------------------------------------------

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build and return the toolset containing the run_command tool."""
        toolset: FunctionToolset[Any] = FunctionToolset()
        toolset.add_function(self.run_command, name='run_command')
        return toolset
