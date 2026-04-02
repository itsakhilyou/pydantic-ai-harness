"""Shell capability: gives agents configurable command execution.

Provides a ``run_command`` tool with timeout support, output truncation,
and optional command allow/deny lists.
"""

from __future__ import annotations

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

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait. Defaults to ``default_timeout``.

        Returns:
            Combined stdout/stderr, with exit code appended on non-zero exit.
        """
        self.check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout

        proc = await anyio.open_process(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self._cwd,
        )
        try:
            assert proc.stdout is not None
            chunks: list[bytes] = []
            with anyio.fail_after(timeout):
                async for chunk in proc.stdout:
                    chunks.append(chunk)
                await proc.wait()
        except TimeoutError:
            proc.kill()
            with anyio.CancelScope(shield=True):
                await proc.wait()
            return f'[Command timed out after {timeout} seconds]'

        output = b''.join(chunks).decode('utf-8', errors='replace')
        output = self.truncate(output)
        exit_code = proc.returncode if proc.returncode is not None else 0

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
