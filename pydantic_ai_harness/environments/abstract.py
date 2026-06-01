"""Abstract base class for all execution environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Self, final


@dataclass(kw_only=True, frozen=True)
class AbstractFile:
    """A file in the environment."""

    name: str
    """The file's name."""

    is_directory: bool
    """Whether the file is a directory."""


@dataclass(kw_only=True, frozen=True)
class AbstractMatch:
    """A line in a file."""

    path: str
    """The path to the file."""

    line: str
    """The line's text, without a trailing newline."""

    lineno: int
    """The line's number."""


@dataclass(kw_only=True, frozen=True)
class ShellCommandResult:
    """The result of a shell command."""

    stdout: bytes
    """The command's stdout."""

    stderr: bytes
    """The command's stderr."""

    return_code: int
    """The command's return code."""

    timed_out: bool
    """Whether the command timed out."""


@dataclass(kw_only=True)
class AbstractEnvironment(ABC):
    """Abstract base class for all execution environments."""

    root: str
    """The environment's canonical absolute root. Equal to what `shell_command('pwd')` reports."""

    _started: bool = False
    """Whether the environment is currently started."""

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Return a file's raw, undecoded bytes.

        Args:
            path: File path, resolved against and confined to `root`.

        Returns:
            The file's raw bytes. Decoding to text is the caller's concern.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvNotFoundError: No file exists at `path`.
            EnvIsADirectoryError: `path` is a directory, not a file.
            EnvNotADirectoryError: A component of `path` is not a directory.
            EnvPermissionError: The backend may not read `path`.
            EnvReadError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def write_file(self, path: str, data: bytes) -> None:
        """Create or overwrite a file with raw bytes. Missing intermediate directories are created.

        Args:
            path: File path, resolved against and confined to `root`.
            data: Raw bytes to write.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvPermissionError: The backend may not write `path` (or create parents).
            EnvWriteError: Any other I/O failure (e.g. writing onto an existing directory).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def ls(self, path: str) -> list[AbstractFile]:
        """List the contents of a directory.

        Includes dotfiles. Symlinks classified by entry (a
        symlink to a directory has `is_directory=False`).

        Args:
            path: Directory path, resolved against and confined to `root`.

        Returns:
            A list of files and directories in the directory.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvNotFoundError: No directory exists at `path`.
            EnvNotADirectoryError: A component of `path` is not a directory.
            EnvPermissionError: The backend may not list `path`.
            EnvReadError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:
        """Search a file or directory tree for a regex pattern.

        `pattern` is a ripgrep-dialect regex (Rust `regex` crate). Every backend uses the
        same engine, so dialect is unambiguous and identical across backends -- there is no
        "portable subset" to commit to or police. Patterns ripgrep accepts are valid;
        patterns it rejects raise `EnvInvalidPatternError`.

        Args:
            path: File path, resolved against and confined to `root`.
            pattern: Regex string (ripgrep dialect).

        Returns:
            A list of matches.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvNotFoundError: No file exists at `path`.
            EnvIsADirectoryError: `path` is a directory, not a file.
            EnvNotADirectoryError: A component of `path` is not a directory.
            EnvPermissionError: The backend may not grep `path`.
            EnvInvalidPatternError: `pattern` is malformed regex (model-fixable).
            EnvReadError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def glob(self, path: str, pattern: str) -> list[str]:
        """Find files under a directory matching a glob pattern.

        Supported pattern syntax is a backend-independent subset the conformance suite enforces:
        `*` (any run of non-separator chars), `?` (one char), `[seq]` (char class), and `**` for
        recursion. A bare pattern like `*.py` matches at **any depth**. Backends that shell out must
        constrain `find`/native globbing to match these semantics rather than expose their own dialect;
        for anything more, the model uses `shell`.

        Dotfile policy follows the underlying engine (ripgrep's `globset`): hidden directories
        are not descended into (`**/*.py` does not enter `.git/`), but a hidden file matched by
        the pattern at the top level IS returned (`*.py` does match `.hidden.py`). To exclude
        top-level dotfiles, narrow the pattern (e.g. `[!.]*.py`).

        Args:
            path: Directory path, resolved against and confined to `root`.
            pattern: Glob pattern (subset above), matched recursively at any depth.

        Returns:
            A list of matching file paths, relative to `root`.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvNotFoundError: No file or directory exists at `path`.
            EnvNotADirectoryError: `path` is a file, not a directory.
            EnvPermissionError: The backend may not read `path`.
            EnvReadError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def shell_command(self, command: str, timeout: float | None = None) -> ShellCommandResult:
        """Run `command` in a shell and return its captured output and exit code.

        The command is shell-interpreted (pipes, `&&`, globs all work) and runs in a fresh process; no
        state (cwd, env, vars) persists between calls. A non-zero exit is **not** an error -- it
        returns a result with that `return_code`; a timeout returns a result with `timed_out=True`.
        Neither raises. Backends must not silently make execution stateful.

        Args:
            command: The shell command to run.
            timeout: Seconds before the process tree is killed and the result returned with
                `timed_out=True`. `None` means no timeout.

        Returns:
            A `ShellCommandResult` for any command that ran, whatever its exit code.

        Raises:
            EnvShellExecutionError: The environment could not start a shell at all (none available, or
                the spawn failed). Not raised for a non-zero exit or a timeout.
        """
        raise NotImplementedError  # pragma: no cover

    async def setup(self) -> None:
        """Allocate backend resources. Override in backends that hold a resource; default is a no-op."""

    async def teardown(self) -> None:
        """Release backend resources. Override in backends that hold a resource; default is a no-op."""

    @final
    async def start(self) -> None:
        """Start the environment. Idempotent: a second call while already started is a no-op.

        `@final`: subclasses must override `setup`, not `start` -- this method owns the idempotency
        gate and the `_started` flag, and overriding it would skip both.
        """
        if self._started:
            return
        # `_started` is flipped only after `setup` returns: a failed allocation leaves the env in
        # the not-started state so the caller gets a clean signal and stop() won't try to tear down
        # something that was never built.
        await self.setup()
        self._started = True

    @final
    async def stop(self) -> None:
        """Stop the environment. Idempotent: calling while not started is a no-op.

        `@final`: subclasses must override `teardown`, not `stop`.
        """
        if not self._started:
            return
        # `_started` stays True if `teardown` raises -- callers are told the resource may still
        # exist rather than silently losing track of it; a retry can call stop() again safely.
        await self.teardown()
        self._started = False

    async def __aenter__(self) -> Self:
        """Start the environment and return self for `async with` use."""
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Stop the environment when the `async with` block exits, on success or exception."""
        await self.stop()
