"""FileSystem capability: gives agents configurable file system access.

Provides tools for reading, writing, editing, listing, and searching files,
all scoped to a configurable root directory with path filtering.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset


def format_lines(text: str, offset: int, limit: int) -> str:
    """Format text with line numbers, similar to ``cat -n``.

    Args:
        text: The raw file content.
        offset: Zero-based line offset to start from.
        limit: Maximum number of lines to include.

    Returns:
        Numbered text with a continuation hint when more lines remain.
    """
    lines = text.splitlines(keepends=True)
    total = len(lines)

    if offset >= total > 0:
        raise ValueError(f'Offset {offset} exceeds file length ({total} lines).')

    selected = lines[offset : offset + limit]
    numbered = [f'{i:>6}\t{line}' for i, line in enumerate(selected, start=offset + 1)]
    result = ''.join(numbered)
    if not result.endswith('\n'):
        result += '\n'

    remaining = total - (offset + len(selected))
    if remaining > 0:
        next_offset = offset + len(selected)
        result += f'... ({remaining} more lines. Use offset={next_offset} to continue reading.)\n'

    return result


@dataclass
class FileSystem(AbstractCapability[Any]):
    """Capability that provides file system access scoped to a root directory.

    All paths supplied by the model are resolved relative to ``root_dir``.
    Traversal above the root is rejected.  Optional allow/deny glob patterns
    restrict which paths may be accessed.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness.filesystem import FileSystem

        agent = Agent('openai:gpt-4o', capabilities=[FileSystem(root_dir='.')])
    """

    root_dir: str | Path = '.'
    """Root directory for all file operations. Defaults to the current directory."""

    allowed_patterns: list[str] = field(default_factory=lambda: list[str]())
    """If non-empty, only paths matching at least one glob pattern are accessible."""

    denied_patterns: list[str] = field(default_factory=lambda: list[str]())
    """Paths matching any of these glob patterns are rejected."""

    max_read_lines: int = 2000
    """Maximum number of lines returned by a single ``read_file`` call."""

    def __post_init__(self) -> None:
        """Resolve the root directory to an absolute path."""
        self._root = Path(self.root_dir).resolve()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def resolve_path(self, path: str) -> Path:
        """Resolve *path* relative to the root, raising on traversal.

        Args:
            path: A relative path within the root directory.

        Returns:
            The resolved absolute path.

        Raises:
            PermissionError: If the resolved path escapes the root.
        """
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise PermissionError(f'Path {path!r} resolves outside the root directory.')
        return resolved

    def check_access(self, path: str) -> None:
        """Raise ``PermissionError`` if *path* is blocked by allow/deny patterns.

        Args:
            path: The relative path to check.
        """
        if self.denied_patterns:
            for pattern in self.denied_patterns:
                if fnmatch.fnmatch(path, pattern):
                    raise PermissionError(f'Path {path!r} is denied by pattern {pattern!r}.')
        if self.allowed_patterns:
            if not any(fnmatch.fnmatch(path, p) for p in self.allowed_patterns):
                raise PermissionError(f'Path {path!r} does not match any allowed pattern.')

    def safe_resolve(self, path: str) -> Path:
        """Resolve and access-check a path in one step.

        Args:
            path: The relative path to resolve and validate.

        Returns:
            The resolved absolute path.
        """
        self.check_access(path)
        return self.resolve_path(path)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return. Defaults to ``max_read_lines``.

        Returns:
            File content with line numbers.
        """
        if limit is None:
            limit = self.max_read_lines
        resolved = self.safe_resolve(path)
        if not resolved.is_file():
            if resolved.is_dir():
                raise FileNotFoundError(f"'{path}' is a directory, not a file.")
            raise FileNotFoundError(f'File not found: {path}')
        text = resolved.read_text(encoding='utf-8')
        return format_lines(text, offset, limit)

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a file.

        Args:
            path: File path relative to the root directory.
            content: The text content to write.

        Returns:
            Confirmation message.
        """
        resolved = self.safe_resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding='utf-8')
        return f'Successfully wrote {len(content)} characters to {path}.'

    def edit_file(self, path: str, old_text: str, new_text: str, *, replace_all: bool = False) -> str:
        """Edit a file by exact string replacement.

        Args:
            path: File path relative to the root directory.
            old_text: The exact text to find.
            new_text: The replacement text.
            replace_all: If True, replace all occurrences.
                Otherwise ``old_text`` must appear exactly once.

        Returns:
            Summary of replacements made.
        """
        resolved = self.safe_resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(f'File not found: {path}')
        text = resolved.read_text(encoding='utf-8')

        count = text.count(old_text)
        if count == 0:
            raise ValueError(f'old_text not found in {path}.')
        if not replace_all and count > 1:
            raise ValueError(
                f'old_text found {count} times in {path}. '
                'Set replace_all=True or provide more surrounding context to make the match unique.'
            )

        new_content = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        resolved.write_text(new_content, encoding='utf-8')
        replacements = count if replace_all else 1
        return f'Replaced {replacements} occurrence(s) in {path}.'

    def list_directory(self, path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators (``/`` for directories).
        """
        resolved = self.safe_resolve(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        entries: list[str] = []
        for entry in sorted(resolved.iterdir()):
            rel = str(entry.relative_to(self._root))
            if entry.is_dir():
                entries.append(f'{rel}/')
            else:
                try:
                    size = entry.stat().st_size
                except OSError:  # pragma: no cover
                    size = 0
                entries.append(f'{rel}  ({size} bytes)')
        return '\n'.join(entries) if entries else '(empty directory)'

    def search_files(self, pattern: str, *, path: str = '.') -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.

        Returns:
            Matching lines formatted as ``file:line_number:text``.
        """
        resolved = self.safe_resolve(path)
        compiled = re.compile(pattern)
        results: list[str] = []

        if resolved.is_file():
            files = [resolved]
        else:
            files = sorted(resolved.rglob('*'))

        for file_path in files:
            if not file_path.is_file():
                continue
            # Skip hidden files/directories
            try:
                rel_parts = file_path.relative_to(self._root).parts
            except ValueError:  # pragma: no cover
                continue
            if any(part.startswith('.') for part in rel_parts):
                continue
            try:
                raw = file_path.read_bytes()
            except OSError:
                continue
            # Skip binary files
            if b'\x00' in raw[:8192]:
                continue
            text = raw.decode('utf-8', errors='replace')
            rel_path = str(file_path.relative_to(self._root))
            for line_num, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    results.append(f'{rel_path}:{line_num}:{line}')
            if len(results) > 1000:
                results.append('[... truncated at 1000 matches]')
                break

        return '\n'.join(results) if results else 'No matches found.'

    # ------------------------------------------------------------------
    # Capability interface
    # ------------------------------------------------------------------

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build and return the toolset containing all file system tools."""
        toolset: FunctionToolset[Any] = FunctionToolset()
        toolset.add_function(self.read_file, name='read_file')
        toolset.add_function(self.write_file, name='write_file')
        toolset.add_function(self.edit_file, name='edit_file')
        toolset.add_function(self.list_directory, name='list_directory')
        toolset.add_function(self.search_files, name='search_files')
        return toolset
