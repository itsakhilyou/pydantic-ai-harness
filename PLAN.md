# FileSystem and Shell Capabilities

Closes #25 and #26.

## Overview

Two `AbstractCapability` subclasses providing file system access and shell execution as composable agent capabilities.

## FileSystem (`src/pydantic_harness/filesystem.py`)

**Tools provided via `get_toolset()`:**
- `read_file(path, offset, limit)` -- reads a text file with numbered lines
- `write_file(path, content)` -- creates or overwrites a file
- `edit_file(path, old_text, new_text, replace_all)` -- exact string replacement editing
- `list_directory(path)` -- directory listing with type/size indicators
- `search_files(pattern, path)` -- regex search across files

**Configuration:**
- `root_dir` -- all paths resolved relative to this, traversal prevented
- `allowed_patterns` -- glob allowlist (if non-empty, only matching paths accessible)
- `denied_patterns` -- glob denylist (matching paths always rejected)
- `max_read_lines` -- per-read line limit (default 2000)

**Security:** Path traversal above `root_dir` is rejected. Hidden files skipped in search. Binary files skipped in search.

## Shell (`src/pydantic_harness/shell.py`)

**Tool provided via `get_toolset()`:**
- `run_command(command, timeout_seconds)` -- execute a shell command

**Configuration:**
- `cwd` -- working directory for commands
- `allowed_commands` -- executable allowlist (mutually exclusive with deny)
- `denied_commands` -- executable denylist
- `default_timeout` -- seconds (default 30)
- `max_output_chars` -- output truncation limit (default 10000)

**Implementation:** Uses `anyio.open_process` for async-backend-agnostic subprocess execution (works with both asyncio and trio).

## Design decisions

1. **Public methods for tool implementations** -- `read_file()`, `write_file()`, etc. are public methods on the capability class, registered with the toolset via `FunctionToolset.add_function()`. This allows direct testing and reuse by subclasses or future Environment abstraction (#52).

2. **No `RunContext` dependency** -- tool implementations are synchronous (FileSystem) or standalone async (Shell), following `get_toolset()` semantics where the toolset is created at agent construction time.

3. **anyio for Shell** -- uses `anyio.open_process` instead of `asyncio.create_subprocess_shell` so the capability works under both asyncio and trio event loops.

4. **pydantic-ai-slim from main** -- the harness `pyproject.toml` sources `pydantic-ai-slim` from the pydantic-ai `main` branch (which includes the capabilities module). This will be updated to a release version once capabilities ship.

## Future considerations

- Integration with the Environment abstraction (#52) -- FileSystem and Shell could become thin wrappers over an `Environment` protocol
- `get_instructions()` -- could provide context-aware system prompt additions
- `for_run()` -- per-run state isolation for sandboxed environments
