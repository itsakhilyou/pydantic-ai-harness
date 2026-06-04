# Coding Harness

A coding agent assembled entirely from `pydantic_ai_harness` capabilities. This
is the integration target for the capability library: the capabilities are the
product, and this harness is the glue that turns them into an agent that can work
in a real repository.

## What it composes

| Capability | Role in the harness | Source |
|---|---|---|
| `FileSystem` | read, search, and edit files under a repo root | merged |
| `Shell` | run commands and tests, with a persistent cwd | merged |
| `Planning` | keep a cache-stable task plan via `write_plan` | #266 |
| `SubAgents` | delegate review/research to focused sub-agents | #267 |

The agent rooted at a directory exposes: `read_file`, `write_file`, `edit_file`,
`create_directory`, `list_directory`, `find_files`, `search_files`, `file_info`,
`run_command`, `check_command`, `start_command`, `stop_command`, `write_plan`,
and `delegate_task`.

## Run it against a repo (needs a model API key)

```bash
PYTHONPATH=examples python -m coding_harness "Fix the failing test in calculator.py" --root /path/to/repo
```

## Run the offline end-to-end demo (no API key)

```bash
PYTHONPATH=examples python -m coding_harness.demo
```

The demo creates a scratch repo with an off-by-one bug in `factorial` and a test
that catches it, then drives the harness with a scripted model. The scripted
model only chooses which tools to call; every file read, edit, and test run is
executed by the real capabilities against the repo on disk. It asserts that the
file was actually edited and that `pytest` goes from red to green.

## Status

This is an experimental integration branch used to exercise the capabilities
while the underlying PRs are in review. It is not meant to merge into `main` as
is. The roadmap is to fold in the recent maintainer capabilities (CodeMode
host-FS access, compaction, tool-orphan-repair) and use the gaps it surfaces to
prioritize the next capabilities.
