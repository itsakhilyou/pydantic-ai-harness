# AbstractEnvironment contract decisions

Pinned the previously-unspecified behaviors of `ls`, `write_file`, `glob`, `AbstractMatch.line`,
and `root` before adding a second backend. Source of the concerns: the audit in
`execution-env-strategy.md` ("The Docker reckoning: conformance as a contract-discovery
instrument") — categories of representation/error choices that were Local-isms we'd chosen
without noticing. Two filters guided every call: **minimum Docker coercion** and **doesn't make
the model walk weird edges**. Where they conflicted, framework consensus broke the tie.

## (a) `ls` includes dotfiles

| Source | Behavior |
|---|---|
| `os.scandir` (Local) | include |
| `find -mindepth 1 -maxdepth 1` (Docker) | include |
| Bill's `FileSystem` (PR #260) | include |
| Claude Code `LS` | include |
| OpenHands `file_editor.view` | include |
| E2B `filesystem.list` | include |

Zero coercion on either backend; universal across frameworks; symmetric with `read_file`.

## (b) `write_file` creates intermediate directories

| Source | Behavior |
|---|---|
| `Path.write_bytes` (Local default) | no — raises |
| Claude Code `Write` | mkdir-parents |
| OpenHands `str_replace_editor create` | mkdir-parents |
| Aider new-file edits | mkdir-parents |
| E2B `filesystem.write` | mkdir-parents |
| Bill's `FileSystem.write_file` | mkdir-parents |
| Cursor agent file ops | mkdir-parents |

Unanimous framework signal. Without it: model writes `pkg/new/__init__.py` → error → needs
`make_dir` (which we don't ship as a tool) → retries. Three turns to do one thing. Docker cost is
trivial (`mkdir -p $(dirname X) && tee X`, or directory entries in the tar upload path).

## (c) `glob` excludes dotfiles for `*`

| Source | Behavior |
|---|---|
| `Path.rglob('*.py')` (Local) | exclude |
| `find -name '*.py'` (Docker default) | include |
| ripgrep / `fd` | exclude unless `--hidden` |
| Claude Code `Glob` | exclude |
| Bill's `FileSystem.glob` | gitignore-aware (excludes `.git/**`) |

Tied on native cost (Docker pays a `-not -path '*/.*'` filter; Local is free). Broken by
consistency with our own `grep` (already ripgrep, already excludes dotfiles) and by the
semantics models are trained on (`fd`, ripgrep, editor "find files"). Asymmetry with `ls` is
intentional: `ls` = "what is here," `glob` = "find code-like files."

## (d) `AbstractMatch.line` has no trailing newline

ripgrep's JSON output keeps the trailing `\n` from the file; the backend strips it. One
canonical form; conformance compares text by equality; future backends don't have to replicate
ripgrep's framing.

## (e) `ls` classifies symlink entries by the entry, not the target

`is_dir(follow_symlinks=False)` in Local; `find -type l` / equivalent in Docker. A symlink to a
directory has `is_directory=False`. Avoids tool-recursion cycles through symlink loops; matches
Bill's FileSystem and Claude Code.

## (f) `root` is the canonical resolved absolute path

`environment.root` equals what `shell_command('pwd')` reports.

- Local: `Path(root).resolve()` in `__post_init__` (macOS `/var/...` → `/private/var/...`,
  symlinked roots collapse to their target).
- Docker: the in-container WORKDIR the backend configured (`/workspace`).

Conformance asserts against `environment.root`, never against `tmp_path.resolve()`. Host-side
seeding via `tmp_path.write_*` still works for Docker because the backend bind-mounts
`tmp_path` into the container.

## What this doc is for

When a future backend (Docker, E2B, remote) implements `AbstractEnvironment`, this is the
checklist of decisions the conformance suite encodes. Each row above is now a pinned test in
`tests/environments/test_conformance.py`; divergence from any of them is a red CI signal, not a
user-reported drift.
