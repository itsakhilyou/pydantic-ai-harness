# code_puppy learnings for DockerEnvironment

Source: https://github.com/mpfaffenberger/code_puppy (cloned at HEAD, June 2026).
Cross-cutting: code_puppy is a Pydantic-AI-based coding agent that runs **everything on the host**. There is no Docker mode, no chroot, no namespace, no resource limit. Isolation is one regex-based "destructive command" guard plus a per-call user-approval prompt. Treat this repo as a deep well of subprocess/streaming/Windows trivia, not as a sandboxing reference.

Status legend (matches `agent_docs/openai-agents-docker-learnings.md`):
- ✅ steal — port the idea directly
- 🛠 adapt — good shape, needs rework for our contract
- 📋 deferred — interesting, out of scope for current slice
- 🚫 don't copy — explicit anti-pattern, documented so we don't repeat it

---

## 1. Execution environment / sandboxing

**There is no sandbox.** Every tool call (`run_shell_command`, `write_to_file`, `_read_file`) runs as the host user, in the host cwd, with the host's env, against the host's FS. The "safety" story is:

1. A regex prefilter for destructive shell patterns (`rm -rf /`, `git reset --hard`, `Format-Volume`, etc.) in `code_puppy/plugins/destructive_command_guard/detector.py:30-80`.
2. An interactive approval prompt before every non-yolo shell command (`code_puppy/tools/command_runner.py:1145-1198`).
3. A `yolo_mode` config flag that disables (2). No (1) bypass either way at this layer; (1) lives in the plugin hook chain.

🚫 **Don't copy as "sandboxing".** This is a UX speedbump, not isolation. A model that has `write_to_file` can edit `~/.ssh/authorized_keys` without ever invoking a shell — there is zero path confinement (see §2). Our `ExecutionEnv` boundary needs to be the *only* path to the FS/process table, not one of many.

🛠 **Cross-platform forking pattern is worth stealing for `LocalEnvironment`** (`command_runner.py:1264-1285`):
```py
if sys.platform.startswith("win"):
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
else:
    preexec_fn = os.setsid if hasattr(os, "setsid") else None
process = subprocess.Popen(command, shell=True, ..., preexec_fn=preexec_fn, creationflags=creationflags)
```
Same two-branch pattern shows up for background/detached procs at `command_runner.py:1053-1073` using `start_new_session=True` on POSIX. This is the canonical "give me a killable process tree" recipe and matches our `shell-run-prior-art.md` decision.

📋 **Background processes return a `pid` + temp log file path** (`command_runner.py:1043-1105`). Detached via `start_new_session` / `CREATE_NEW_PROCESS_GROUP`, output redirected to `tempfile.NamedTemporaryFile(delete=False)`, agent reads the log later. Decent UX shape if we ever do a long-running-process slice — but note the temp file is never cleaned up (`delete=False`, no atexit), which is a leak we should fix on adoption.

---

## 2. File I/O primitives

### read_file — `code_puppy/tools/file_operations.py:461-557`

✅ **Good: surrogateescape + post-sanitize for bad UTF-8.** They open with `errors="surrogateescape"` (`file_operations.py:479`), then re-encode/decode with `surrogatepass`→`replace` (`:504-513`) to ensure no lone surrogates leak into JSON-serialised tool output. Worth stealing for our text-mode read path; the JSON-serialisation foot-gun is real (Pydantic AI tool returns get re-serialised).

🛠 **Token-cap as content cap.** `num_tokens > 10000` returns an error refusing to read (`:517-522`). Crude (`len // 4` heuristic), but the *idea* of "reject before serialising into context" is sound. We should size-cap in **bytes** at the env boundary, not in approximate tokens at the capability boundary.

🚫 **No bytes mode at all.** Everything is text. Binary files come back as mojibake-with-replacement-chars. This is exactly the scope mistake my prior-art memory warned about — they locked in `str` at the public contract and now there is no way to read a PNG without going through shell + base64. Our locked-in `bytes` is right.

🚫 **No path safety.** `file_path = os.path.abspath(os.path.expanduser(file_path))` (`:467`) and that is the entire check. No symlink resolution, no `commonpath` against a root, no rejection of `..`. `grep -rn "is_symlink\|realpath\|commonpath" code_puppy/tools/*.py` returns zero hits. A model can `read_file("/etc/shadow")` or `write_to_file("../../../../etc/cron.d/x")` and it Just Works. **This is the single biggest reason `ExecutionEnv` exists** — we have to confine at the env, because they prove the capability layer can't be trusted to.

### write_to_file — `code_puppy/tools/file_modifications.py:356-415`

🛠 **`overwrite=False` default with "cowardly refusing".** (`:367-374`). Good default for an LLM-driven agent. We should default `WriteFile` to no-clobber too, with an explicit `overwrite=True`.

🛠 **Parent-dir auto-creation.** `os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)` (`:400`). Pragmatic. The `or "."` guard is a small but real detail — `dirname("foo.txt")` is `""` and `os.makedirs("")` raises.

🚫 **No atomic write.** `open(file_path, "w") ; f.write(content)` (`:401-402`). A crash mid-write truncates the file. The proper pattern is write-to-temp-in-same-dir then `os.replace` (atomic on POSIX, mostly-atomic on Windows). For our `WriteFile` we should do tempfile-and-rename; their approach is a known data-loss bug class.

🚫 **Diff is generated by reading the old file again then writing the new one with no locking.** TOCTOU between the read at `:379` and write at `:401`. Not a security issue for them since there is no sandbox to defeat, but if we ever add concurrent agents this is the kind of race that bites.

🚫 **Encoding asymmetry.** Reads with `errors="surrogateescape"`, writes with default strict encoding (`:401`). A file containing invalid UTF-8 round-tripped through read→write will raise `UnicodeEncodeError` mid-write, leaving a partial file on disk.

---

## 3. Shell command execution — `code_puppy/tools/command_runner.py`

This is their best file. 1411 lines, mostly painful subprocess plumbing learned the hard way.

✅ **Aggressive process-group kill ladder** (`:141-205`):
- POSIX: `SIGTERM` (wait 1.0s) → `SIGINT` (wait 0.6s) → `SIGKILL` (wait 0.5s) → loop `os.kill SIGKILL` x3.
- Windows: `taskkill /F /T /PID` (tree kill via `/T`) → fallback `proc.kill()`.

Steal the *ladder shape*. The exact timeouts are arbitrary; our `shell-run-prior-art.md` decision (cancellation must kill remote) maps cleanly to this. One subtle bit: they `os.killpg(os.getpgid(pid), ...)` rather than caching the pgid at spawn — fine because they spawn with `setsid`, but worth a comment when we port.

✅ **Track running processes in a module-level `Set[Popen]`** (`:109, 131-139, 208-241`) so `Ctrl-X` / shutdown can kill every live child. Useful for our env's `aclose()` to guarantee no leaked children.

🛠 **Dual timeout: absolute + inactivity.** `run_shell_command_streaming` tracks `last_output_time[0]` and kills if `now - last_output_time > timeout` (`:921-925`). That is an *inactivity* timeout, not wall-clock. Their `timeout=60` arg actually means "60s of silence". Useful for "build is still printing" UX, but bad as a default contract: a `sleep 600 && echo done` survives forever as long as it's the only command in the pipeline. We should default to wall-clock and offer inactivity as an opt-in.

🚫 **Three-thread streaming with manual `select` / `PeekNamedPipe`** (`:704-839`). Two reader threads (stdout, stderr) + main thread doing the timeout/poll loop. Reasons this is bad:
1. Threads block on `readline()` and have to be unstuck by closing the pipe from the cleanup path (`:225-233`) — a hack their own comments acknowledge.
2. The Windows path busy-waits with `time.sleep(0.1)` because `select` doesn't work on Windows pipes (`:716-748`). They call `PeekNamedPipe` via `ctypes` (`:46-93`). Cute but fragile.
3. `stop_event` only gets checked between `select` rounds, so threads can lag the kill by 100ms+.
4. The whole sync function is then wrapped in `loop.run_in_executor(_SHELL_EXECUTOR, ...)` (`:1316-1319`) — so they pay thread-pool cost *and* internal-thread cost per command.

We already decided to use `asyncio.create_subprocess_exec` + async stream readers. That sidesteps every issue above. Their file is a great reference for *what platform-specific bugs to expect* once we start running on Windows — keep this section bookmarked.

🛠 **Line-length truncation per line + last-N lines on overflow** (`:36-43, 894-895`): `MAX_LINE_LENGTH = 256`, then `"\n".join(stdout_lines[-256:])`. Useful pattern for keeping pathological outputs (e.g. `find /`) from blowing the context. Worth doing in our env's output capture, not the capability layer.

🚫 **`shell=True` with raw user-supplied string** (`:1057, 1067, 1276-1285`). Means every command is interpreted by `/bin/sh -c` or `cmd.exe`. Mostly unavoidable for "run this command" UX, but combined with §1's lack of sandboxing it means the LLM has unrestricted host shell. Our env should accept `list[str]` argv as the primitive and treat shell-string as a wrapper, not the reverse.

---

## 4. Grep / glob

### grep — `file_operations.py:586-705`

✅ **Ripgrep with `--json`.** Parse line-by-line, filter `type == "match"` events. This is the right engine choice — fast, gitignore-aware, JSON output is stable. Worth stealing as our default grep backend when `rg` is available.

✅ **Ripgrep discovery: PATH then `sys.executable`'s `bin/`/`Scripts/` dir** (`:613-623`). Catches the common case where `rg` is installed as a Python package's binary (e.g. via `pip install ripgrep`). Cheap and worth copying.

🛠 **Temp ignore-file built from `DIR_IGNORE_PATTERNS`** (`:644-652`). Reasonable, but they leak the temp file on most error paths (only cleaned up in a `finally` they don't show in the excerpt I read — verify before copying). Prefer writing to a pre-created temp dir scoped to the env's lifetime.

🚫 **`shlex.split(search_string)` to "support flags like --ignore-case"** (`:653-659`). This is the bug from issue #259: on Windows, `shlex` POSIX-mode swallows backslashes, so `\bdef\b` and `C:\Users\me` get mangled before reaching ripgrep, and ripgrep's non-zero return is silently treated as "no matches". Fixed in PR #339 by using `posix=False`. **Lesson: never `shlex.split` model-supplied query strings, and always surface non-(0|1) return codes from `rg`.**

🚫 **No glob tool at all.** `grep -n "glob\|fnmatch" code_puppy/tools/*.py` shows `fnmatch` is used only for *ignore-pattern matching* against directory paths (`file_operations.py:125-145`), not as a user-facing tool. Their "list files" is `rg --files` (`file_operations.py:201-226`). That's a valid choice — `rg --files` + filter is more useful than POSIX `glob.glob` for an LLM — but means they have no way to answer "find me `**/*.py` excluding tests" without shelling out. Our glob tool should probably also be `rg --files` + a pattern filter rather than `pathlib.Path.glob`, which is shockingly slow on large trees.

---

## 5. Known bug classes from their issues/PRs

Quote-form, one-line each:

- **#259 / PR #339** — `shlex.split` POSIX-mode mangles backslashes in grep queries; ripgrep non-zero returns swallowed. *Lesson: argv construction must be platform-aware; always surface tool exit codes.*
- **#324** — `/cd` fails on Windows when target path contains `.` (regex/path-parsing bug). *Lesson: Windows paths break naive parsers.*
- **#88** — Frequent tool-call timeouts in v0.250, root cause never pinned down; suspected inactivity-timeout-not-resetting. *Lesson: dual timeout semantics (inactivity vs wall-clock) are confusing and need clear contract.*
- **#199** — `_run_with_streaming_retry` didn't catch `RemoteProtocolError` / `ReadTimeout`. *Lesson: streaming HTTP retry has more failure modes than non-streaming.*
- **#288** — MessageQueue thread `join(timeout=1.0)` plus no `atexit` handler leaves zombie threads on exit. *Lesson: any background thread/process needs a registered shutdown path, not just a best-effort join.*
- **#244** — Cbreak-mode stdin reader for Ctrl-X discards bytes it doesn't recognise; mouse escape sequences fragmented across reads leak as garbage into next prompt. *Lesson: don't read raw TTY bytes in a side thread while another component (prompt_toolkit) also reads stdin — partial escape sequences become impossible to reconstruct.*
- **#222** — Session history not saved when agent errors or is cancelled. *Lesson: cancellation paths need explicit "save partial state" semantics; can't rely on the happy-path finally.*

---

## 6. Hard nos

Things code_puppy does that **we explicitly should not copy**:

1. **"Sandbox = regex + approval prompt."** The destructive-command detector and the per-command approval are a UX layer, not a security layer. Anything that lives outside `shell_run` (write_file, edit_file, read_file) bypasses both entirely. Our `ExecutionEnv` must be the only path to the host; capability-layer guards are advisory at best.

2. **No path confinement of any kind.** `os.path.abspath` is the entire path-safety story. Zero symlink resolution, zero root-jail check. We are doing this *because* of how easy it is to skip — confinement belongs in the env, with `realpath` + `commonpath`-against-root, and reject-on-violation.

3. **`shell=True` as the only spawning mode.** Forces every command through `/bin/sh -c`, makes argv-mode impossible. Our primitive should be argv-list; shell-string is a documented wrapper.

4. **Thread-pool wrapping a sync subprocess that internally spawns two reader threads.** `command_runner.py:1276-1319` is "blocking I/O wrapped in `run_in_executor` wrapped around threading.Thread readers". `asyncio.create_subprocess_exec` with `await proc.stdout.readline()` is one layer, not three, and cancels cleanly.

5. **Non-atomic writes.** `open("w"); f.write()`. We tempfile-in-same-dir then `os.replace`.

6. **Text-only file API.** `errors="surrogateescape"` is a Band-Aid for a contract that should have been bytes. We already chose bytes — don't backslide.

7. **Inactivity timeout as the default `timeout` arg.** Confusing semantics, hard to predict, was the suspected cause of #88. Default is wall-clock; inactivity is an opt-in flag.

8. **`shlex.split` of model-supplied strings.** #259. If we need to support flags-in-pattern, parse them ourselves with explicit, platform-neutral rules; don't outsource argv-splitting to a POSIX-by-default helper.

9. **Module-global mutable sets and threading primitives for process tracking** (`_RUNNING_PROCESSES`, `_AWAITING_USER_INPUT`, `_KEYBOARD_CONTEXT_REFCOUNT`, `_ACTIVE_STOP_EVENTS`...). Cute for a single-shot CLI, terrible for a library where users compose multiple envs. Track running processes *on the env instance*, with an async lock.

10. **Temp files with `delete=False` and no atexit/finally cleanup** (background-process log files at `command_runner.py:1043-1075`; grep ignore-file at `:644-652`). Always tie temp artifacts to an explicit owner with a lifecycle hook.

---

## Files to bookmark

- `code_puppy/tools/command_runner.py:141-205` — process-group kill ladder (✅ shape)
- `code_puppy/tools/command_runner.py:1264-1285` — POSIX/Windows spawn flags (✅ shape)
- `code_puppy/tools/command_runner.py:1041-1105` — detached background process (📋 future slice)
- `code_puppy/tools/file_operations.py:461-557` — text read w/ surrogateescape sanitisation (🛠 partial steal for text-mode read)
- `code_puppy/tools/file_operations.py:586-705` — ripgrep `--json` grep (✅ engine choice)
- `code_puppy/plugins/destructive_command_guard/detector.py` — regex command classifier (📋 maybe useful as an advisory capability much later; not as a security boundary)
