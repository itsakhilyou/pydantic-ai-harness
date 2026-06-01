# OpenAI Agents Python — Docker Sandbox: Learnings for `DockerEnvironment`

Source: deep-read of `openai/openai-agents-python` `src/agents/sandbox/sandboxes/docker.py`
(1593 lines) plus issue tracker, commit `464043a`. Captured because they have the most
mature `docker-py`-based agent sandbox in the ecosystem and we adopt or reject each
choice consciously.

## Status legend

- ✅ Already applied in our `DockerEnvironment`
- 🛠 Applying now (this slice)
- 📋 Captured for the hardening slice
- 🚫 Hard no — explicitly not copying

## Top 5 lessons (their ranking, our triage)

| # | Lesson | Status |
|---|---|---|
| 1 | Shared bounded `ThreadPoolExecutor`, not `asyncio.to_thread` (their comment: *"so repeated timeouts do not leak one new thread per command"*). `docker.py:81-84`, `538-542`. | 🛠 |
| 2 | Never use `get_archive` / `put_archive` for file I/O when volume-driver mounts are attached. Use `cat` / streamed `cat >` via exec. `docker.py:663-666`, `709-711`. | 📋 |
| 3 | Validate every host-supplied path and every tar member. `PurePosixPath` for in-container paths; reject `..` and absolute-target symlinks; cap archive size + member count *before* writing. Issues #3093, #3169, #3274, #3452. | 📋 |
| 4 | PTY pidfile + second-exec kill: `printf "%s" "$$" > pidfile && exec "$@"`. The `pkill -f` non-PTY fallback is a hack; pidfile everywhere. `docker.py:815-869`, `1170-1196`. | ✅ |
| 5 | `ExitCode is None` is a transport error, not success. `Running == False && ExitCode != None` is the only reliable finished signal. Close BOTH socket AND underlying HTTP response on exec teardown. `docker.py:463-474`, `1029-1033`, `148-156`. | 🛠 |

## 17 themes

### 1. Container lifecycle

**Their approach.** `containers.create(..., detach=True, entrypoint=['tail'], command=['-f', '/dev/null'])` then `container.start()` (`docker.py:1437-1480`, `1454-1460`). Idle entrypoint keeps the container alive; all real work is via `container.exec_run(...)`. `resume(state)` reattaches by `container_id`, falling back to recreate if gone; flips `workspace_root_ready=False` and `_resume_workspace_probe_pending=True` so the next exec probes the workspace dir.

Teardown is split: `DockerSandboxSession._shutdown_backend` does best-effort `container.stop()` (`docker.py:754-765`); `DockerSandboxClient.delete` removes container + named volumes (`docker.py:1375-1404`). Both swallow `docker.errors.NotFound`.

**Steal:** ✅ idle entrypoint + exec-only execution model; create/start/delete split.
**Don't:** 🚫 silent `except Exception: pass` in shutdown — log at minimum.

### 2. Working directory and FS layout

**Their approach.** "Workspace root" lives on the `Manifest`, not in the docker class. Each exec passes `workdir=manifest.root` only after `_workspace_root_ready` is `True`; before bootstrap, `workdir=None` so the daemon doesn't fail with "no such directory" (`docker.py:545-548`). Staging area `/tmp/sandbox-docker-archive` (line 169-171); per-op subpaths are uuid-prefixed (line 264). Mounts are Docker **named volumes**, never bind mounts (line 1542-1551).

**Steal:** 📋 the "workspace root not ready yet" gate + staged probe-on-resume; per-op uuid'd staging dir under `/tmp` for archive writes.
**Don't:** assumption `/tmp` is writable — provide a fallback path.

### 3. File I/O (read / write / archive)

**Their approach.** They explicitly **avoid `get_archive` / `put_archive`** because volume-driver plugins re-trigger mount setup and reject the duplicate mount call (`docker.py:663-666`):

```python
# Read from inside the container instead of `get_archive()`: with Docker
# volume-driver-backed mounts attached, daemon archive operations can re-run volume mount
# setup and some plugins reject the duplicate `Mount` call for the same container id.
```

- **read**: `cat -- <path>` via `exec_run`, bytes-in-bytes-out (`docker.py:660-677`). No encoding assumption.
- **write (no user)**: stream to staging via `sh -lc 'cat > "$1"'`, then `cp -- staging dest`, then best-effort `rm -rf` of staging (`docker.py:712-742`).
- **write (with user)**: stream directly via `sh -lc 'mkdir -p "$(dirname "$1")" && cat > "$1"'` (`docker.py:691-703`).
- **bulk hydrate**: `tar -x -C <root>` via stdin stream (`docker.py:1267-1271`) after validating the tar **on the host** with `validate_tarfile`.
- **bulk persist**: `cp -R` into a staging dir (pruning skips), then HTTP GET `/containers/{id}/archive?path=...` streamed back, response held open and closed in `finally` (`docker.py:1294-1316`). Falls back to `container.get_archive()` if private API attrs aren't present.

`_stream_into_exec` (`docker.py:557-630`) is the streamed-stdin workhorse: `exec_create(stdin=True, ...)`, hijack the raw socket via `_start_exec_socket`, `sendall(chunk)` in 1MB blocks, `shutdown(SHUT_WR)` to signal EOF, drain, `exec_inspect` for exit code.

**Steal:** 📋 read via `cat`, write via streamed `cat >`, NOT `get_archive`/`put_archive`; the `shutdown(SHUT_WR)` + drain recipe; host-side tar validation.
**Don't:** 🚫 reaching into `api._post_json` / `api._url` as the primary path — use public `api.exec_start(..., socket=True)` first; private path is a perf-only optimization.

### 4. Shell command (the exec codepath beyond kill)

**PTY vs non-PTY** — two separate paths.

- **Non-PTY**: synchronous `container.exec_run(demux=True, ...)` shipped to a module-level `ThreadPoolExecutor(max_workers=8, thread_name_prefix='agents-docker-sandbox')` (`docker.py:81-84`, `417-479`) and wrapped in `asyncio.wait_for`. `demux=True` gives stdout/stderr as separate bytes.
- **PTY**: low-level `api.exec_create(tty=tty)` + hijacked socket + reader thread (`docker.py:786-918`, `978-999`).

**Timeout.** Comment at `docker.py:538-541`:
> `docker-py` is synchronous and can block indefinitely (e.g. hung process, daemon issues). Run in a worker thread so we can enforce a timeout without requiring `timeout(1)` in the container image.

Only `asyncio.wait_for` is used. On non-PTY timeout: `pkill -f -- '<pattern>'` (best-effort). For PTY: pidfile (`_kill_pty_pid_path`, `docker.py:1170-1196`).

**Output.** Always bytes. `demux=True` for non-PTY. PTY output goes through `output.decode('utf-8', errors='replace')` then `truncate_text_by_tokens` then re-encode — only the PTY path does encoding, and only because token truncation is text-defined.

**Env vars.** Resolved once per session at create time; per-exec env override is not supported. Per-exec `workdir` and `user` are.

**User.** `user` coerced via `_coerce_exec_user` (`docker.py:504-508`). If passed, takes `_exec_internal_for_user` path which skips the base class's `sudo -u` wrapper and uses Docker's native `--user` flag — cleaner.

**Steal:** 🛠 shared bounded `ThreadPoolExecutor`; ✅ `demux=True`; ✅ `asyncio.wait_for`; 📋 native `--user` flag (not `sudo -u`).
**Don't:** 🚫 `pkill -f` non-PTY fallback (already chose pidfile everywhere).

### 5. Error taxonomy

**Their approach.** `src/agents/sandbox/errors.py`. Single base `SandboxError` dataclass with `(message, error_code, op, context, cause)` (line 77-100). Subtree: `ConfigurationError`, `SandboxRuntimeError`, `ArtifactError`, `SnapshotError`. Exec-specific (line 214-322): `ExecFailureError` → `ExecNonZeroError`, `ExecTimeoutError`, `ExecTransportError`. Docker layer maps `asyncio.TimeoutError` → `ExecTimeoutError`, other Docker exceptions → `ExecTransportError` with `retry_safe` in `context`. **The exec primitive never raises on nonzero**; caller decides.

`ExitCode is None` → transport error, not success (`docker.py:463-474`).

`docker.errors.NotFound` is caught and remapped (`docker.py:1221-1224`) — never leaks through.

**Steal:** 🛠 the `exit_code is None` defensive check; 📋 `error_code` enum + `op` + `context` dict + `cause` shape; 📋 `retry_safe: True` in context for the retry layer.
**Don't:** ~30 concrete subclasses; we keep the family small and discriminate via `error_code`.

### 6. Resource isolation

**Their approach.** Almost nothing. Only knobs are *conditional add-privilege* when manifest needs fuse/rclone (`docker.py:1465-1475`):

```python
if _manifest_requires_fuse(manifest):
    create_kwargs.update(
        devices=['/dev/fuse'],
        cap_add=['SYS_ADMIN'],
        security_opt=['apparmor:unconfined'],
    )
```

No CPU limits, no memory limits, no pids limit, no ulimit, no `read_only` rootfs, no `cap_drop`, no seccomp profile, no `network_mode="none"`, no `tmpfs`.

**Steal:** 📋 conditional add-privilege pattern.
**Don't:** 🚫 wide-open defaults. Harness should default to `cap_drop=['ALL']`, optional `read_only=True` with explicit `tmpfs={'/tmp': ''}`, optional `mem_limit`, `pids_limit`, `network_mode='none'` opt-in.

### 7. Sidecar-ish helpers baked into the container

**Their approach.** Two:
1. `_PREPARE_USER_PTY_PID_SCRIPT` (`docker.py:88-97`) — shell snippet creating the pidfile parent (mode 0711), the pidfile itself (mode 0600), owned by the target user. Solves "unprivileged user can't write to root-owned pidfile path."
2. `RESOLVE_WORKSPACE_PATH_HELPER` — `RuntimeHelperScript` installed at `/tmp/openai-agents/bin/...` via `_runtime_helpers()` (line 266-270), cached per `container_id`.

PTY wrapper itself (`docker.py:817-825`) is a small inline shell template:
```python
wrapped_cmd = [
    'sh', '-lc',
    'mkdir -p "$1" && printf "%s" "$$" > "$2" && shift 2 && exec "$@"',
    'sh', sandbox_path_str(pty_pid_path.parent),
    sandbox_path_str(pty_pid_path),
    *cmd,
]
```

`printf "%s" "$$"` writes the PID, then `exec "$@"` replaces the shell with the real command so the PID file actually contains the right PID.

**Steal:** ✅ the `exec "$@"` (or `exec /bin/sh -c <cmd>` as in our impl) trick after writing `$$`; 📋 helper-cache-keyed-on-container-id pattern so resumes re-install helpers.

### 8. Concurrency

**Their approach.** Module-level `ThreadPoolExecutor(max_workers=8, thread_name_prefix='agents-docker-sandbox')` (`docker.py:81-84`). Multiple execs run in parallel on the same container — Docker supports this natively. Race avoidance:
- PTY pidfile paths are uuid-named (`docker.py:262-264`).
- `_pty_lock = asyncio.Lock()` (line 185) protects `_pty_processes` dict and `_reserved_pty_process_ids` set.
- `output_lock = asyncio.Lock()` per PTY entry for the deque.
- `_pty_processes` capped at `PTY_PROCESSES_MAX` with LRU pruning (`docker.py:1131-1144`).

Reader thread per PTY (`docker.py:863-869`) bridges blocking socket I/O to asyncio via `asyncio.run_coroutine_threadsafe`.

**Steal:** 🛠 module-level bounded executor (prevent thread leak); ✅ uuid-named pidfiles.
**Don't:** the `future.result()` call inside `_pump_pty_socket` (`docker.py:988`) — blocks the reader thread on every chunk. Throughput cliff. Batch.

### 9. Image management

**Their approach.** Pull-on-demand, no digest pinning, no Dockerfile support, no tool-presence validation (`docker.py:1445-1450`):

```python
if not self.image_exists(image):
    repo, tag = parse_repository_tag(image)
    self.docker_client.images.pull(repo, tag=tag or None, all_tags=False)
```

No check that `kill`, `cat`, `sh`, `pkill`, `tar` exist — fail at first use.

**Steal:** `parse_repository_tag` for clean repo/tag split; pull-on-demand idempotency.
**Don't:** lack of image validation. 📋 At minimum document required binaries (`sh`, `cat`, `kill`, `tar`, `mkdir`); ideally probe once at session start.

### 10. Streaming output

**Their approach.** Non-PTY: no streaming — `exec_run` blocks and returns full bytes. PTY: chunked yields — `pty_exec_start` returns after `yield_time_ms` (default 10s start, 250ms stdin writes) with accumulated output; caller polls `pty_write_stdin(chars='')` to drain more (`docker.py:920-967`). `truncate_text_by_tokens` (`docker.py:1099`) caps output by token count.

**Steal:** 📋 "yield time" deadline pattern for interactive runs; 📋 token-aware truncation if exposing output to a model.

### 11. macOS / Colima / remote daemon support

**Their approach.** No platform-specific code in `docker.py`. `DockerClient` is **passed in by the user** (`docker.py:1324-1334`), so any daemon URL works. The `sandboxes/__init__.py` gates the Unix-local sandbox behind `sys.platform != 'win32'` (fix for issue #2938), but Docker has no platform gate.

**Steal:** 📋 user-supplied `DockerClient` — keeps platform/daemon choice out of our code. Currently we do `docker.from_env()`; could accept injected client.

### 12. Tests

**Their approach.** Two layers:
- **Unit tests** (`tests/sandbox/test_docker.py`, 2925 lines): zero real daemon. Fakes: `_FakeDockerContainer`, `_FakeDockerClient`, `_StreamingArchiveAPI`, `_FakePtyApi` (`tests/sandbox/test_docker.py:65-2070`). `@pytest.mark.asyncio`. Also `_HostBackedDockerSession` — a subclass of `DockerSandboxSession` that runs everything against the local filesystem to validate session-level orchestration without docker.
- **Integration tests** (`tests/sandbox/integration_tests/`): two files, `test_model.py` + `test_runner_pause_resume.py`. Shared `_helpers.py` canonical fixtures across backends.

No `pytest.mark.slow` or `requires_docker` marker — they invest heavily in fakes. Retry/flake guard: `retry_async` decorator (`src/agents/sandbox/util/retry.py`) used on `persist_workspace` (`docker.py:1205-1207`) with `TRANSIENT_HTTP_STATUS_CODES`.

**Steal:** 📋 host-backed-session pattern (subclass real session, swap `_exec_run` to run against tmpdir — single biggest test-velocity unlock); `retry_async` on archive-stream calls; shared workspace-fixture helper as canonical scenario set across backends.
**Don't:** the 2925-line single test file. Split.

### 13. Performance

**Their approach.** Containers reused for entire session via `tail -f /dev/null`. Image pulled once per `create_container`; no warmup, no benchmarks, no caching layer beyond Docker's. Archive streaming uses `DEFAULT_DATA_CHUNK_SIZE` from docker-py (`docker.py:1308`). Shared 8-thread executor is the only concurrency bound.

**Steal:** container reuse via idle entrypoint (✅ ours uses `sleep infinity` which is equivalent); `DEFAULT_DATA_CHUNK_SIZE` import for archive streaming chunk size.

### 14. Cleanup safety

**Their approach.** Weak. No `--rm`, no labels, no `weakref.finalize`/`atexit`. If host crashes mid-exec: container stays running (`tail -f /dev/null`), volumes survive. Next session resume by `container_id` reattaches. If `container_id` isn't persisted, orphan.

`DockerSandboxClient.delete` is the only path that removes containers + volumes (`docker.py:1383-1403`); only called explicitly. Deterministic volume names (`session_uuid_hex + sanitized_path + sha256[:12]`, line 1585-1592) prevent collisions.

**Steal:** deterministic-named volumes with session-uuid prefix.
**Don't:** 📋 no orphan reaping. Add `labels={'pydantic-ai-harness': '1', 'session': session_id}` to containers/volumes so a prune helper can clean up later.

### 15. Known bugs and issues (from their tracker)

Half of all docker-adjacent bugs are **path / symlink / tar safety**.

| # | Title | Status | Root cause | Hits us? |
|---|---|---|---|---|
| #2938 | Windows import fails: `ModuleNotFoundError: fcntl` | CLOSED | Eager import of Unix-only stdlib | If we ship unix backend — gate behind `sys.platform`. |
| #2962 | `WorkspaceStartError` on Windows | OPEN | Host paths joined with backslashes, passed as Linux PATH | Yes — use `PurePosixPath`/`.as_posix()` for container paths. |
| #3093 | Workspace hydrate accepts symlinks outside archive root | CLOSED | `validate_tarfile` didn't validate symlink `linkname` | Yes — reject absolute or `..`-escaping symlink targets. |
| #3274 | Archive extraction has no resource limits | CLOSED | No cap on bytes, member count, declared extract size | Yes — set limits before extraction. |
| #3170 | GitRepo artifact leaves temp clone after copy failure | CLOSED | Missing finally-cleanup on materialization failure | Indirectly — any "stage then copy" path needs finally-cleanup. |
| #3169 | LocalFile/LocalDir abs src paths can read outside base_dir | CLOSED | Path validation missing | Yes — validate every host-side path against an allowlist before reading. |
| #3452 | LocalDir copy can follow symlink-swapped sources | OPEN | TOCTOU symlink race during copy | Yes if we add LocalDir-style materialization. |

### 16. Public API ergonomics

**Their approach.**
```python
client = DockerSandboxClient(docker_client=DockerClient.from_env())
opts = DockerSandboxClientOptions(image='python:3.12', exposed_ports=(8000,))
session = await client.create(options=opts, manifest=manifest)
# session.exec(...), session.read(...), session.write(...)
await client.delete(session)
```

`SandboxSession` is `async with` (`base_sandbox_session.py:337-365`). `DockerSandboxClient` is NOT — long-lived factory.

- **DI** of `DockerClient`, not constructed internally.
- `Options` is a Pydantic-validated model with `type: Literal['docker']` for discriminated unions (`docker.py:106-122`).
- `resume(state)` + `deserialize_session_state(payload)` for crash recovery (`docker.py:1406-1432`).
- `register_pre_stop_hook` on base session for capability cleanup ordering.

**Steal:** 📋 DI of `DockerClient`; ✅ `Session` is `async with`; 📋 Pydantic options with discriminated `type`; 📋 `resume_from_state`; 📋 `register_pre_stop_hook`.

### 17. Other hard-won lessons

- **`exec_inspect` after `exec_start` to get the exit code** (`docker.py:615`). The exit code is NOT in the start response.
- **`shutdown(SHUT_WR)` to signal EOF on stdin** for streamed-stdin execs (`docker.py:596-601`). Without it, in-container `cat` hangs forever.
- **Use `--` after every command flag** when paths can start with `-` (`docker.py:314-318`, `667`, `730`). Universally applied to `cp`, `cat`, `rm`.
- **`exec_inspect.Running == False && ExitCode != None`** is the only reliable "finished" signal (`docker.py:1029-1033`).
- **`_DockerExecSocket` close must close BOTH the wrapped socket and the underlying HTTP response** (`docker.py:148-156`). Forgetting the response leaks an HTTP connection.
- **`Accept-Encoding: identity` on archive streams** (`docker.py:1299`) — disable gzip to keep the tar stream raw; otherwise urllib3 decompresses and corrupts offsets.
- **`auto_remove=True` is incompatible** with long-lived idle entrypoint (any stop = removal = lost state). Choose explicit delete + resume.
- **`_validate_path_access`** called on every read/write **before** any docker call (`docker.py:661`, `688`). Path policy enforcement is **in the session**, not delegated to container OS — works even if user is `root` inside.

## Hard nos (collected)

- No `cap_drop`, `read_only`, `pids_limit`, `mem_limit`, `network_mode` defaults wide open.
- No `pkill -f` based kill (use pidfile pattern uniformly).
- No silent `except Exception: pass` in shutdown.
- No skipping image-tools validation at session start.
- No orphan containers without labels.
- No reaching into `api._post_json` / `api._url` as primary path.
