"""Shared ripgrep subprocess helper used by every backend that exposes `grep`/`glob`.

Centralizing here is what lets the ABC promise a single regex/glob dialect across backends:
both `LocalEnvironment` and `DockerEnvironment` shell out to the same `rg` binary with the
same argv, so a pattern that works on one works identically on the other. The conformance
suite only needs to verify one engine.
"""

import asyncio
import json
from pathlib import Path, PurePath

from .abstract import AbstractMatch
from .exceptions import EnvInvalidPatternError, EnvReadError

# rg's documented exit-code contract: 0 = matches found, 1 = no matches (normal empty result),
# 2 = usage error / pattern failed to compile / per-file I/O error mid-walk. We disambiguate
# the 2 case by stdout shape -- empty stdout means compile failure; non-empty means a partial
# successful walk we still want to surface as matches.
RG_EXIT_USAGE_OR_PATTERN = 2


async def run_ripgrep_files(root: Path, target: Path, pattern: str) -> list[str]:
    """List files under `target` matching glob `pattern`, returned as paths relative to `root`.

    Uses `rg --files -g <pattern>` so glob and grep share a single engine (the `globset` crate)
    -- the conformance suite has exactly one dialect to verify across backends. Same argv-safety
    discipline as `run_ripgrep`:

    - `--glob=<pattern>` carries the pattern inside one argv element, can never be parsed as a flag.
    - `--no-ignore` so glob results don't depend on whether the dir is a git repo (matches the
      historical `Path.rglob` behavior LocalEnvironment shipped).
    - `--no-config` + `--one-file-system` defend against hostile env vars / mount escapes.
    - `--` then `<target>` so a path starting with `-` is never interpreted as a flag.

    rg exits 0 with matches or 1 with no matches; both are normal. Exit 2 means the glob did not
    compile (`EnvInvalidPatternError`) or rg hit an I/O error mid-walk (`EnvReadError`).
    """
    argv = [
        'rg',
        '--files',
        '--no-ignore',
        '--no-config',
        '--one-file-system',
        f'--glob={pattern}',
        '--',
        str(target),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:  # pragma: no cover
        raise EnvReadError(f'failed to spawn ripgrep for {target!s}: {e}') from e

    stdout, stderr = await proc.communicate()
    assert proc.returncode is not None

    # Same exit-code discrimination as `run_ripgrep`: empty stdout + exit 2 means the glob did
    # not compile (model-fixable); exit 2 with output means rg walked but hit a per-file error
    # mid-stream, which we surface as `EnvReadError`. Exit 1 (no matches) returns an empty list.
    if proc.returncode == RG_EXIT_USAGE_OR_PATTERN:
        msg = stderr.decode(errors='replace').strip()
        if not stdout:
            raise EnvInvalidPatternError(f'invalid glob {pattern!r}: {msg}')
        raise EnvReadError(f'ripgrep failed for {target!s}: {msg}')

    # Dotfile policy is rg's: hidden directories are NOT descended into (so `.cache/x.py` is
    # excluded), but top-level dotfile leaves matched by `--glob` ARE returned (`.hidden.py`
    # comes through). We accept that behavior rather than layering a second filter -- single
    # engine, single dialect, no policy drift. The ABC docstring documents the same rule.
    return [str(Path(line.decode()).relative_to(root)) for line in stdout.splitlines() if line]


async def run_ripgrep(root: Path, target: Path, pattern: str) -> list[AbstractMatch]:
    """Invoke `rg` against `target` and return matches as `AbstractMatch` objects.

    The argv is constructed defensively so the model-supplied `pattern` cannot escape into
    flag interpretation:

    - `--regexp=<pattern>` puts the pattern inside a single argv element with `=`, so it can
      never be parsed as a flag (even values that look like `--pre=/bin/sh` are just regex).
    - `--` separates flags from the target, defending against any future change where target
      paths could begin with `-`.
    - `--no-config` ignores `$RIPGREP_CONFIG_PATH` and `~/.ripgreprc` so a hostile env var or
      home dir cannot inject flags into our invocation.
    - `--one-file-system` refuses to traverse mount boundaries, defending against escape via
      a deliberately mounted path inside `root`.

    rg's exit codes drive our error mapping:
      0 -> matches found; parse JSON
      1 -> no matches (a normal empty result -- NOT an error)
      2 -> usage error, regex parse failure, or I/O error -- raise based on stderr
    """
    argv = [
        'rg',
        '--json',
        '--no-config',
        '--one-file-system',
        f'--regexp={pattern}',
        '--',
        str(target),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:  # pragma: no cover
        # Defensive: `ripgrep` PyPI wheel ships the binary into the venv `bin/`, so this only
        # trips if the wheel is missing/broken or the venv is corrupt. No test exercises it.
        raise EnvReadError(f'failed to spawn ripgrep for {target!s}: {e}') from e

    stdout, stderr = await proc.communicate()
    assert proc.returncode is not None

    # rg's exit codes alone don't tell us what went wrong on exit 2: that lumps regex parse
    # failure (model-fixable -> ModelRetry) together with per-file IO errors (permission
    # denied on one file out of many, broken symlinks -- NOT actionable; rg keeps walking
    # and matches what it can). We need to distinguish without scraping stderr, because
    # error text isn't a stable API.
    #
    # The structural discriminator is the SHAPE of stdout: rg compiles the pattern BEFORE
    # walking. If compilation fails, rg writes the error to stderr and exits 2 *with no
    # JSON output at all*. If compilation succeeds and rg then stumbles on a file mid-walk,
    # the per-file error goes to stderr but every file rg did open emits its begin/match/end
    # events -- so stdout is non-empty. Empty stdout + exit 2 is therefore unambiguously
    # "the pattern didn't compile"; anything else with output is a partial success we
    # return as matches.
    if proc.returncode == RG_EXIT_USAGE_OR_PATTERN and not stdout:
        raise EnvInvalidPatternError(f'invalid regex {pattern!r}: {stderr.decode(errors="replace").strip()}')

    return parse_ripgrep_json(stdout, root)


def parse_ripgrep_json(stdout: bytes, root: PurePath) -> list[AbstractMatch]:
    """Translate rg's `--json` output into `AbstractMatch` instances.

    rg emits one JSON object per line; we only care about `type: "match"` events. The data
    schema is documented at https://docs.rs/grep-printer/latest/grep_printer/struct.JSON.html;
    the fields we use (`data.path.text`, `data.line_number`, `data.lines.text`) are stable
    across rg versions. Path is normalized to be relative to `root` so AbstractMatch.path
    matches LocalEnvironment's previous behavior. `root` is `PurePath` so callers can pass
    `Path` (host) or `PurePosixPath` (container) without conversion -- `type(root)` is used to
    construct the path object that gets `relative_to`'d.
    """
    path_cls = type(root)
    matches: list[AbstractMatch] = []
    for raw_line in stdout.splitlines():
        event = json.loads(raw_line)
        if event.get('type') != 'match':
            # Skip begin/end/summary/context events; only 'match' carries the data we map.
            continue
        data = event['data']
        # `path.text` is always absolute -- rg returns the path it walked, and we always pass it
        # an absolute target (the resolved root + user path). `lines.text` is the matched line,
        # newline-terminated as rg printed it. `line_number` is 1-indexed.
        path_text: str = data['path']['text']
        line_text: str = data['lines']['text']
        lineno: int = data['line_number']
        rel_path = str(path_cls(path_text).relative_to(root))
        # rstrip('\n'), not rstrip(): preserve trailing whitespace that was part of the line.
        matches.append(AbstractMatch(path=rel_path, line=line_text.rstrip('\n'), lineno=lineno))
    return matches
