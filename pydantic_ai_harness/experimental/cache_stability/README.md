# CacheStabilityMonitor

Warn when a run's prompt cache hit collapses between model requests.

Prompt caching pays off only while the cacheable prefix (tools, then system
instructions, then message history) stays byte-stable across a run's consecutive
requests. When something moves that prefix -- reordered tools, a timestamp
injected into instructions, a serialization-level block hop -- the provider
re-charges tokens it could have served from cache. `CacheStabilityMonitor` makes
that collapse visible.

This is the **observe** signal: it reads the provider's own verdict rather than
guessing from the structured request. On each response it reads
`usage.cache_read_tokens` and tracks the largest cacheable prefix the run has
established (`cache_read_tokens + cache_write_tokens`, a high-water mark). Because
message history is append-only, a stable prefix means each request reads back at
least what the previous one cached; a large drop is the observable signature of a
bust, whatever the cause.

The verdict is cross-provider for free -- pyai normalizes every provider into the
`cache_read_tokens` / `cache_write_tokens` fields on `RequestUsage`.

> This capability is experimental and private. It is not re-exported from
> `pydantic_ai_harness`; import it from its own module. Its API may change or be
> removed in any release.

## Minimal usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.cache_stability import CacheStabilityMonitor

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[CacheStabilityMonitor()])
await agent.run('...')  # a CacheBustWarning fires if a cached prefix collapses mid-run
```

The monitor is silent when caching is off or unreported (`cache_read_tokens`
stays 0), so it never fires spuriously in runs that don't use caching. That is the
honest scope of a runtime signal -- the deterministic, always-on structural catch
belongs at the wire level in tests, not here.

## Options

- `collapse_ratio` (default `0.5`): warn when a request reads back less than this
  fraction of the established prefix. Conservative by default so ordinary rounding
  or a partial miss does not fire; raise toward `1.0` to warn on smaller
  regressions.
- `min_prefix_tokens` (default `1024`): only judge collapse once the established
  prefix reaches this many tokens. Below a provider's minimum cacheable size
  (Anthropic's is 1024) `cache_read_tokens` is noisy or zero.

## Silencing and escalation

There is no bespoke suppression API. Use the stdlib `warnings` machinery, exactly
as you would manage any other `UserWarning`:

```python
import warnings
from pydantic_ai_harness.experimental.cache_stability import CacheBustWarning

# Silence the whole category:
warnings.filterwarnings('ignore', category=CacheBustWarning)

# Silence one intentional bust, scoped to the operation that causes it:
with warnings.catch_warnings():
    warnings.simplefilter('ignore', CacheBustWarning)
    result = agent.run_sync('...')  # e.g. a step that switches models or adds a file

# Treat every bust as an error (dev/CI enforcement):
warnings.filterwarnings('error', category=CacheBustWarning)
```

In tests, assert an intentional bust with `pytest.warns(CacheBustWarning)`, or
silence a legitimately-busting test with
`@pytest.mark.filterwarnings('ignore::pydantic_ai_harness.experimental.cache_stability.CacheBustWarning')`.

## Composition

- The monitor only implements `for_run` and `after_model_request`; it adds no
  tools, instructions, or model settings, so it composes with any other
  capability, toolset, or `ToolSearch` setup without interference.
- Per-run state (the high-water mark) is materialized in `for_run`, so one
  `CacheStabilityMonitor` instance can be reused across many `Agent.run` calls --
  each run is judged independently.

## Scope

- **Observational only.** It reports that a cached prefix collapsed, not why. The
  structural explanation ("what moved the prefix this turn") is a separate job.
- **Fires only when caching is enabled and reported.** A run that never establishes
  a cache never warns.
