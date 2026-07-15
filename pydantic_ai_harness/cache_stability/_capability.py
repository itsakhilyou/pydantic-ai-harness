"""CacheStabilityMonitor: an observational cache-collapse warning.

This is the runtime `observe` arm of the cache-prefix-stability work. It does not
inspect the structured request (that signal false-positives on internal metadata
serialization strips, and is blind to serialization-level busts). Instead it reads
the provider's own ground-truth verdict -- `response.usage.cache_read_tokens` -- and
warns when a cache hit that was previously established collapses. That verdict is
cross-provider for free: pyai normalizes every provider into the `cache_read_tokens`
/ `cache_write_tokens` fields on `RequestUsage` via genai-prices.

The monitor only fires when caching is actually enabled and reported, which is the
honest scope of a runtime signal. The deterministic, always-on structural catch
lives at the wire level in `tests/` (VCR cassette prefix assertion), not here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

# A response's (provider_name, model_name) -- identifies which provider cache the tokens came from.
_CacheKey = tuple[str | None, str | None]

_SILENCE_HINT = (
    '    import warnings\n'
    '    from pydantic_ai_harness.cache_stability import CacheBustWarning\n'
    "    warnings.filterwarnings('ignore', category=CacheBustWarning)  # silence\n"
    "    warnings.filterwarnings('error', category=CacheBustWarning)   # escalate in dev/CI"
)


class CacheBustWarning(UserWarning):
    """Warned when a previously-established prompt cache hit collapses on a later request.

    Emitted by `CacheStabilityMonitor` when this run read back far fewer cached tokens for the
    same provider and model than a prior request established. The likely causes are a moved
    cacheable prefix (reordered tools, injected timestamps, a serialization-level block hop) or a
    provider-side cache expiry under an unchanged prefix (a gap between requests longer than the
    cache TTL). The monitor observes the collapse; it does not attribute the cause.

    Silence it, or escalate it to an error in dev/CI, with the stdlib `warnings` machinery
    (no bespoke API):

        import warnings
        from pydantic_ai_harness.cache_stability import CacheBustWarning

        # Silence the whole category:
        warnings.filterwarnings('ignore', category=CacheBustWarning)

        # Silence one intentional bust, scoped to the operation that causes it:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CacheBustWarning)
            result = agent.run_sync('...')  # e.g. a step that switches models or adds a file

        # Treat every bust as an error (dev/CI enforcement):
        warnings.filterwarnings('error', category=CacheBustWarning)

    In tests, assert an intentional bust with `pytest.warns(CacheBustWarning)`, or silence
    a legitimately-busting test with
    `@pytest.mark.filterwarnings('ignore::pydantic_ai_harness.cache_stability.CacheBustWarning')`.
    """


@dataclass
class CacheStabilityMonitor(AbstractCapability[AgentDepsT]):
    """Warn when a run's prompt cache hit collapses between requests.

    Attach it to any agent whose model uses prompt caching. On each response the monitor
    reads `usage.cache_read_tokens` and tracks the largest cacheable prefix the run has
    established so far (`cache_read_tokens + cache_write_tokens`, a high-water mark), keyed by
    the response's `(provider_name, model_name)`. When a later request for the same key reads
    back fewer than `collapse_ratio` of that established prefix, it emits a `CacheBustWarning`.

    Keying per provider and model means a mid-run model switch does not warn: a `FallbackModel`
    failover or a per-step model change uses a different cache key, so it starts a fresh high-water
    mark for that key instead of comparing against the previous model's. Marks are kept per key
    rather than reset, so switching back to an earlier model within its cache TTL still compares
    against that model's established prefix.

    Because message history is append-only, a stable prefix means each request reads back at
    least what the previous one cached. A large drop is the observable signature of a collapse,
    whether the cause is a moved prefix (reordered tools, injected timestamps, a
    serialization-level block hop) or a provider-side cache expiry when the gap between requests
    exceeds the cache TTL. The monitor surfaces the collapse; it does not attribute the cause.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.cache_stability import CacheStabilityMonitor

    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[CacheStabilityMonitor()])
    await agent.run('...')  # a CacheBustWarning fires if a cached prefix collapses mid-run
    ```

    The monitor is silent when caching is off or unreported (`cache_read_tokens` stays 0), so
    it never fires spuriously in tests that don't exercise caching. Silencing and dev/CI
    escalation both go through the stdlib `warnings` filters -- see `CacheBustWarning`.
    """

    collapse_ratio: float = 0.5
    """Warn when a request reads back less than this fraction of the established prefix.

    Conservative by default (0.5): only a drop below half the previously-cached prefix counts
    as a collapse, so ordinary provider rounding or a partial cache miss does not fire. Raise
    it toward 1.0 to warn on smaller regressions.
    """

    min_prefix_tokens: int = 1024
    """Only judge collapse once the established prefix reaches this many tokens.

    Below a provider's minimum cacheable size (Anthropic's is 1024) `cache_read_tokens` is
    noisy or zero, so small prefixes are ignored to avoid false positives.
    """

    cache_ttl_seconds: float = 300.0
    """Assumed provider cache TTL, in seconds (Anthropic's default is 300, refreshed on each hit).

    Message-only: when the gap since the previous request in the run exceeds this, the warning
    notes that the collapse may be a provider-side cache expiry rather than a moved prefix. It
    does not change whether a warning fires. Lower it for providers with a shorter cache lifetime.
    """

    _marks: dict[_CacheKey, int] = field(default_factory=dict[_CacheKey, int], compare=False)
    _step: int = field(default=0, compare=False)
    _last_time: datetime | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.collapse_ratio <= 1.0:
            raise ValueError('collapse_ratio must be between 0.0 and 1.0')
        if self.min_prefix_tokens < 0:
            raise ValueError('min_prefix_tokens must be non-negative')
        if self.cache_ttl_seconds < 0:
            raise ValueError('cache_ttl_seconds must be non-negative')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Reset per-run state (per-key high-water marks, step, timing) so each run is judged alone."""
        return replace(self, _marks={}, _step=0, _last_time=None)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Compare this response's cache read against the established prefix for its model, then update it."""
        self._step += 1
        usage = response.usage
        read = usage.cache_read_tokens
        key = (response.provider_name, response.model_name)
        established = self._marks.get(key, 0)
        # 0.0 on the run's first request, which has no predecessor to measure a gap against.
        gap = 0.0 if self._last_time is None else (response.timestamp - self._last_time).total_seconds()
        self._last_time = response.timestamp
        if established >= self.min_prefix_tokens and read < established * self.collapse_ratio:
            wasted = established - read
            if gap > self.cache_ttl_seconds:
                expiry = (
                    f' -- the previous request was ~{gap:.0f}s earlier, '
                    f'past the assumed ~{self.cache_ttl_seconds:.0f}s cache TTL'
                )
            else:
                expiry = ' (e.g. a gap longer than the cache TTL)'
            warnings.warn(
                f'Cache hit collapsed at model request {self._step}: read {read} cached tokens but '
                f'a prior request established ~{established} (~{wasted} tokens re-sent uncached). '
                f"The cacheable prefix moved between requests, or the provider's cache expired{expiry}.\n\n"
                f'To silence or escalate:\n\n{_SILENCE_HINT}\n',
                CacheBustWarning,
                stacklevel=2,
            )
        self._marks[key] = max(established, read + usage.cache_write_tokens)
        return response
