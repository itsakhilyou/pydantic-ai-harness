"""Cache-friendly tool catalog disclosure for `CodeMode`.

Moves the per-tool signature block out of `run_code.description` (which lives in the
prompt-cache-keyed tool-definitions block) and into agent instructions, and announces
newly-discovered tools as a `SystemPromptPart` injected via the pending message queue
instead of by mutating the cached `ToolDefinition.description`. See the design write-up
in pydantic-ai-notes (`features/tool-search/2026-05-07 claude grapple ...`) and the
Tier-2 section of pydantic/pydantic-ai-harness#232.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    ModelResponse,
    NativeToolSearchReturnPart,
    SystemPromptPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from pydantic_ai_harness.code_mode import CodeMode

# Tier 2 has to coordinate with `code_mode/_toolset.py`'s internals to swap out the
# `run_code` tool definition. The toolset module is intentionally private to discourage
# user code from depending on these symbols, but a sibling capability is the legitimate
# consumer.
from pydantic_ai_harness.code_mode._toolset import (
    _RUN_CODE_TOOL_NAME,  # pyright: ignore[reportPrivateUsage]
    CodeModeToolset,
    _RunCodeTool,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart


_DISCOVERY_ANNOUNCEMENT_PREFIX = (
    'New functions are now available inside `run_code`. Their signatures have been '
    'added to the available-functions catalog in the system prompt'
)


@dataclass
class CodeModeDynamicCatalog(AbstractCapability[AgentDepsT]):
    """Move CodeMode's tool catalog from `run_code.description` into agent instructions.

    Pair this capability with [`CodeMode`][pydantic_ai_harness.CodeMode] (it requires
    `CodeMode` and is sequenced to wrap it) to make the tool-disclosure surface
    cache-friendly. Out of the box, `CodeMode` renders every sandboxed tool's signature
    into the `run_code` tool's description, which lives in the prompt-cache-keyed
    tool-definitions block — any change (e.g. Tool Search revealing a new tool) busts
    the prefix cache from that point forward.

    With this capability:

    - `run_code.description` keeps only the base prose (sandbox restrictions,
      return-value contract). It's static, never re-rendered → the tool-defs block
      stays cache-stable across discoveries.
    - The "available functions" catalog (TypedDict definitions + function signatures)
      moves into instructions as a dynamic [`InstructionPart`][pydantic_ai.messages.InstructionPart],
      which providers with static/dynamic instruction splitting (Anthropic, Bedrock)
      can place after the static cache breakpoint.
    - When [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] reveals new tools,
      a short [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is enqueued
      via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] announcing the
      discoveries, so the model knows the new functions are callable.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai_harness import CodeMode, CodeModeDynamicCatalog

    agent = Agent(
        'anthropic:claude-sonnet-4-5',
        capabilities=[
            ToolSearch(),
            CodeMode(),
            CodeModeDynamicCatalog(),
        ],
    )
    ```

    !!! note "Tradeoff vs. default CodeMode"
        Without this capability, the catalog sits in `run_code.description`, which keeps
        the system prompt simpler and shorter when the toolset never changes. With it,
        the system prompt grows by the catalog size — but the *tool-definitions* portion
        of every request becomes byte-stable, so the prefix cache (and Anthropic's static
        instruction cache) survives discoveries. Pair it with `ToolSearch` for the win;
        without `ToolSearch` (and with a fixed toolset) the default behavior is fine.
    """

    _announced_tools: set[str] = field(default_factory=set[str], init=False, repr=False)

    def get_ordering(self) -> CapabilityOrdering:
        # `requires=[CodeMode]`: this capability is a no-op without CodeMode in the agent.
        # `wraps=[CodeMode]`: our `get_wrapper_toolset` must wrap CodeMode's so we see the
        # assembled `run_code` tool and can rewrite its `ToolDefinition`.
        # `position='outermost'`: CodeMode also declares itself outermost — without matching
        # that, the outermost-tier rule + our `wraps=[CodeMode]` form a cycle (CodeMode must
        # precede all non-outermost capabilities, but `wraps` says we must precede CodeMode).
        # Both being outermost lets relative `wraps`/`wrapped_by` settle the intra-tier order
        # without contention.
        return CapabilityOrdering(position='outermost', requires=[CodeMode], wraps=[CodeMode])

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CodeModeDynamicCatalog[AgentDepsT]:
        # Fresh per-run state so two concurrent runs don't share `_announced_tools`.
        return CodeModeDynamicCatalog()

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return _CatalogToolset(wrapped=toolset)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Announce newly-discovered tools from a local `search_tools` return.

        The native-search path is handled by [`after_model_request`][pydantic_ai_harness.CodeModeDynamicCatalog.after_model_request]
        instead (server-side search emits a `NativeToolSearchReturnPart` rather than a
        regular tool execute result).
        """
        if tool_def.tool_kind == 'tool-search':
            self._announce_newly_discovered(ctx, _extract_discovered_names(result))
        return result

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: Any,
        response: ModelResponse,
    ) -> ModelResponse:
        """Announce newly-discovered tools from a native (server-side) tool-search return."""
        for part in response.parts:
            if isinstance(part, NativeToolSearchReturnPart):
                self._announce_newly_discovered(ctx, _extract_discovered_names(part.content))
        return response

    def _announce_newly_discovered(self, ctx: RunContext[AgentDepsT], names: Sequence[str]) -> None:
        """Enqueue a system-prompt announcement for any names we haven't already announced."""
        fresh = [n for n in names if n not in self._announced_tools]
        if not fresh:
            return
        self._announced_tools.update(fresh)
        listing = ', '.join(f'`{name}`' for name in fresh)
        # Enqueue as a `ModelRequest(SystemPromptPart)` so it's framed as system-level
        # context. PR #4980's `EnqueueContent` excludes bare `SystemPromptPart` (provider
        # mappings vary today — see pydantic/pydantic-ai#5437), but a ModelRequest
        # passthrough is allowed and rendered inline. Once #5437 lands, providers that
        # currently hoist mid-conversation system content will instead inline it as an
        # XML-wrapped user prompt, making this fully cache-safe across providers.
        ctx.enqueue(
            ModelRequest(
                parts=[SystemPromptPart(content=f'{_DISCOVERY_ANNOUNCEMENT_PREFIX}: {listing}.')],
            )
        )


@dataclass
class _CatalogToolset(WrapperToolset[AgentDepsT]):
    """Wrapper toolset that strips the catalog from `run_code.description` and surfaces it as instructions.

    Sits outside `CodeModeToolset` so it sees the already-assembled `_RunCodeTool`. Each
    `get_tools` call rebuilds the catalog string and stashes it on the wrapper instance;
    [`get_instructions`][pydantic_ai_harness.code_mode_dynamic_catalog._capability._CatalogToolset.get_instructions]
    returns the latest value as a dynamic `InstructionPart`. The toolset → instructions
    handoff is single-instance per step, so the stash is consistent.
    """

    _last_catalog: str = field(default='', init=False, repr=False)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # `WrapperToolset.for_run_step` produces a new instance via `replace(self, wrapped=...)`
        # when the wrapped toolset changed — that drops `_last_catalog` because it's
        # `init=False`. Mirror what `CodeModeToolset.for_run_step` does for `_repl`:
        # explicitly carry the stash across.
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        new_self = replace(self, wrapped=new_wrapped)
        new_self._last_catalog = self._last_catalog
        return new_self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await self.wrapped.get_tools(ctx)
        run_code = tools.get(_RUN_CODE_TOOL_NAME)
        if not isinstance(run_code, _RunCodeTool):
            # No CodeMode in the chain (or a custom run_code tool replaced it): nothing to do.
            self._last_catalog = ''
            return tools
        self._last_catalog = CodeModeToolset._render_catalog(run_code.callable_defs)  # pyright: ignore[reportPrivateUsage]
        if not self._last_catalog:
            return tools
        # Replace `run_code.description` with the base prose only. We can't import the
        # base-prose constant without coupling, so use `_render_catalog`-aware rebuilding:
        # `_build_description({})` returns exactly the base prose.
        base_description = CodeModeToolset._build_description({})  # pyright: ignore[reportPrivateUsage]
        new_td = replace(run_code.tool_def, description=base_description)
        new_run_code = replace(run_code, tool_def=new_td)
        return {**tools, _RUN_CODE_TOOL_NAME: new_run_code}

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        upstream = await self.wrapped.get_instructions(ctx)
        if not self._last_catalog:
            return upstream
        # `dynamic=True` so providers that split static vs dynamic instructions (Anthropic,
        # Bedrock) place a cache breakpoint *before* this catalog — discoveries change the
        # catalog but leave the static prefix cache intact.
        catalog_part = InstructionPart(content=self._last_catalog, dynamic=True)
        if upstream is None:
            return catalog_part
        if isinstance(upstream, (str, InstructionPart)):
            return [upstream, catalog_part]
        return [*upstream, catalog_part]


def _extract_discovered_names(content: Any) -> list[str]:
    """Read newly-discovered tool names from a tool-search return content.

    Accepts both the local `ToolSearchReturnContent` (TypedDict shape) and the same shape
    on a `NativeToolSearchReturnPart`. Returns `[]` for any malformed/unexpected input —
    the announcement is a courtesy nudge, not load-bearing logic.
    """
    if not isinstance(content, dict):
        return []
    typed = cast(dict[str, Any], content)
    raw = typed.get('discovered_tools')
    if not isinstance(raw, list):
        return []
    raw_list = cast(list[Any], raw)
    names: list[str] = []
    for match in raw_list:
        if isinstance(match, dict):
            name = cast(dict[str, Any], match).get('name')
            if isinstance(name, str):
                names.append(name)
    return names


__all__ = ['CodeModeDynamicCatalog']
