"""Tests for the `CodeModeDynamicCatalog` capability.

The capability has two surfaces:

1. **`_CatalogToolset` wrapper** — strips the catalog from `run_code.description` and
   re-exposes it as a dynamic `InstructionPart`. Exercised directly against
   `CodeModeToolset` instances.
2. **Discovery announcements** — `after_tool_execute` (local-search path) and
   `after_model_request` (native-search path) enqueue a `SystemPromptPart` so the
   model knows freshly-discovered tools are now callable.

Tests follow the `BackgroundTools` style: module-level `pytestmark = pytest.mark.anyio`,
an `anyio_backend` fixture, async tests, and a `FunctionModel` end-to-end exercise.
"""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from pydantic_ai import (
    AbstractToolset,
    Agent,
    RunContext,
    Tool,
    ToolDefinition,
)
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolReturnPart,
    NativeToolSearchReturnPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage

from pydantic_ai_harness import CodeMode, CodeModeDynamicCatalog
from pydantic_ai_harness.code_mode import CodeModeToolset
from pydantic_ai_harness.code_mode._toolset import _RUN_CODE_TOOL_NAME  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.code_mode_dynamic_catalog._capability import (  # pyright: ignore[reportPrivateUsage]
    _CatalogToolset,
    _extract_discovered_names,
)

pytestmark = pytest.mark.anyio

T = TypeVar('T')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _build_run_context(deps: T = None, run_step: int = 0) -> RunContext[T]:  # pyright: ignore[reportArgumentType]
    return RunContext[T](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


def _add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _build_code_mode_toolset(*tools: Any) -> CodeModeToolset[None]:
    base = FunctionToolset[None](tools=[Tool(t) for t in tools])
    return CodeModeToolset[None](wrapped=base, tool_selector='all')


# ---------------------------------------------------------------------------
# `_CatalogToolset` — moving the catalog from description to instructions
# ---------------------------------------------------------------------------


class TestCatalogMove:
    """Verify the wrapper toolset rewrites `run_code.description` and emits instructions."""

    async def test_description_becomes_static_prose(self) -> None:
        wrapper = _CatalogToolset(wrapped=_build_code_mode_toolset(_add))
        tools = await wrapper.get_tools(_build_run_context())

        description = tools[_RUN_CODE_TOOL_NAME].tool_def.description
        assert description is not None
        # The catalog (function signature) is gone from the description...
        assert 'async def _add' not in description
        # ...but the base prose is still there.
        assert 'sandboxed environment' in description

    async def test_catalog_surfaces_as_dynamic_instruction_part(self) -> None:
        wrapper = _CatalogToolset(wrapped=_build_code_mode_toolset(_add))
        ctx = _build_run_context()
        await wrapper.get_tools(ctx)
        instructions = await wrapper.get_instructions(ctx)

        # With no upstream instructions, the catalog is the only InstructionPart returned.
        assert isinstance(instructions, InstructionPart)
        assert 'async def _add' in instructions.content
        # `dynamic=True` so Anthropic/Bedrock place the cache breakpoint before this block.
        assert instructions.dynamic is True

    async def test_get_instructions_appends_to_upstream_string(self) -> None:
        """If the wrapped toolset returns an instructions string, the catalog appends to it."""

        class _UpstreamToolset(CodeModeToolset[None]):
            async def get_instructions(self, ctx: RunContext[None]) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
                return 'wrapped instructions'

        base = FunctionToolset[None](tools=[Tool(_add)])
        wrapper = _CatalogToolset(wrapped=_UpstreamToolset(wrapped=base, tool_selector='all'))
        ctx = _build_run_context()
        await wrapper.get_tools(ctx)
        instructions = await wrapper.get_instructions(ctx)

        assert isinstance(instructions, list)
        assert instructions[0] == 'wrapped instructions'
        assert isinstance(instructions[1], InstructionPart)
        assert 'async def _add' in instructions[1].content

    async def test_get_instructions_appends_to_upstream_sequence(self) -> None:
        """If the wrapped toolset returns a sequence, the catalog extends it."""

        class _UpstreamToolset(CodeModeToolset[None]):
            async def get_instructions(  # pyright: ignore[reportIncompatibleMethodOverride]
                self, ctx: RunContext[None]
            ) -> list[str | InstructionPart]:
                return ['a', InstructionPart(content='b')]

        base = FunctionToolset[None](tools=[Tool(_add)])
        wrapper = _CatalogToolset(wrapped=_UpstreamToolset(wrapped=base, tool_selector='all'))
        ctx = _build_run_context()
        await wrapper.get_tools(ctx)
        instructions = await wrapper.get_instructions(ctx)

        assert isinstance(instructions, list)
        assert instructions[0] == 'a'
        assert isinstance(instructions[1], InstructionPart) and instructions[1].content == 'b'
        # The catalog is appended at the end.
        assert isinstance(instructions[2], InstructionPart) and 'async def _add' in instructions[2].content

    async def test_no_run_code_in_chain_is_no_op(self) -> None:
        """When the wrapped toolset doesn't produce a `run_code` tool, nothing is rewritten."""
        base = FunctionToolset[None](tools=[Tool(_add)])
        wrapper = _CatalogToolset(wrapped=base)
        ctx = _build_run_context()

        tools = await wrapper.get_tools(ctx)
        assert _RUN_CODE_TOOL_NAME not in tools
        instructions = await wrapper.get_instructions(ctx)
        # No catalog stashed; upstream is None for FunctionToolset → wrapper returns None.
        assert instructions is None

    async def test_empty_catalog_leaves_description_alone(self) -> None:
        """A CodeModeToolset with no sandboxed tools has an empty catalog → no rewrite."""
        wrapper = _CatalogToolset(wrapped=_build_code_mode_toolset())
        ctx = _build_run_context()

        tools = await wrapper.get_tools(ctx)
        description = tools[_RUN_CODE_TOOL_NAME].tool_def.description
        assert description is not None
        # No catalog produced → no need to override the description.
        assert 'sandboxed environment' in description
        instructions = await wrapper.get_instructions(ctx)
        # Empty catalog → upstream (None for FunctionToolset).
        assert instructions is None

    async def test_for_run_step_preserves_catalog_stash(self) -> None:
        """Per-step rebuild must carry `_last_catalog` so instructions stay populated."""

        class _ChangingToolset(CodeModeToolset[None]):
            async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:  # pyright: ignore[reportIncompatibleMethodOverride]
                # Force `WrapperToolset.for_run_step` to take the `new_wrapped is not self.wrapped`
                # branch by returning a distinct (but equivalent) wrapped instance.
                new_wrapped = await self.wrapped.for_run_step(ctx)
                return type(self)(wrapped=new_wrapped, tool_selector=self.tool_selector)

        wrapper = _CatalogToolset(
            wrapped=_ChangingToolset(
                wrapped=FunctionToolset[None](tools=[Tool(_add)]),
                tool_selector='all',
            )
        )
        ctx = _build_run_context()
        await wrapper.get_tools(ctx)
        stashed = wrapper._last_catalog  # pyright: ignore[reportPrivateUsage]
        assert stashed  # populated

        # Step boundary clones the wrapper. The clone must keep the stash.
        new_wrapper = await wrapper.for_run_step(ctx)
        assert isinstance(new_wrapper, _CatalogToolset)
        assert new_wrapper is not wrapper
        assert new_wrapper._last_catalog == stashed  # pyright: ignore[reportPrivateUsage]

    async def test_for_run_step_returns_self_when_wrapped_unchanged(self) -> None:
        """If the wrapped toolset is unchanged across a step, the wrapper is the same instance."""
        wrapper = _CatalogToolset(wrapped=_build_code_mode_toolset(_add))
        ctx = _build_run_context()
        await wrapper.get_tools(ctx)

        same = await wrapper.for_run_step(ctx)
        assert same is wrapper


# ---------------------------------------------------------------------------
# Capability ordering + per-run state
# ---------------------------------------------------------------------------


class TestCapabilityShape:
    def test_ordering_requires_and_wraps_code_mode(self) -> None:
        ordering = CodeModeDynamicCatalog[None]().get_ordering()
        assert isinstance(ordering, CapabilityOrdering)
        # Without `requires`, the capability is silently a no-op.
        assert CodeMode in ordering.requires
        # `wraps` ensures the wrapper toolset sees CodeMode's assembled `run_code`.
        assert CodeMode in ordering.wraps
        # `position='outermost'` so CodeMode (also outermost) doesn't form a cycle with us.
        assert ordering.position == 'outermost'

    async def test_for_run_returns_fresh_state(self) -> None:
        cap = CodeModeDynamicCatalog[None]()
        cap._announced_tools.add('foo')  # pyright: ignore[reportPrivateUsage]
        fresh = await cap.for_run(_build_run_context())
        assert fresh is not cap
        assert fresh._announced_tools == set()  # pyright: ignore[reportPrivateUsage]

    def test_get_wrapper_toolset_returns_catalog_wrapper(self) -> None:
        base = FunctionToolset[None](tools=[Tool(_add)])
        wrapped = CodeModeDynamicCatalog[None]().get_wrapper_toolset(base)
        assert isinstance(wrapped, _CatalogToolset)
        assert wrapped.wrapped is base


# ---------------------------------------------------------------------------
# Discovery announcement — local path (`after_tool_execute`)
# ---------------------------------------------------------------------------


def _search_tool_def(name: str = 'search_tools') -> ToolDefinition:
    return ToolDefinition(name=name, description='', parameters_json_schema={}, tool_kind='tool-search')


def _other_tool_def() -> ToolDefinition:
    return ToolDefinition(name='unrelated', description='', parameters_json_schema={})


class TestLocalSearchAnnouncement:
    async def test_announce_on_local_search_return(self) -> None:
        cap = CodeModeDynamicCatalog[None]()
        ctx = _build_run_context()

        result = {'discovered_tools': [{'name': 'weather', 'description': '...'}]}
        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id='c1'),
            tool_def=_search_tool_def(),
            args={},
            result=result,
        )

        assert len(ctx.pending_messages) == 1
        request = ctx.pending_messages[0].payload
        assert isinstance(request, ModelRequest)
        [part] = request.parts
        assert isinstance(part, SystemPromptPart)
        assert '`weather`' in part.content

    async def test_announce_skipped_when_no_discoveries(self) -> None:
        cap = CodeModeDynamicCatalog[None]()
        ctx = _build_run_context()

        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id='c1'),
            tool_def=_search_tool_def(),
            args={},
            result={'discovered_tools': []},
        )
        assert ctx.pending_messages == []

    async def test_no_announce_for_non_search_tool(self) -> None:
        """`tool_kind != 'tool-search'` short-circuits before reading the result."""
        cap = CodeModeDynamicCatalog[None]()
        ctx = _build_run_context()

        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='add', args={}, tool_call_id='c1'),
            tool_def=_other_tool_def(),
            args={},
            # A non-search tool happens to return a `discovered_tools` shape — we still
            # don't announce: the tool_kind guard is the source of truth.
            result={'discovered_tools': [{'name': 'spurious'}]},
        )
        assert ctx.pending_messages == []

    async def test_no_duplicate_announcement_for_same_tool(self) -> None:
        """A second discovery of an already-announced tool does nothing."""
        cap = CodeModeDynamicCatalog[None]()
        ctx = _build_run_context()

        result = {'discovered_tools': [{'name': 'weather'}]}
        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id='c1'),
            tool_def=_search_tool_def(),
            args={},
            result=result,
        )
        await cap.after_tool_execute(
            ctx,
            call=ToolCallPart(tool_name='search_tools', args={}, tool_call_id='c2'),
            tool_def=_search_tool_def(),
            args={},
            result=result,
        )
        # Only one announcement for the (now-already-known) tool.
        assert len(ctx.pending_messages) == 1


# ---------------------------------------------------------------------------
# Discovery announcement — native path (`after_model_request`)
# ---------------------------------------------------------------------------


class TestNativeSearchAnnouncement:
    async def test_announce_on_native_search_return_part(self) -> None:
        cap = CodeModeDynamicCatalog[None]()
        ctx = _build_run_context()
        response = ModelResponse(
            parts=[
                NativeToolSearchReturnPart(
                    tool_name='tool_search',
                    content={'discovered_tools': [{'name': 'weather', 'description': 'Get the weather.'}]},
                    tool_call_id='c1',
                )
            ],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

        await cap.after_model_request(ctx, request_context=None, response=response)

        assert len(ctx.pending_messages) == 1
        request = ctx.pending_messages[0].payload
        assert isinstance(request, ModelRequest)
        [part] = request.parts
        assert isinstance(part, SystemPromptPart) and '`weather`' in part.content

    async def test_no_announce_for_unrelated_response_parts(self) -> None:
        cap = CodeModeDynamicCatalog[None]()
        ctx = _build_run_context()
        response = ModelResponse(
            parts=[
                TextPart('hi'),
                # A non-search NativeToolReturnPart should be ignored.
                NativeToolReturnPart(tool_name='whatever', content='ignored', tool_call_id='c1'),
            ],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

        await cap.after_model_request(ctx, request_context=None, response=response)
        assert ctx.pending_messages == []


# ---------------------------------------------------------------------------
# `_extract_discovered_names` — edge cases
# ---------------------------------------------------------------------------


class TestExtractDiscoveredNames:
    @pytest.mark.parametrize(
        ('content', 'expected'),
        [
            ('not a dict', []),
            ({}, []),
            ({'discovered_tools': 'not a list'}, []),
            ({'discovered_tools': [{'name': 'a'}, 'not a dict', {'no_name': 1}, {'name': 42}]}, ['a']),
        ],
    )
    def test_handles_malformed(self, content: Any, expected: list[str]) -> None:
        assert _extract_discovered_names(content) == expected


# ---------------------------------------------------------------------------
# End-to-end via `Agent.run` with `FunctionModel`
# ---------------------------------------------------------------------------


class TestAgentEndToEnd:
    async def test_agent_run_announces_discovery_and_lists_catalog_in_instructions(self) -> None:
        """`Agent.run` end-to-end: catalog in instructions, discovery enqueues announcement.

        Two-step run:
          1. Model asks for `search_tools(['weather'])` (the discovery surface).
          2. After the local tool-search returns, `CodeModeDynamicCatalog.after_tool_execute`
             enqueues a `SystemPromptPart` announcement; the pending message queue drains
             it into the next `ModelRequest`. The model sees the announcement and replies
             with the final text.
        """
        from pydantic_ai.capabilities import ToolSearch

        captured_system_prompts: list[list[str]] = []
        captured_request_descriptions: list[list[str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Snapshot the run_code description (should be static prose, no signatures)...
            run_code_def = next(td for td in info.function_tools if td.name == 'run_code')
            assert run_code_def.description is not None
            captured_request_descriptions.append([run_code_def.description])

            # ...and any system-prompt parts in the latest request (so we can verify
            # the announcement landed where we expected on turn 2).
            last_request = messages[-1]
            if isinstance(last_request, ModelRequest):
                captured_system_prompts.append(
                    [p.content for p in last_request.parts if isinstance(p, SystemPromptPart)]
                )

            # Turn 1: kick off a local tool-search.
            if len(captured_request_descriptions) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['weather']}, tool_call_id='c1')],
                    usage=RequestUsage(input_tokens=1, output_tokens=1),
                )
            # Turn 2+: the announcement has been drained into our request → reply.
            return ModelResponse(
                parts=[TextPart('done')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
            )

        # `defer_loading=True` keeps `weather` out of the eager catalog. ToolSearch's
        # local fallback (default strategy with no native support on FunctionModel) is
        # what fires the discovery.
        def weather(city: str) -> str:
            """Get the weather."""
            return f'sunny in {city}'  # pragma: no cover — only the signature matters.

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            tools=[Tool(weather, defer_loading=True)],
            capabilities=[ToolSearch[None](), CodeMode[None](), CodeModeDynamicCatalog[None]()],
        )

        result = await agent.run('please find a weather tool')

        # `run_code.description` stayed static across both turns — no `async def`
        # signature anywhere in the tool-defs block.
        assert all('async def' not in d for descs in captured_request_descriptions for d in descs)

        # Discovery announcement landed in turn 2's request as a `SystemPromptPart`.
        # (Turn 1's request had no system prompt parts from us.)
        assert len(captured_system_prompts) >= 2
        announcement_on_turn_2 = '\n'.join(captured_system_prompts[1])
        assert 'weather' in announcement_on_turn_2

        # The local `ToolSearchReturnPart` is in history; sanity-check that.
        history = result.all_messages()
        assert any(
            isinstance(p, ToolSearchReturnPart) for msg in history if isinstance(msg, ModelRequest) for p in msg.parts
        )
        # And the `add`/result tool return path ran cleanly.
        assert any(
            isinstance(p, ToolReturnPart) and p.tool_name == 'search_tools'
            for msg in history
            if isinstance(msg, ModelRequest)
            for p in msg.parts
        )
        assert result.output == 'done'
