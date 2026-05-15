# Code Mode -- Dynamic Tool Catalog

Cache-friendly tool catalog disclosure for [`CodeMode`](../code_mode/README.md).

## The problem

Out of the box, `CodeMode` renders every sandboxed tool's signature into the `run_code` tool's `description` string. That description lives in the model API's *tool-definitions* block — the part of the prompt that's keyed on byte equality by Anthropic, OpenAI, and Google prompt caching. Whenever the catalog changes (e.g. [Tool Search](https://ai.pydantic.dev/tools-advanced/#tool-search) reveals a new tool, or a per-step toolset swap), the description changes → the cache prefix is invalidated from that point forward.

The bigger structural issue is that the tool-definitions block isn't designed to be re-keyed dynamically. There's no separation between "stable preamble" and "discovery-driven addenda", and the catalog also duplicates whatever lives in the model's `tools[]` array on the wire.

## The solution

`CodeModeDynamicCatalog` reshapes the disclosure surface:

1. **`run_code.description` becomes static.** Only the base prose (sandbox restrictions, return-value contract) is left. The tool-defs block stays byte-stable across discoveries.
2. **The "available functions" catalog moves into instructions** as a dynamic [`InstructionPart`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.InstructionPart). Providers that split static vs. dynamic instructions (Anthropic, Bedrock) place a cache breakpoint *before* this block, so the static instruction prefix survives discoveries.
3. **Newly-discovered tools are announced via [`RunContext.enqueue`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext.enqueue).** A short `SystemPromptPart` lands in the next request's history, telling the model "these new functions are now callable from `run_code`". Append-only — doesn't touch any cached prefix.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai_harness import CodeMode, CodeModeDynamicCatalog

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[ToolSearch(), CodeMode(), CodeModeDynamicCatalog()],
)
```

The capability requires `CodeMode` (it's a no-op without it) and sequences itself outermost so its wrapper toolset sees `CodeMode`'s assembled `run_code` tool.

## When to use it

| Scenario | Default `CodeMode` | `+ CodeModeDynamicCatalog` |
|---|---|---|
| Fixed toolset, no `ToolSearch` | Catalog in `run_code.description`. Cache-stable. | Catalog in instructions. Cache-stable. Slightly more verbose prompt. |
| `ToolSearch` + many deferred tools | Each discovery rebuilds `run_code.description` → tool-defs cache busts every time. | `run_code.description` never changes. Only the dynamic instructions / pending-message announcements grow. |
| Per-step toolset swaps | Description churns with the toolset. | Description stays static; only the instructions catalog reflects the swap. |

The break-even point is roughly *"do you have a `ToolSearch` corpus or a churning toolset?"* — if yes, this saves a cache bust per discovery / swap. If no, default `CodeMode` is fine and slightly cheaper on first-turn prompt size.

## Related

- [`CodeMode`](../code_mode/README.md) — required companion.
- [Pydantic AI message history — injecting messages mid-run](https://ai.pydantic.dev/message-history/#injecting-messages-mid-run) — the [`enqueue`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext.enqueue) primitive announcements ride on.
- [pydantic-ai-harness#232](https://github.com/pydantic/pydantic-ai-harness/issues/232) — Tier-1 fix that makes deferred-loading tools cooperate with `CodeMode`; this capability is the Tier-2 reshape.
- [pydantic/pydantic-ai#4980](https://github.com/pydantic/pydantic-ai/pull/4980) — the pending message queue PR.
- [pydantic/pydantic-ai#5437](https://github.com/pydantic/pydantic-ai/issues/5437) — mid-conversation `SystemPromptPart` mapping fix, which makes the discovery announcement fully cache-safe across providers.

## Limitations

- **Native search**: announcements for tools discovered via server-side `tool_search` (Anthropic BM25 / regex, OpenAI Responses) land via `after_model_request`, on the same turn as the response containing the search return. Tools discovered via local `search_tools` (the fallback path) are announced via `after_tool_execute`, also one step earlier than the user's next turn.
- **Cache safety across providers**: until pydantic/pydantic-ai#5437 lands, the announcement enqueues a `SystemPromptPart` in a `ModelRequest`. OpenAI Chat/Responses inline it at position (cache-safe). Anthropic and Google hoist it to the top-level system block, which busts the prefix cache. The win still applies on the `run_code.description` side (it's no longer churning), but the announcement itself isn't free on those providers yet. Once #5437 lands, mid-conversation system parts will be rendered as XML-wrapped user prompts on the affected providers, fully closing the loop.
- **Catalog placement**: rendered as a single `InstructionPart(dynamic=True)`. Providers without a static/dynamic instruction split (most non-Anthropic/Bedrock) see it as plain appended instructions; that's still better than the description-rebuild because the *tool-defs* block stays cache-stable.

## Agent spec (YAML/JSON)

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-5
capabilities:
  - ToolSearch: {}
  - CodeMode: {}
  - CodeModeDynamicCatalog: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode, CodeModeDynamicCatalog

agent = Agent.from_file('agent.yaml', custom_capability_types=[CodeMode, CodeModeDynamicCatalog])
```
