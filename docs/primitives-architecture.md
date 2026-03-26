# Pydantic Harness: Primitives Architecture

> Design document for the core primitives that pydantic-harness must provide so that
> **capabilities do the heavy lifting** — inspired by pi-mono's thin-core/fat-extension
> philosophy, grounded in what pydantic-ai already gives us, and informed by the
> production patterns in Hermes Agent.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [What Pydantic-AI Already Provides](#2-what-pydantic-ai-already-provides)
3. [The Primitives We Need to Build](#3-the-primitives-we-need-to-build)
4. [Primitive 1: Context Window Awareness](#4-primitive-1-context-window-awareness)
5. [Primitive 2: Message Compaction](#5-primitive-2-message-compaction)
6. [Primitive 3: Persistent Memory](#6-primitive-3-persistent-memory)
7. [Primitive 4: Session Management](#7-primitive-4-session-management)
8. [Primitive 5: Approval & Permissions](#8-primitive-5-approval--permissions)
9. [Primitive 6: Platform Gateway](#9-primitive-6-platform-gateway)
10. [Primitive 7: Sub-Agent Orchestration](#10-primitive-7-sub-agent-orchestration)
11. [Primitive 8: Skill Discovery & Registration](#11-primitive-8-skill-discovery--registration)
12. [Primitive 9: Run Control & Limits](#12-primitive-9-run-control--limits)
13. [Primitive 10: Configuration Resolution](#13-primitive-10-configuration-resolution)
14. [Primitive 11: User Hooks](#14-primitive-11-user-hooks)
15. [Primitive 12: Display Adapter](#15-primitive-12-display-adapter)
16. [How Primitives Compose](#16-how-primitives-compose)
17. [Implementation Priority](#17-implementation-priority)
18. [Appendix: Comparison Matrix](#appendix-comparison-matrix)

---

## 1. Design Philosophy

### The pi-mono Lesson

pi-mono's architecture achieves "extensions do the heavy lifting" through three principles:

1. **Thin core, fat extensions.** The framework provides an agent loop, an LLM abstraction, and basic tools. Everything else — custom tools, providers, memory, compaction, platform integration — is an extension that hooks into 31 well-defined events.

2. **Inversion of control via events.** The core *calls* hooks; extensions *define* behavior. Extensions never import each other — they all go through the framework. This gives zero-coupling and graceful degradation.

3. **Registration, not inheritance.** Extensions register tools, commands, providers, and renderers at runtime via a factory function that receives the extension API. Late binding means new capabilities don't require rebuilding the core.

### How This Maps to Pydantic-AI

Pydantic-AI's `AbstractCapability` system is the closest analogue to pi-mono's extension API, though the mapping is approximate — pi-mono uses event-based composition with priority ordering, while pydantic-ai uses hook-based composition with registration order:

| pi-mono Extension API | pydantic-ai Nearest Equivalent | Notes |
|---|---|---|
| `pi.on("before_agent_start", ...)` | `capability.before_run(ctx)` | pi-mono fires after prompt submission, before agent loop; pydantic-ai fires before the entire run |
| `pi.on("context", ...)` | `capability.before_model_request(ctx, request_context)` | Both allow message mutation before LLM call |
| `pi.on("tool_call", ...)` | `capability.before_tool_execute(ctx, ...)` | Direct analogue — can block/modify tool calls |
| `pi.on("tool_result", ...)` | `capability.after_tool_execute(ctx, ...)` | Direct analogue — can modify tool results |
| `pi.registerTool(...)` | `capability.get_toolset()` | Direct analogue |
| `pi.on("session_before_compact", ...)` | No direct equivalent — must be implemented as a primitive | pydantic-ai has no compaction event; `wrap_model_request` is a general request wrapper, not compaction-specific |
| CombinedCapability | CombinedCapability (already exists) | pi-mono uses event priority; pydantic-ai uses registration order for `before_*`, reverse for `after_*`/`wrap_*` |

**The key insight:** pydantic-ai already has the extension mechanism (capabilities). What it lacks are the **domain primitives** — the protocols, base classes, and state containers that capabilities need to do their work. That's what pydantic-harness must provide.

### Design Rules

1. **Primitives are protocols and base classes, not capabilities.** A primitive defines the *contract* (e.g., `MemoryStore` protocol). A capability uses the primitive (e.g., `MemoryCapability` wraps a `MemoryStore` and hooks into the agent lifecycle).

2. **Each primitive should be independently useful.** You can use `CompactionEngine` without `MemoryStore`. You can use `PlatformAdapter` without `SessionManager`. Composition is opt-in via capabilities.

3. **Provide one reference implementation per primitive.** The primitive is the protocol; the reference impl proves it works. Community builds alternatives.

4. **State lives in the primitive, not the capability.** Capabilities are ephemeral (created per-run via `for_run()`). Primitives persist across runs and sessions.

5. **No framework lock-in.** Every primitive should be usable outside of pydantic-ai. A `MemoryStore` is just a class with `load()` and `save()`. The capability layer is the only thing that knows about pydantic-ai.

---

## 2. What Pydantic-AI Already Provides

These are primitives we **do not need to build** — they exist in pydantic-ai and are sufficient:

### Agent Loop (95% sufficient)
- Graph-based execution: `UserPromptNode → ModelRequestNode → CallToolsNode → End`
- Manual stepping via `AgentRun.next(node)` for fine-grained control
- `GraphAgentState` tracks: messages, usage, retries, run_step, run_id, metadata

### Capability System (95% sufficient)
- `AbstractCapability` with 25 lifecycle hooks across 4 phases:
  - **Run lifecycle** (5): `before_run`, `after_run`, `wrap_run`, `on_run_error`, `for_run`
  - **Node lifecycle** (4): `before_node_run`, `after_node_run`, `wrap_node_run`, `on_node_run_error`
  - **Model request** (5): `before_model_request`, `after_model_request`, `wrap_model_request`, `on_model_request_error`, `wrap_run_event_stream`
  - **Tool processing** (8): `before/after/wrap/on_error` for both `tool_validate` and `tool_execute`
  - Plus: `prepare_tools`, `get_wrapper_toolset`
- `CombinedCapability` for composition (before→forward, after→LIFO, wrap→middleware)
- `for_run(ctx)` for per-run state isolation (async, returns self by default, override for fresh instances)
- Configuration methods: `get_toolset()`, `get_instructions()`, `get_model_settings()`, `get_builtin_tools()`

### Toolset Abstractions (95% sufficient)
- `AbstractToolset` with `get_tools()`, `call_tool()`, visitor pattern
- Fluent composition: `.filtered()`, `.prefixed()`, `.prepared()`, `.renamed()`, `.approval_required()`
- `FunctionToolset`, `CombinedToolset`, `WrapperToolset`, `FastMCPToolset`
- Async context manager lifecycle (`__aenter__`/`__aexit__`)

### Model Layer (90% sufficient)
- `Model` ABC with `request()`, `request_stream()`, `count_tokens()`
- `ModelProfile` with feature flags (supports_tools, supports_thinking, etc.)
- `FallbackModel` with exception/handler/response-based triggers
- 20+ providers, 400+ model variants, Anthropic prompt caching support

### Run Context (85% sufficient)
- `RunContext` passed everywhere: deps, model, usage, messages, run_step, retry, tool_call_id
- `get_current_run_context()` context variable for implicit access
- `UsageLimits`: request_limit (default 50), token limits, pre-request counting

### Tool Approval (partial)
- `Tool.requires_approval` exists for per-tool approval gating
- `tool_call_approved` field on `RunContext`
- `DeferredToolRequests` output type for external approval flows
- **Missing:** No `ApprovalPolicy` protocol, no persistent approval storage, no pattern-based detection

### What's Missing from pydantic-ai Core
- `context_window` on `ModelProfile` (exists on `context-window-models` branch, not merged)
- `context_window_used` property on `RunContext` (same branch)
- Max-turns / stuck detection in the agent loop
- No built-in message history persistence
- No built-in context compression
- No memory system
- No platform/gateway abstraction
- No approval policy / permission management beyond per-tool flags
- No configuration file discovery
- No user-defined hooks (shell commands triggered by events)
- No display/output rendering abstraction

---

## 3. The Primitives We Need to Build

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                │
│  (User code: Agent(..., capabilities=[...]))                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                         CAPABILITY LAYER                                    │
│  CompactionCapability │ MemoryCapability │ ApprovalCapability │ ...         │
│  (hooks into agent lifecycle, composes primitives)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                          PRIMITIVES LAYER                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ ContextWindow │ │ Compaction   │ │ Memory       │ │ Approval     │      │
│  │ Tracker       │ │ Engine       │ │ Store        │ │ Policy       │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Session      │ │ Platform     │ │ SubAgent     │ │ Skill        │      │
│  │ Store        │ │ Adapter      │ │ Spawner      │ │ Registry     │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ RunControl   │ │ Config       │ │ UserHook     │ │ Display      │      │
│  │ Policy       │ │ Resolver     │ │ Runner       │ │ Adapter      │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
├─────────────────────────────────────────────────────────────────────────────┤
│                        PYDANTIC-AI CORE                                     │
│  Agent │ AbstractCapability │ AbstractToolset │ Model │ RunContext          │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each primitive below is specified as:
- **Protocol** — the abstract contract
- **Reference Implementation** — what we ship
- **Capability** — the pydantic-ai integration layer
- **Extension Points** — where community/users customize

---

## 4. Primitive 1: Context Window Awareness

### Problem
Capabilities like compaction, memory injection, and sub-agent spawning all need to know how much of the context window is consumed and how much remains. Today, pydantic-ai tracks `UsageLimits` (request/token ceilings) but not context window *utilization*.

### Protocol

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ContextBudget:
    """Snapshot of context window utilization."""
    context_window: int          # Total window size in tokens
    system_prompt_tokens: int    # Tokens consumed by system prompt
    message_tokens: int          # Tokens consumed by message history
    pending_tokens: int          # Estimated tokens for next response
    available_tokens: int        # Remaining budget

    @property
    def utilization(self) -> float:
        """Fraction of context window consumed (0.0 to 1.0)."""
        return 1.0 - (self.available_tokens / self.context_window)


class ContextWindowTracker(Protocol):
    """Tracks context window utilization across the agent run."""

    def estimate(
        self,
        messages: list[ModelMessage],
        system_prompt: str | None = None,
    ) -> ContextBudget:
        """Estimate current context utilization from messages."""
        ...

    def should_compact(self, budget: ContextBudget, threshold: float = 0.5) -> bool:
        """Whether compaction should be triggered."""
        ...

    @property
    def context_window(self) -> int:
        """Total context window size for the current model."""
        ...
```

### Reference Implementation

```python
class TokenCountingTracker:
    """Context tracker using rough character-based estimation.

    Uses chars/4 as default heuristic (matches Hermes's proven
    _CHARS_PER_TOKEN = 4 constant). When the model supports
    count_tokens(), uses actual token counts for system prompt
    (cached across turns) and estimates for messages.
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, context_window: int, *, model: Model | None = None):
        self._context_window = context_window
        self._model = model
        self._system_prompt_tokens: int | None = None

    def estimate(self, messages, system_prompt=None) -> ContextBudget:
        # Cache system prompt token count (stable across turns)
        if system_prompt and self._system_prompt_tokens is None:
            self._system_prompt_tokens = len(system_prompt) // self.CHARS_PER_TOKEN

        sys_tokens = self._system_prompt_tokens or 0
        msg_tokens = self._estimate_messages(messages)
        pending = self._estimate_next_response(messages)
        available = max(0, self._context_window - sys_tokens - msg_tokens - pending)

        return ContextBudget(
            context_window=self._context_window,
            system_prompt_tokens=sys_tokens,
            message_tokens=msg_tokens,
            pending_tokens=pending,
            available_tokens=available,
        )
```

### Dependency
This primitive **requires** `context_window` on `ModelProfile` (the `context-window-models` branch). Landing that branch is a prerequisite.

### Extension Points
- Swap in a `model.count_tokens()`-based implementation for exact counts
- Custom `should_compact()` thresholds per use case
- Provider-specific estimation (Anthropic cache-aware counting)

---

## 5. Primitive 2: Message Compaction

### Problem
Long-running agents exhaust their context window. Compaction replaces old messages with a structured summary while preserving critical context. This is the single most important primitive for production agents.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass


@dataclass
class CompactionResult:
    """Output of a compaction operation."""
    summary_message: ModelMessage     # The summary to inject
    surviving_messages: list[ModelMessage]  # Messages that were kept (tail)
    tokens_freed: int                 # Estimated tokens recovered
    messages_compressed: int          # Number of messages summarized
    previous_summary: str | None      # Prior summary (for iterative updates)


class CompactionEngine(Protocol):
    """Compresses message history into summaries."""

    async def compact(
        self,
        messages: list[ModelMessage],
        *,
        context_budget: ContextBudget,
        previous_summary: str | None = None,
        system_prompt: str | None = None,
    ) -> CompactionResult:
        """Compress messages to fit within budget.

        Args:
            messages: Full message history.
            context_budget: Current context window state.
            previous_summary: Summary from a prior compaction (for iterative updates).
            system_prompt: Current system prompt (excluded from compression).
        """
        ...
```

### Reference Implementation: Four-Phase Compaction

Directly modeled on Hermes's proven production pattern (which uses four phases, not three — the orphan repair phase is critical and non-trivial):

```python
class FourPhaseCompactionEngine:
    """
    Phase 1: Tool Output Pruning (no LLM, cheap)
        Walk backward from end. Old tool results with content >200 chars
        get replaced with "[Old tool output cleared to save context space]".
        Protects the most recent N messages.

    Phase 2: Boundary Alignment (no LLM, preserves structure)
        Align compression boundaries to tool call/result group edges.
        Never split a tool_call from its tool_result. Determine head
        protection (first N exchanges) and tail protection (most recent
        messages by token budget).

    Phase 3: Structured Summarization (LLM call)
        First compaction: generate structured summary with sections:
            Goal, Constraints, Progress (Done/InProgress/Blocked),
            Key Decisions, Relevant Files, Next Steps, Critical Context.
        Iterative compaction: update previous summary with new progress.
        Budget: 20% of compressed content tokens, min 2K, max 12K.

    Phase 4: Assembly & Orphan Repair (no LLM, structural)
        Rebuild the message list from [summary + surviving tail].
        Repair orphaned tool_call/result pairs:
        - Remove tool results with no matching call
        - Insert stub results for calls with no matching result
        This phase is critical — LLM providers reject malformed
        tool_call/result sequences.
    """

    def __init__(
        self,
        *,
        summarizer_model: Model | str = "gemini-2.5-flash",
        protect_first_n: int = 3,
        protect_tail_ratio: float = 0.3,
        summary_target_ratio: float = 0.2,
        max_summary_tokens: int = 12_000,
        min_summary_tokens: int = 2_000,
    ): ...
```

### Capability: CompactionCapability

```python
class CompactionCapability(AbstractCapability[DepsT]):
    """Hooks compaction into the agent lifecycle.

    Triggers compaction via before_model_request when context
    utilization exceeds threshold. Uses HistoryProcessor hook
    to rewrite message history with summary.
    """

    def __init__(
        self,
        engine: CompactionEngine | None = None,    # Default: FourPhaseCompactionEngine
        tracker: ContextWindowTracker | None = None,
        threshold: float = 0.5,
        *,
        pre_compact_flush: Callable | None = None,  # e.g., flush memories before compression
    ): ...

    async def before_model_request(self, ctx, request_context):
        """Check utilization, trigger compaction if needed."""
        budget = self._tracker.estimate(ctx.messages, ...)
        if self._tracker.should_compact(budget, self._threshold):
            result = await self._engine.compact(
                ctx.messages,
                context_budget=budget,
                previous_summary=self._last_summary,
            )
            # Replace message history with summary + surviving tail
            request_context.messages[:] = [
                result.summary_message,
                *result.surviving_messages,
            ]
            self._last_summary = result.previous_summary
```

### Extension Points
- Custom `CompactionEngine` implementations (e.g., embedding-based, topic-segmented)
- Custom summary templates for domain-specific agents
- Custom summarizer model selection
- `pre_compact_flush` callback for memory persistence before summarization
- Custom boundary alignment strategies

### Critical Design: Tool Pair Integrity (Phase 4)

Compaction must never create orphaned tool call/result pairs. Phase 4 must:
1. Collect all surviving `tool_call` IDs from assistant messages
2. Collect all `tool_result` call IDs
3. Remove tool results with no matching call
4. Insert stub results for calls with no matching result:
   `"[Result from earlier conversation — see context summary above]"`

This is not a nice-to-have — LLM providers (Anthropic, OpenAI) will reject requests with malformed tool sequences.

---

## 6. Primitive 3: Persistent Memory

### Problem
Agents need to accumulate knowledge across sessions — user preferences, project conventions, learned facts. Memory must survive session boundaries and context compactions.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """A single memory record."""
    key: str                    # Unique identifier
    content: str                # Memory content
    category: str = "general"   # Categorization (user, project, feedback, reference)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryStore(Protocol):
    """Persistent, cross-session memory storage."""

    async def load(self) -> list[MemoryEntry]:
        """Load all memories from storage."""
        ...

    async def save(self, entry: MemoryEntry) -> None:
        """Create or update a memory entry."""
        ...

    async def remove(self, key: str) -> bool:
        """Remove a memory entry by key. Returns True if found."""
        ...

    async def search(self, query: str, *, limit: int = 10) -> list[MemoryEntry]:
        """Search memories by relevance to query."""
        ...

    def format_for_prompt(self) -> str:
        """Format memories for injection into system prompt.

        Returns a snapshot string. This snapshot should be frozen
        at session start and not mutated mid-session (preserves
        prefix cache for Anthropic models).
        """
        ...
```

### Reference Implementation: File-Backed Memory

Inspired by Hermes's `MEMORY.md` / `USER.md` pattern (character-bounded entries with delimiter-based deduplication):

```python
class FileMemoryStore:
    """Character-limited, file-backed memory with frozen snapshots.

    Stores memories as structured Markdown files in a configurable
    directory. Entries are delimited and deduplicated on load.
    format_for_prompt() returns a snapshot frozen at first call
    (preserves prefix cache). Subsequent writes update disk but
    not the cached snapshot — the snapshot refreshes on next session.

    Layout:
        {base_dir}/
            MEMORY.md          # Agent observations (project, conventions)
            USER.md            # User preferences, communication style
            {category}.md      # Extensible categories
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        max_chars_per_file: int = 2200,
        delimiter: str = "§",
    ): ...
```

### Capability: MemoryCapability

```python
class MemoryCapability(AbstractCapability[DepsT]):
    """Integrates memory into the agent lifecycle.

    - Injects frozen memory snapshot into system prompt via get_instructions()
    - Provides memory tools (add/replace/remove/search) via get_toolset()
    - Flushes pending writes before compaction
    """

    def __init__(
        self,
        store: MemoryStore | None = None,  # Default: FileMemoryStore
        *,
        inject_in_prompt: bool = True,     # Include snapshot in system prompt
        provide_tools: bool = True,         # Give agent memory read/write tools
        auto_flush_on_compact: bool = True, # Flush before compaction
    ): ...

    def get_instructions(self) -> str:
        """Return frozen memory snapshot + behavioral guidance."""
        return self._store.format_for_prompt() + MEMORY_GUIDANCE

    def get_toolset(self) -> AbstractToolset:
        """Return memory tools: memory_add, memory_replace, memory_remove, memory_search."""
        return self._memory_toolset
```

### Extension Points
- **SQLite FTS5 store** — full-text search across session transcripts (like Hermes's `messages_fts` table)
- **Embedding-based store** — vector similarity search (e.g., ChromaDB, pgvector)
- **Honcho integration** — dialectic user modeling with LLM-powered synthesis (like Hermes's `HonchoSessionManager`)
- **Redis/cloud store** — for multi-instance deployments
- Custom `format_for_prompt()` for domain-specific memory layouts
- Custom memory categories beyond the defaults

### Critical Design: Frozen Snapshot Pattern

Memory content injected into the system prompt **must be frozen at session start**. This is not optional — it directly affects Anthropic prefix cache hit rates:

```
Session start → load memories → snapshot → inject in system prompt
                                    ↓
                              FROZEN for all turns
                                    ↓
Mid-session memory writes → update disk → NOT reflected in prompt
                                    ↓
Next session start → reload → new snapshot
```

Compaction is the one exception: when messages are compressed, the system prompt is rebuilt and the memory snapshot is refreshed from disk.

**Note:** This frozen snapshot pattern applies broadly — skills, configuration, and any other content injected into the system prompt should follow the same discipline. Consider extracting a general `FrozenPromptContent` protocol rather than baking the pattern into each capability independently.

---

## 7. Primitive 4: Session Management

### Problem
Production agents need to persist conversation state across process restarts, track session lineage across compactions, and resolve session identity from diverse contexts (CLI, messaging platforms, API).

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionMetadata:
    """Metadata about a session."""
    session_id: str
    title: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    parent_session_id: str | None = None  # Lineage across compactions
    source: str | None = None              # "cli", "discord", "api", etc.
    model: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


class SessionStore(Protocol):
    """Persistent session storage."""

    async def create(self, metadata: SessionMetadata) -> str:
        """Create a new session. Returns session_id."""
        ...

    async def save_messages(
        self, session_id: str, messages: list[ModelMessage]
    ) -> None:
        """Persist messages for a session (incremental append)."""
        ...

    async def load_messages(self, session_id: str) -> list[ModelMessage]:
        """Load all messages for a session."""
        ...

    async def end(self, session_id: str, *, reason: str = "normal") -> None:
        """Mark a session as ended."""
        ...

    async def get_metadata(self, session_id: str) -> SessionMetadata | None:
        """Get session metadata."""
        ...

    async def list_sessions(
        self, *, limit: int = 50, source: str | None = None
    ) -> list[SessionMetadata]:
        """List recent sessions."""
        ...

    async def search(
        self, query: str, *, limit: int = 10
    ) -> list[tuple[SessionMetadata, list[ModelMessage]]]:
        """Full-text search across session transcripts."""
        ...


class SessionResolver(Protocol):
    """Resolves session identity from diverse contexts."""

    def resolve(self, context: dict[str, Any]) -> str:
        """Determine session key from context.

        Context may include: platform, user_id, channel_id,
        directory, repo_root, explicit session name, etc.

        Resolution strategies:
        - per-session: unique ID per conversation
        - per-directory: directory basename
        - per-repo: git repo root name
        - per-user: user ID on platform
        - global: single session
        """
        ...
```

### Reference Implementation

```python
class SQLiteSessionStore:
    """SQLite-backed session store with FTS5 search.

    Tables:
        sessions(id, title, created_at, ended_at, parent_id, source, model, tags_json)
        messages(id, session_id, role, content_json, timestamp, token_estimate)
        messages_fts(content) USING fts5  -- for full-text search

    Features:
        - WAL mode for concurrent access
        - Incremental message append (not full rewrite)
        - Parent session linkage for compaction lineage
        - Title propagation with auto-numbering across compressions
        - FTS5 indexing with automatic triggers for insert/update/delete sync
        - Foreign key constraint: parent_session_id REFERENCES sessions(id)
        - Index on parent_session_id for lineage queries
    """
```

### Capability: SessionCapability

```python
class SessionCapability(AbstractCapability[DepsT]):
    """Session persistence integrated into agent lifecycle.

    - Loads message history on run start (before_run)
    - Saves messages incrementally (after_model_request)
    - Handles compaction-triggered session splits
    - Provides session tools (search, list, switch)
    """
```

### Extension Points
- PostgreSQL/Redis store for multi-instance deployments
- Custom session resolvers for platform-specific logic
- Session branching (fork/navigate tree, like pi-mono's JSONL with id/parentId tree structure)
- Session export/import (JSON, OpenAI format)

---

## 8. Primitive 5: Approval & Permissions

### Problem
Production agents execute tools with real-world side effects — running shell commands, writing files, making API calls, sending messages. Every production agent (Hermes, Claude Code, pi-mono) has a permission/approval system that gates dangerous operations. pydantic-ai has `Tool.requires_approval` but no policy framework for *how* approval decisions are made, cached, or persisted.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass
from enum import Enum


class ApprovalDecision(Enum):
    ALLOW = "allow"              # Proceed with execution
    DENY = "deny"                # Block execution
    ASK = "ask"                  # Prompt the user for approval


class ApprovalScope(Enum):
    ONCE = "once"                # This invocation only
    SESSION = "session"          # Remainder of this session
    PERMANENT = "permanent"      # Persist across sessions


@dataclass
class ApprovalRequest:
    """Information about a tool call requiring approval."""
    tool_name: str
    args: dict[str, Any]
    description: str | None = None    # Human-readable description of what the tool will do
    risk_level: str = "normal"        # "safe", "normal", "dangerous", "destructive"


class ApprovalPolicy(Protocol):
    """Evaluates whether a tool call should be allowed, denied, or escalated."""

    async def check(
        self,
        request: ApprovalRequest,
        *,
        session_id: str | None = None,
    ) -> ApprovalDecision:
        """Evaluate whether this tool call should proceed.

        Implementations may check:
        - Permanent allowlists/blocklists
        - Session-level approvals already granted
        - Pattern-based detection of destructive operations
        - Tool-specific risk assessment
        """
        ...

    async def record(
        self,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        scope: ApprovalScope,
    ) -> None:
        """Record an approval decision for future lookups."""
        ...


class DangerousCommandDetector(Protocol):
    """Detects dangerous operations within tool arguments."""

    def classify(self, tool_name: str, args: dict[str, Any]) -> str:
        """Classify the risk level of a tool call.

        Returns: "safe", "normal", "dangerous", "destructive"

        Examples of destructive: rm -rf, git push --force, DROP TABLE
        Examples of dangerous: git reset, file overwrites, pip install
        Examples of safe: read-only operations, git status, ls
        """
        ...
```

### Reference Implementation

```python
class PatternBasedApprovalPolicy:
    """Pattern-matching approval policy with persistent allowlists.

    Features:
        - Regex-based pattern matching for command classification
        - Session-level approval caching (approve once per session)
        - Permanent allowlist persisted to ~/.pydantic-harness/approvals.json
        - Configurable approval modes: ask_always, ask_first, auto_approve
        - Built-in patterns for common destructive operations:
          rm -rf, git push --force, DROP TABLE, kill -9, etc.

    Inspired by Hermes's tools/approval.py pattern-based detection
    with session-level and permanent approval storage.
    """

    def __init__(
        self,
        *,
        default_mode: str = "ask_first",      # ask_always, ask_first, auto_approve
        allowlist_path: Path | None = None,     # Persistent allowlist
        dangerous_patterns: list[str] | None = None,  # Additional regex patterns
    ): ...
```

### Capability: ApprovalCapability

```python
class ApprovalCapability(AbstractCapability[DepsT]):
    """Integrates approval policy into the agent lifecycle.

    - Intercepts tool calls via before_tool_execute
    - Classifies risk level via DangerousCommandDetector
    - Delegates approval decision to ApprovalPolicy
    - Prompts user via callback when ASK decision is returned
    - Records decisions for session/permanent caching
    """

    def __init__(
        self,
        policy: ApprovalPolicy | None = None,
        detector: DangerousCommandDetector | None = None,
        *,
        approval_callback: Callable[[ApprovalRequest], Awaitable[tuple[ApprovalDecision, ApprovalScope]]] | None = None,
    ): ...

    async def before_tool_execute(self, ctx, tool_call, ...):
        """Check approval before tool execution."""
        request = ApprovalRequest(
            tool_name=tool_call.name,
            args=tool_call.args,
            risk_level=self._detector.classify(tool_call.name, tool_call.args),
        )
        decision = await self._policy.check(request, session_id=ctx.run_id)
        if decision == ApprovalDecision.ASK:
            decision, scope = await self._approval_callback(request)
            await self._policy.record(request, decision, scope)
        if decision == ApprovalDecision.DENY:
            raise ToolExecutionDenied(request)
```

### Extension Points
- Custom risk classifiers for domain-specific operations
- Integration with external authorization systems (RBAC, OAuth scopes)
- Approval audit logging
- Rate-limiting policies (max N tool calls per minute)
- Per-user approval policies in multi-user gateway deployments

---

## 9. Primitive 6: Platform Gateway

### Problem
Agents need to operate on messaging platforms (Discord, Telegram, Slack, WhatsApp) where the interaction model is fundamentally different from HTTP request/response. Each platform has different APIs, message formats, rate limits, media handling, and connection types.

### Protocol

```python
from typing import Protocol, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MessageType(Enum):
    TEXT = "text"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    LOCATION = "location"
    COMMAND = "command"


@dataclass
class InboundMessage:
    """Normalized incoming message from any platform."""
    text: str
    message_type: MessageType = MessageType.TEXT
    platform: str = ""                     # "discord", "telegram", "slack", etc.
    user_id: str = ""                      # Platform-specific user identifier
    channel_id: str = ""                   # Platform-specific channel/chat identifier
    message_id: str | None = None
    media_paths: list[str] = field(default_factory=list)  # Local cached paths
    media_types: list[str] = field(default_factory=list)
    reply_to_id: str | None = None
    reply_to_text: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)  # Platform-specific extras


@dataclass
class OutboundMessage:
    """Message to send back to the platform."""
    text: str
    reply_to_id: str | None = None         # Platform message to reply to
    media_path: str | None = None          # File to attach
    media_type: str | None = None          # MIME type
    parse_mode: str | None = None          # "markdown", "html", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    message_id: str | None = None
    error: str | None = None


class PlatformAdapter(Protocol):
    """Adapts a messaging platform for agent interaction.

    Lifecycle:
        1. connect() — authenticate and establish connection
        2. messages() — async iterator of incoming messages
        3. send() — send responses back
        4. disconnect() — clean shutdown

    Adapters handle platform-specific concerns:
        - Authentication (bot tokens, OAuth, API keys)
        - Connection management (WebSocket, long-polling, webhooks)
        - Message normalization (platform format → InboundMessage)
        - Media handling (download, cache locally, return paths)
        - Rate limiting and message splitting
        - Typing indicators and read receipts
        - Streaming responses via message editing
    """

    @property
    def platform_name(self) -> str:
        """Identifier for this platform (e.g., 'discord', 'telegram')."""
        ...

    async def connect(self) -> None:
        """Authenticate and connect to the platform."""
        ...

    async def disconnect(self) -> None:
        """Clean shutdown."""
        ...

    def messages(self) -> AsyncIterator[InboundMessage]:
        """Async iterator of incoming messages."""
        ...

    async def send(self, channel_id: str, message: OutboundMessage) -> SendResult:
        """Send a message to a channel/chat."""
        ...

    # Optional capabilities — adapters declare what they support

    async def send_typing(self, channel_id: str) -> None:
        """Show typing indicator. No-op if unsupported."""
        ...

    async def edit_message(
        self, channel_id: str, message_id: str, new_text: str
    ) -> SendResult:
        """Edit a previously sent message. Used for streaming responses."""
        ...

    async def send_media(
        self, channel_id: str, path: str, *, caption: str | None = None
    ) -> SendResult:
        """Send a media file (image, audio, document)."""
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether edit_message can be used for token-by-token streaming."""
        ...

    @property
    def max_message_length(self) -> int:
        """Maximum characters per message (for splitting)."""
        ...
```

### Gateway Runner

The gateway orchestrates multiple platform adapters and routes messages to agent sessions:

```python
class GatewayRunner:
    """Multi-platform gateway that routes messages to agent sessions.

    Responsibilities:
        - Manage multiple PlatformAdapter connections concurrently
        - Route incoming messages to the correct agent session
        - Handle session creation/lookup via SessionResolver
        - Manage interrupt handling (new message while agent is responding)
        - Stream agent responses back via edit_message or chunked sends
        - Platform-specific formatting (Markdown → platform markup)
    """

    def __init__(
        self,
        adapters: list[PlatformAdapter],
        *,
        agent_factory: Callable[..., Agent],    # Creates agent per session
        session_store: SessionStore | None = None,
        session_resolver: SessionResolver | None = None,
    ): ...

    async def run(self) -> None:
        """Start all adapters and begin processing messages."""
        ...
```

### Reference Implementation: Telegram Adapter

```python
class TelegramAdapter:
    """Telegram adapter using python-telegram-bot.

    Features:
        - Long-polling connection
        - MarkdownV2 formatting
        - Photo/voice/document handling with local caching
        - Streaming via edit_message (batched, rate-limited)
        - /command handling
        - Reply context extraction
        - Typing indicators
        - Message splitting for >4096 char responses
    """
```

### Extension Points
- Community-built adapters: Discord, Slack, WhatsApp, Signal, Matrix, IRC
- Custom `SessionResolver` for per-platform session strategies
- Custom response formatting (Markdown → platform-specific markup)
- Webhook-based adapters vs. long-polling/WebSocket
- Rate limiting strategies per platform
- Media transcoding (voice → text via STT)

### Design Note: Why Not pydantic-ai's UIAdapter?

pydantic-ai's `UIAdapter` is HTTP/SSE-based (AG-UI protocol) — it's built for web UIs that make requests and receive streaming responses. Messaging platforms are fundamentally different:

| UIAdapter (HTTP/SSE) | PlatformAdapter (Event-Driven) |
|---|---|
| Client initiates request | Platform pushes events |
| Single request/response | Long-lived connection |
| Stateless (per-request) | Stateful (connection lifecycle) |
| Server sends SSE stream | Bot edits its own messages |
| No media handling | Image/audio/doc caching |
| No typing indicators | Typing indicators required |

These are different primitives. `UIAdapter` is for web; `PlatformAdapter` is for messaging.

---

## 10. Primitive 7: Sub-Agent Orchestration

### Problem
Complex tasks benefit from spawning specialized sub-agents — a research agent, a coding agent, a review agent — that operate in isolated contexts with their own tools, system prompts, and capabilities.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass


@dataclass
class SubAgentSpec:
    """Specification for spawning a sub-agent."""
    name: str                              # Identifier
    system_prompt: str | None = None       # Override system prompt
    model: str | Model | None = None       # Override model
    capabilities: list[AbstractCapability] | None = None
    toolsets: list[AbstractToolset] | None = None
    blocked_tools: list[str] | None = None # Tools to exclude from child
    output_type: type | None = None        # Expected output schema
    max_turns: int = 50
    context_sharing: ContextSharing = ContextSharing.SUMMARY
    # What context to share from parent:
    #   NONE — clean slate
    #   SUMMARY — compressed summary of parent context
    #   FULL — full message history (careful with token budgets)
    #   SELECTIVE — specific messages selected by parent


class ContextSharing(Enum):
    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"
    SELECTIVE = "selective"


@dataclass
class SubAgentResult:
    """Result from a sub-agent run."""
    output: Any                            # Structured output from sub-agent
    messages: list[ModelMessage]           # Full message history
    usage: RunUsage                        # Token usage
    summary: str | None = None             # Auto-generated summary of work done


@dataclass
class IterationBudget:
    """Shared iteration budget between parent and child agents.

    Prevents runaway delegation chains. When a parent spawns a child,
    the child draws from the same budget pool. Inspired by Hermes's
    IterationBudget with pressure warnings at 70% and 90% thresholds.
    """
    max_iterations: int
    consumed: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_iterations - self.consumed)

    @property
    def pressure(self) -> float:
        """0.0 to 1.0 — used for nudging the LLM about budget limits."""
        return self.consumed / self.max_iterations if self.max_iterations > 0 else 1.0

    def is_exhausted(self) -> bool:
        return self.consumed >= self.max_iterations


class SubAgentSpawner(Protocol):
    """Spawns and manages sub-agent runs."""

    async def spawn(
        self,
        spec: SubAgentSpec,
        *,
        task: str,                         # The task to give the sub-agent
        parent_context: list[ModelMessage] | None = None,
        iteration_budget: IterationBudget | None = None,
    ) -> SubAgentResult:
        """Spawn a sub-agent and wait for completion."""
        ...

    async def spawn_parallel(
        self,
        specs: list[tuple[SubAgentSpec, str]],  # (spec, task) pairs
        *,
        parent_context: list[ModelMessage] | None = None,
        iteration_budget: IterationBudget | None = None,
    ) -> list[SubAgentResult]:
        """Spawn multiple sub-agents concurrently."""
        ...
```

### Reference Implementation

```python
class InProcessSubAgentSpawner:
    """Spawns sub-agents as in-process pydantic-ai Agent runs.

    Each sub-agent gets:
    - Its own Agent instance with specified capabilities/tools
    - Toolset filtering via blocked_tools (e.g., block delegation to
      prevent recursive spawning, block memory to avoid conflicts)
    - Isolated message history (with optional context sharing)
    - Independent usage tracking
    - Shared IterationBudget with parent (prevents runaway delegation)
    - Automatic result summarization

    Modeled on Hermes's delegate_tool.py, which spawns child AIAgent
    instances in-process with isolated conversations and restricted
    toolsets (blocks delegate_task, clarify, memory, send_message,
    execute_code by default).
    """

    def __init__(
        self,
        *,
        default_model: str | Model = "anthropic:claude-sonnet-4-20250514",
        summarizer_model: str | Model = "gemini-2.5-flash",
        compaction: CompactionCapability | None = None,  # Sub-agents get compaction too
        default_blocked_tools: list[str] | None = None,
    ): ...
```

### Capability: SubAgentCapability

```python
class SubAgentCapability(AbstractCapability[DepsT]):
    """Provides sub-agent spawning as a tool.

    Gives the agent a `spawn_agent` tool that lets it delegate
    tasks to specialized sub-agents. Results are injected back
    into the parent conversation.
    """

    def __init__(
        self,
        spawner: SubAgentSpawner | None = None,
        *,
        agent_specs: dict[str, SubAgentSpec] | None = None,  # Pre-defined agent types
        allow_dynamic_specs: bool = False,  # Can the agent define its own sub-agents?
        iteration_budget: IterationBudget | None = None,  # Shared budget
    ): ...
```

### Extension Points
- **Subprocess spawner** — sub-agents as separate processes (isolation, different Python envs)
- **Remote spawner** — sub-agents on different machines (distributed workloads)
- **Pre-defined agent library** — named agents with fixed capabilities (researcher, coder, reviewer)
- Custom context sharing strategies
- Result aggregation patterns (voting, consensus, best-of-N)
- Custom toolset filtering per sub-agent type

### Design Note: How Hermes Actually Does Delegation

Hermes's `delegate_tool.py` spawns child `AIAgent` instances **in-process** — each child gets a fresh conversation, a restricted toolset (no delegation, no memory, no external messaging), and a focused system prompt built from the delegated goal. The parent waits for completion via `ThreadPoolExecutor`. This is *not* HTTP-based — the `send_message_tool.py` is for cross-platform messaging (Telegram, Discord, etc.), not sub-agent coordination.

---

## 11. Primitive 8: Skill Discovery & Registration

### Problem
Skills are declarative instruction bundles (Markdown with frontmatter) that teach the agent specialized behaviors — how to use a specific API, follow a coding convention, or execute a complex workflow. They're the knowledge counterpart to tools (which provide actions).

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillDefinition:
    """A loaded skill."""
    name: str                              # Unique identifier (e.g., "github-pr-workflow")
    description: str                       # One-line summary
    content: str                           # Full Markdown instructions
    category: str | None = None            # Grouping (e.g., "github", "devops")
    source_path: Path | None = None        # Where it was loaded from
    platforms: list[str] | None = None     # Platform constraints (e.g., ["cli", "discord"])
    conditions: dict[str, Any] = field(default_factory=dict)  # Activation conditions
    references: dict[str, str] = field(default_factory=dict)  # Supporting files


class SkillRegistry(Protocol):
    """Discovers, loads, and manages skills."""

    async def discover(self) -> list[SkillDefinition]:
        """Discover all available skills from configured sources."""
        ...

    async def get(self, name: str) -> SkillDefinition | None:
        """Get a specific skill by name."""
        ...

    async def activate(self, name: str) -> None:
        """Mark a skill as active for the current session."""
        ...

    async def deactivate(self, name: str) -> None:
        """Remove a skill from the active set."""
        ...

    def format_active_for_prompt(self) -> str:
        """Format all active skills for system prompt injection."""
        ...

    def format_index_for_prompt(self) -> str:
        """Format skill index (names + descriptions) for discovery."""
        ...
```

### Reference Implementation

```python
class FileSkillRegistry:
    """File-system based skill discovery.

    Discovery hierarchy (like pi-mono):
        1. Project-local: .harness/skills/
        2. User-global: ~/.pydantic-harness/skills/
        3. Explicit paths from configuration

    Skill format (SKILL.md):
        ---
        name: github-pr-workflow
        description: "Create and manage GitHub pull requests"
        category: github
        platforms: [cli]
        ---

        # Instructions
        When creating a PR, follow these steps...

    Supports:
        - Category grouping via DESCRIPTION.md in parent dirs
        - Reference files in references/ subdirectory
        - Platform filtering
        - Conditional activation (e.g., only when certain tools are available)
    """
```

### Capability: SkillCapability

```python
class SkillCapability(AbstractCapability[DepsT]):
    """Skills integrated into agent lifecycle.

    - Injects active skill instructions via get_instructions()
    - Provides skill management tools (list, activate, deactivate, search)
    - Auto-activates skills based on conditions (tool presence, platform, directory)
    - Follows frozen snapshot pattern for system prompt injection
    """
```

### Extension Points
- **Remote skill registries** — download skills from a marketplace (like Hermes's index-cache)
- **MCP-based skill sharing** — skills as MCP resources
- **Conditional auto-activation** — skills activate when relevant tools/context detected
- **Skill composition** — skills that reference other skills

---

## 12. Primitive 9: Run Control & Limits

### Problem
Production agents need guardrails beyond pydantic-ai's `UsageLimits`: max turns, stuck detection, cost limits, time limits, and graceful degradation when limits are hit.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass


@dataclass
class RunLimits:
    """Comprehensive run limits."""
    max_turns: int = 100                   # Maximum agent turns (not just LLM requests)
    max_cost_usd: float | None = None      # Dollar cost ceiling
    max_wall_time_seconds: float | None = None  # Wall-clock timeout
    max_consecutive_errors: int = 3        # Stuck detection
    max_consecutive_same_tool: int = 5     # Loop detection (same tool called N times)
    context_utilization_ceiling: float = 0.95  # Hard stop if context nearly full


@dataclass
class RunControlState:
    """Tracks run progress against limits."""
    turns: int = 0
    errors: int = 0
    consecutive_same_tool: int = 0
    last_tool_name: str | None = None
    estimated_cost_usd: float = 0.0
    start_time: float = 0.0


class RunControlPolicy(Protocol):
    """Evaluates whether the run should continue."""

    def check(self, state: RunControlState, limits: RunLimits) -> RunDecision:
        """Evaluate whether to continue, warn, or stop."""
        ...


class RunDecision(Enum):
    CONTINUE = "continue"                  # All clear
    WARN = "warn"                          # Approaching a limit
    STOP = "stop"                          # Hard limit reached
    COMPACT_AND_CONTINUE = "compact"       # Compact then continue
```

### Capability: RunControlCapability

```python
class RunControlCapability(AbstractCapability[DepsT]):
    """Enforces run limits via agent lifecycle hooks.

    - Tracks turns via wrap_node_run
    - Detects stuck loops via after_tool_execute
    - Enforces cost/time limits via before_model_request
    - Triggers compaction when context ceiling approached
    - Injects warning messages when approaching limits
    - Shares IterationBudget with SubAgentCapability when both are present
    """
```

### Integration with Sub-Agent Budgets

When `RunControlCapability` and `SubAgentCapability` are both present, the run control state should feed into the shared `IterationBudget`. Hermes implements this with pressure warnings at 70% and 90% thresholds — the LLM is told how much budget remains so it can prioritize.

---

## 13. Primitive 10: Configuration Resolution

### Problem
Production agents need layered configuration that merges settings from multiple sources: global defaults, user preferences, project-specific overrides, and environment variables. Both Hermes and Claude Code have sophisticated config systems with file-based discovery. Without this primitive, every application built on pydantic-harness must reinvent config loading.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigLayer:
    """A single configuration layer with its source."""
    source: str              # "default", "global", "project", "env", "cli"
    path: Path | None        # File path (None for env/cli layers)
    values: dict[str, Any]   # Configuration values
    priority: int            # Higher priority overrides lower


class ConfigResolver(Protocol):
    """Discovers and merges configuration from multiple sources."""

    def resolve(self, *, project_dir: Path | None = None) -> dict[str, Any]:
        """Resolve merged configuration from all layers.

        Layer priority (highest wins):
            1. CLI arguments / explicit overrides
            2. Environment variables (HARNESS_*)
            3. Project-local: .harness/config.yaml
            4. User-global: ~/.pydantic-harness/config.yaml
            5. Built-in defaults

        Returns the merged configuration dict.
        """
        ...

    def get_layers(self) -> list[ConfigLayer]:
        """Return all discovered layers for debugging/inspection."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get a single resolved configuration value."""
        ...
```

### Reference Implementation

```python
class FileConfigResolver:
    """YAML/JSON config resolution with environment variable overrides.

    Discovery:
        - Global: ~/.pydantic-harness/config.yaml
        - Project: .harness/config.yaml (walks up to git root)
        - Env vars: HARNESS_* prefix, dot-path mapped (HARNESS_MODEL_NAME → model.name)

    Config schema includes:
        - model: default model name
        - memory: base_dir, max_chars_per_file
        - session: db_path, resolver_strategy
        - compaction: threshold, summarizer_model
        - approval: default_mode, dangerous_patterns
        - gateway: platform configs
        - hooks: event → shell command mappings
        - skills: discovery paths
    """
```

### Capability: ConfigCapability

```python
class ConfigCapability(AbstractCapability[DepsT]):
    """Makes resolved configuration available to other capabilities.

    - Resolves config once at agent creation
    - Provides config as a dependency via deps
    - Other capabilities can read config values from deps
    """
```

### Extension Points
- TOML support for Rust/Python ecosystem alignment
- Remote config (fetch from API, feature flags)
- Config validation via Pydantic models
- Watch mode (reload on file change for long-running gateway processes)

---

## 14. Primitive 11: User Hooks

### Problem
Users need to define automated behaviors triggered by agent events — run a linter before every code write, send a notification after every completed task, validate outputs against a schema. This is different from developer-built capabilities (which are Python code in the capability system). User hooks are shell commands configured via config files, not Python.

Claude Code implements this as "hooks" — shell commands that execute in response to tool calls. pi-mono achieves it through its extension event system. Without a hooks primitive, users cannot customize agent behavior without writing Python capabilities.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass, field


class HookEvent(Enum):
    """Events that can trigger user hooks."""
    BEFORE_TOOL_EXECUTE = "before_tool_execute"
    AFTER_TOOL_EXECUTE = "after_tool_execute"
    BEFORE_MODEL_REQUEST = "before_model_request"
    AFTER_MODEL_REQUEST = "after_model_request"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    BEFORE_COMPACT = "before_compact"
    PROMPT_SUBMIT = "prompt_submit"


@dataclass
class HookDefinition:
    """A user-defined hook."""
    event: HookEvent
    command: str                           # Shell command to execute
    tool_filter: str | None = None         # Only trigger for specific tools (glob pattern)
    timeout_seconds: float = 30.0
    on_failure: str = "warn"               # "warn", "block", "ignore"


@dataclass
class HookResult:
    """Result of running a user hook."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    blocked: bool = False                  # True if on_failure="block" and hook failed


class UserHookRunner(Protocol):
    """Loads and executes user-defined hooks."""

    def load(self, config: dict[str, Any]) -> list[HookDefinition]:
        """Load hook definitions from configuration."""
        ...

    async def run(
        self,
        event: HookEvent,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[HookResult]:
        """Execute all hooks registered for the given event.

        Context dict includes event-specific data:
        - tool_name, tool_args (for tool events)
        - model_name, message_count (for model events)
        - session_id (for session events)

        Returns results for all executed hooks.
        """
        ...
```

### Reference Implementation

```python
class ShellHookRunner:
    """Executes user hooks as shell subprocesses.

    Hook configuration in .harness/config.yaml:
        hooks:
          before_tool_execute:
            - command: "eslint --fix {file_path}"
              tool_filter: "write_file"
              on_failure: block
            - command: "echo 'Tool {tool_name} called'"
              on_failure: ignore
          after_model_request:
            - command: "./scripts/log-usage.sh {tokens_used}"

    Features:
        - Template variable substitution in commands
        - Timeout enforcement
        - Stdout/stderr capture for agent context
        - Block/warn/ignore failure modes
    """
```

### Capability: UserHookCapability

```python
class UserHookCapability(AbstractCapability[DepsT]):
    """Integrates user hooks into the agent lifecycle.

    Maps HookEvents to AbstractCapability lifecycle hooks:
    - before_tool_execute → BEFORE_TOOL_EXECUTE hooks
    - after_tool_execute → AFTER_TOOL_EXECUTE hooks
    - before_model_request → BEFORE_MODEL_REQUEST hooks
    - etc.

    Hook results with on_failure="block" can prevent tool execution.
    Hook stdout is optionally injected as context for the agent.
    """
```

### Extension Points
- HTTP webhook hooks (POST to URL instead of shell command)
- Python callable hooks (for users who want to avoid shell)
- Hook chaining (pipe output of one hook into another)
- Hook marketplace (share useful hooks like "auto-format on write")

---

## 15. Primitive 12: Display Adapter

### Problem
Agent-facing applications need to render output to diverse surfaces — CLI terminals, web UIs, messaging platforms, log files. Display concerns (spinners, progress bars, tool previews, streaming output formatting, markdown rendering) are separate from platform concerns (how to send a message to Telegram). Both Hermes and Claude Code have rich display layers. Without this primitive, every frontend built on pydantic-harness must build its own rendering.

### Protocol

```python
from typing import Protocol
from dataclasses import dataclass
from enum import Enum


class DisplayEvent(Enum):
    """Types of display events."""
    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_END = "tool_end"
    STREAM_DELTA = "stream_delta"
    THINKING = "thinking"
    STATUS = "status"
    ERROR = "error"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"


@dataclass
class DisplayUpdate:
    """A display event to render."""
    event: DisplayEvent
    content: str | None = None             # Text content
    tool_name: str | None = None           # For tool events
    tool_preview: str | None = None        # One-line summary of tool call
    progress: float | None = None          # 0.0 to 1.0 for progress events
    metadata: dict[str, Any] | None = None


class DisplayAdapter(Protocol):
    """Renders agent activity to a display surface."""

    async def update(self, event: DisplayUpdate) -> None:
        """Render a display event."""
        ...

    async def stream_text(self, delta: str) -> None:
        """Render a streaming text delta."""
        ...

    async def show_spinner(self, message: str) -> None:
        """Show a spinner/loading indicator."""
        ...

    async def hide_spinner(self) -> None:
        """Hide the spinner."""
        ...

    def format_markdown(self, text: str) -> str:
        """Format markdown for this display surface.

        CLI: ANSI codes, web: HTML, messaging: platform-specific markup.
        """
        ...
```

### Reference Implementation

```python
class CLIDisplayAdapter:
    """Rich terminal display with spinners, progress, and tool previews.

    Features:
        - Animated spinners during LLM calls
        - One-line tool call previews (e.g., "Reading config.yaml")
        - Streaming text output with cursor management
        - Markdown rendering via rich/mdformat
        - Color themes / skins

    Inspired by Hermes's agent/display.py (KawaiiSpinner, tool
    preview summaries, skin-aware emoji) and Claude Code's
    tool progress display.
    """
```

### Capability: DisplayCapability

```python
class DisplayCapability(AbstractCapability[DepsT]):
    """Routes agent lifecycle events to a DisplayAdapter.

    Maps capability hooks to display events:
    - before_model_request → show spinner
    - after_model_request → hide spinner
    - before_tool_execute → show tool preview
    - after_tool_execute → show tool result summary
    - wrap_run_event_stream → stream text deltas
    """
```

### Extension Points
- Web display (HTML/React components)
- Structured logging display (JSON lines for observability)
- Silent/headless mode (suppress all display)
- Custom tool preview formatters
- Theme/skin system

---

## 16. How Primitives Compose

The power of this architecture is that primitives compose naturally through the capability system. Here's a production-ready agent configuration:

```python
from pydantic_ai import Agent
from pydantic_harness import (
    # Primitives
    TokenCountingTracker,
    FourPhaseCompactionEngine,
    FileMemoryStore,
    SQLiteSessionStore,
    FileSkillRegistry,
    PatternBasedApprovalPolicy,
    FileConfigResolver,
    ShellHookRunner,
    CLIDisplayAdapter,
    # Capabilities
    CompactionCapability,
    MemoryCapability,
    SessionCapability,
    ApprovalCapability,
    SkillCapability,
    RunControlCapability,
    SubAgentCapability,
    UserHookCapability,
    DisplayCapability,
    # Execution (exists in pydantic-harness)
    ExecutionEnv,
    CodeMode,
    LocalEnvironment,
)

# --- Configuration ---
config = FileConfigResolver().resolve(project_dir=Path("."))

# --- Primitives (state, persistence) ---
tracker = TokenCountingTracker(context_window=200_000)
compaction = FourPhaseCompactionEngine(summarizer_model="gemini-2.5-flash")
memory = FileMemoryStore(base_dir=Path("~/.pydantic-harness/memory"))
sessions = SQLiteSessionStore(db_path=Path("~/.pydantic-harness/sessions.db"))
skills = FileSkillRegistry(paths=["./skills", "~/.pydantic-harness/skills"])
approval = PatternBasedApprovalPolicy(default_mode="ask_first")
hooks = ShellHookRunner(config.get("hooks", {}))
display = CLIDisplayAdapter()

# --- Capabilities (lifecycle integration) ---
agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    capabilities=[
        # P0: Context management
        CompactionCapability(engine=compaction, tracker=tracker, threshold=0.5),
        MemoryCapability(store=memory),
        ApprovalCapability(policy=approval),

        # P1: Run control
        RunControlCapability(limits=RunLimits(max_turns=100, max_cost_usd=5.0)),

        # P1: Sub-agents
        SubAgentCapability(agent_specs={
            "researcher": SubAgentSpec(model="gemini-2.5-pro", max_turns=20),
            "coder": SubAgentSpec(capabilities=[CodeMode()], max_turns=50),
        }),

        # P2: Session persistence
        SessionCapability(store=sessions),

        # P2: Skills
        SkillCapability(registry=skills),

        # P2: User hooks & display
        UserHookCapability(runner=hooks),
        DisplayCapability(adapter=display),

        # Execution environments
        ExecutionEnv(environment=LocalEnvironment(cwd="/workspace")),
    ],
)
```

### Composition Order Matters

`CombinedCapability` processes hooks in registration order for `before_*` and reverse for `wrap_*`/`after_*`. The recommended ordering:

```
1. CompactionCapability     ← must run before_model_request FIRST to compress before others see messages
2. MemoryCapability         ← injects memory into prompt, provides tools
3. ApprovalCapability       ← gates tool execution before other before_tool_execute hooks
4. RunControlCapability     ← checks limits before each model request
5. SubAgentCapability       ← provides spawn tool
6. SessionCapability        ← persists messages after each model request (LIFO = runs last in after_*)
7. SkillCapability          ← injects instructions
8. UserHookCapability       ← runs user hooks around tool execution
9. DisplayCapability        ← renders events (outermost wrapper for wrap_* hooks)
10. ExecutionEnv/CodeMode   ← provides tools
```

### Gateway Composition

For messaging platform deployment:

```python
from pydantic_harness.gateway import GatewayRunner, TelegramAdapter

gateway = GatewayRunner(
    adapters=[
        TelegramAdapter(token="BOT_TOKEN"),
        # DiscordAdapter(token="..."),  # Community-built
    ],
    agent_factory=lambda session_ctx: Agent(
        "anthropic:claude-sonnet-4-20250514",
        capabilities=[
            CompactionCapability(...),
            MemoryCapability(...),
            ApprovalCapability(...),
            SessionCapability(...),
            # No ExecutionEnv — messaging agents typically don't run code
            # No DisplayCapability — the gateway handles rendering via PlatformAdapter
        ],
    ),
    session_store=sessions,
    session_resolver=PlatformSessionResolver(strategy="per-user"),
)

await gateway.run()
```

---

## 17. Implementation Priority

Based on dependency chain and competitive impact:

### P0 — Must Have (blocks everything else)

| Primitive | Est. LOC | Depends On | Rationale |
|---|---|---|---|
| **ContextWindowTracker** | ~200 | `context-window-models` branch | Foundation for compaction, run control, sub-agents |
| **CompactionEngine + Capability** | ~800 | ContextWindowTracker | Without compaction, agents die at context limit. Every competitor has this. |
| **MemoryStore + Capability** | ~800 | CompactionCapability (flush) | Cross-session intelligence. This is the main differentiator. |
| **ApprovalPolicy + Capability** | ~500 | — | Every production agent needs permission gating. Without it, agents are unsafe to deploy. |

### P1 — High Impact

| Primitive | Est. LOC | Depends On | Rationale |
|---|---|---|---|
| **RunControlPolicy + Capability** | ~300 | ContextWindowTracker | Production guardrails. Prevents runaway costs. |
| **SubAgentSpawner + Capability** | ~500 | CompactionCapability | Complex task decomposition. Major capability gap vs competitors. |
| **ConfigResolver** | ~400 | — | Every other primitive needs configuration. Unblocks user customization. |
| Land `context-window-models` branch | upstream | — | Prerequisite for ContextWindowTracker |

### P2 — Platform Reach

| Primitive | Est. LOC | Depends On | Rationale |
|---|---|---|---|
| **SessionStore + Capability** | ~600 | — | Session persistence across restarts |
| **PlatformAdapter + GatewayRunner** | ~1,200 | SessionStore | Multi-platform deployment |
| **Telegram reference adapter** | ~600 | PlatformAdapter | Prove the abstraction works |
| **DisplayAdapter + Capability** | ~500 | — | CLI rendering, unblocks polished UX |

### P3 — Ecosystem

| Primitive | Est. LOC | Depends On | Rationale |
|---|---|---|---|
| **SkillRegistry + Capability** | ~500 | — | Declarative agent knowledge |
| **UserHookRunner + Capability** | ~400 | ConfigResolver | User-defined automation |
| Docker execution environment | ~400 | ExecutionEnv | Sandboxed code execution |
| Additional platform adapters | ~600 each | PlatformAdapter | Community-driven |

---

## Appendix: Comparison Matrix

How each system handles each primitive:

| Primitive | pydantic-ai (today) | pi-mono | Hermes Agent | pydantic-harness (target) |
|---|---|---|---|---|
| **Agent Loop** | Graph DAG (nodes) | Event-driven loop (31 events) | While loop + tool dispatch | Use pydantic-ai as-is |
| **Extension Mechanism** | AbstractCapability (25 hooks, 4 phases) | ExtensionAPI (31 events, registration) | Distributed registry pattern (`tools/registry.py`, self-registering tool modules) | AbstractCapability + primitives |
| **Context Tracking** | UsageLimits (token ceilings) | ContextUsage (live tracking) | chars/4 estimate (`_CHARS_PER_TOKEN=4`) + threshold% (default 50%) | ContextWindowTracker protocol |
| **Compaction** | HistoryProcessor hook exists, no impl | `session_before_compact` event | Four-phase (prune→align→summarize→repair orphans) | FourPhaseCompactionEngine |
| **Memory** | None | Custom entries in session storage | MEMORY.md/USER.md (file-backed) + Honcho (cross-session) + SQLite FTS5 (search) | MemoryStore protocol + FileMemoryStore |
| **Session Persistence** | None | SessionManager (JSONL with id/parentId tree, branching, forking) | SQLite with WAL + parent lineage + FTS5 | SessionStore protocol + SQLiteSessionStore |
| **Approval/Permissions** | `Tool.requires_approval` flag only | Extension-based (`tool_call` event can block) | Pattern-based regex detection + session/permanent approval caching | ApprovalPolicy protocol + PatternBasedApprovalPolicy |
| **Platform Gateway** | UIAdapter (HTTP/SSE, AG-UI protocol) | Interactive/RPC/Print modes | BasePlatformAdapter (13 concrete adapters: Telegram, Discord, Slack, WhatsApp, Signal, Matrix, etc.) | PlatformAdapter protocol + GatewayRunner |
| **Sub-Agents** | Can manually create nested Agent() | Extension-based subprocess spawning | In-process AIAgent spawning via `delegate_tool.py` (isolated conversation, restricted toolset, shared iteration budget) | SubAgentSpawner protocol |
| **Skills** | None | SKILL.md with frontmatter | SKILL.md with categories, refs, and multi-source registries (official, community, trusted) | SkillRegistry protocol |
| **Run Control** | UsageLimits (request/token limits) | None (manual) | IterationBudget with pressure warnings at 70%/90% | RunControlPolicy (turns, cost, time, stuck) |
| **Configuration** | None (programmatic only) | Config files + CLI flags | Layered config (`~/.hermes/`, project-level, env vars) | ConfigResolver protocol |
| **User Hooks** | None | Extension event system (user-authored extensions) | None (behavior configured via system prompt) | UserHookRunner protocol |
| **Display** | None | TUI library (`pi-tui` package) | KawaiiSpinner + tool previews + skin system (`agent/display.py`) | DisplayAdapter protocol |
| **Tools** | AbstractToolset (fluent composition) | AgentTool + Operations interfaces | Distributed registry (`tools/registry.py`) with self-registering modules | Use pydantic-ai toolsets |
| **Model Abstraction** | Model ABC, 20+ providers, FallbackModel | ApiProvider registry, StreamFunction | OpenAI-compatible + native Anthropic adapter | Use pydantic-ai as-is |
| **Prompt Caching** | Anthropic cache_* settings | Provider-specific | Frozen snapshot pattern (memory/skills frozen at session start) | Frozen snapshot via MemoryCapability + general FrozenPromptContent pattern |

### Key Takeaway

pydantic-ai provides the **engine** (agent loop, capabilities, toolsets, models). pydantic-harness provides the **fuel** (context awareness, compaction, memory, permissions, sessions, platforms, skills, configuration, hooks, display). The primitives layer is what makes the engine capable of running production-grade, long-running agents across diverse deployment targets.

The 12 primitives fall into four natural groups:
1. **Context management** (ContextWindowTracker, CompactionEngine, MemoryStore) — keep the agent running
2. **Safety & control** (ApprovalPolicy, RunControlPolicy, UserHookRunner) — keep the agent safe
3. **Persistence & deployment** (SessionStore, PlatformAdapter, ConfigResolver, DisplayAdapter) — put the agent in production
4. **Knowledge & delegation** (SkillRegistry, SubAgentSpawner) — make the agent capable
