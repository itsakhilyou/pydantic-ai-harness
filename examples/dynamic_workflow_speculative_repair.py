"""Flagship `DynamicWorkflow` example: a self-verifying repair tournament.

`DynamicWorkflow` is inspired by Claude Code's dynamic workflows — one orchestration script,
many sub-agents — and does it the Pydantic way: typed end to end, cache-stable, with each
sub-agent a full Pydantic AI `Agent`.

The orchestrator is handed a failing test and, in a *single* tool call, writes a Python
script that:

1. runs a **scout** sub-agent to reproduce the failure and produce a *typed* diagnosis,
2. fans out three **fixer** sub-agents in parallel, each pursuing a different strategy and
   returning a *typed* proposed diff (it proposes, it does not apply — so the fan-out never
   conflicts),
3. has a **referee** sub-agent score each proposal, and
4. picks the smallest passing diff — all in ordinary Python control flow, one model turn.

What that buys, in practice:

- **Each leaf is a confined capability surface.** The scout gets a read-only `FileSystem` + a
  `Shell` allowlisted to `pytest`/`git`; the fixer gets `code_mode` over a write-confined
  `FileSystem`. Every sub-agent is a full Pydantic AI agent with its own guardrails, and they
  can even be on different models/providers — containment lives where the blast radius is.
- **Typed handoffs end-to-end.** The scout returns a `Diagnosis`, each fixer a `FixProposal`,
  the referee a `Score` — the script indexes real fields, not parsed strings. A shared typed
  `deps` threads through the whole tree.
- **Exact, host-enforced budget** across the whole tree (`max_agent_calls`), plus Monty
  resource limits on the orchestration script itself.
- **Loads on demand, stays cache-stable.** `DynamicWorkflow` is `defer_loading=True`: it costs
  ~one line of prompt until the model actually needs to orchestrate. And a specialist
  `security_audit` sub-agent is *revealed at runtime* — appended to the live `agents` list only
  when the diagnosis touches auth code — without ever moving the prompt-cache prefix.

Run it (needs an Anthropic key):

    export ANTHROPIC_API_KEY=sk-...
    uv run --with 'pydantic-ai-harness[code-mode]' --with anthropic --with logfire \
        python examples/dynamic_workflow_speculative_repair.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from pydantic_ai_harness import CodeMode, DynamicWorkflow, FileSystem, Shell, WorkflowAgent

# The full tree shows up in one trace: the orchestrator turn, the `run_workflow` tool call with
# the exact script the model wrote, and every nested sub-agent run with its typed input/output.
logfire.configure(send_to_logfire='if-token-present', service_name='dynamic-workflow-repair')
logfire.instrument_pydantic_ai()

MODEL = 'anthropic:claude-opus-4-8'
REPO = '.'  # point at the repo under repair


# ---- Typed handoffs ---------------------------------------------------------------------------


class Diagnosis(BaseModel):
    """The scout's typed read on the failure."""

    summary: str
    suspect_files: list[str]
    touches_auth: bool  # gate for revealing the security specialist at runtime


class FixProposal(BaseModel):
    """One fixer's proposed (not applied) repair."""

    strategy: str
    rationale: str
    unified_diff: str  # proposed, not applied — parallel fixers never collide
    risk: int  # 0-10


class Score(BaseModel):
    """A referee's 0-10 judgement of a single proposal."""

    value: int  # 0-10
    reason: str


# ---- Shared deps: the host keeps a handle on the live agent catalog ---------------------------


@dataclass
class RepairDeps:
    """Deps shared by the orchestrator and (because deps are forwarded) every sub-agent."""

    # The orchestrator holds the *same list object* passed to DynamicWorkflow(agents=...), so a
    # tool (or the host) can append a sub-agent mid-run and have it become callable next step.
    agents: list[WorkflowAgent[RepairDeps]]


# ---- The leaf sub-agents, each with its own confined capability surface -----------------------
#
# All declare `deps_type=RepairDeps`: the orchestrator forwards its deps into every sub-agent, so
# the deps type must match even though these leaves don't read it.

scout = Agent(
    MODEL,
    name='scout',
    deps_type=RepairDeps,
    output_type=Diagnosis,
    instructions=(
        'Reproduce the failing test, read the relevant source, and return a tight diagnosis. '
        'Set touches_auth=True if the suspect code involves authentication or authorization.'
    ),
    capabilities=[
        FileSystem(root_dir=REPO, protected_patterns=['**']),  # read-only: every path protected
        Shell(
            cwd=REPO, allowed_commands=['pytest', 'python', 'git'], denied_commands=[]
        ),  # allowlist replaces the denylist
    ],
)

fixer = Agent(
    MODEL,
    name='fixer',
    deps_type=RepairDeps,
    output_type=FixProposal,
    instructions=(
        'You are given a diagnosis and a single strategy to try. Produce a minimal unified diff '
        'that implements ONLY that strategy. Do not apply it — return it as `unified_diff`.'
    ),
    capabilities=[
        CodeMode(),  # let the fixer compute its diff with sandboxed Python tool-use
        FileSystem(root_dir=REPO, protected_patterns=['**/tests/**']),  # may read all, not edit tests
    ],
)

referee = Agent(
    MODEL,
    name='referee',
    deps_type=RepairDeps,
    output_type=Score,
    instructions='Score a single fix proposal 0-10 for correctness and minimality. One-line reason.',
)

# A heavyweight specialist that is NOT in the starting catalog. It is appended at runtime only if
# the diagnosis says auth is involved — progressive disclosure without bloating the base prompt.
security_audit = Agent(
    MODEL,
    name='security_audit',
    deps_type=RepairDeps,
    output_type=Score,
    instructions='Audit a proposed auth fix for privilege-escalation / bypass risk. Score 0-10, lower = safer.',
)


def make_orchestrator() -> tuple[Agent[RepairDeps, str], list[WorkflowAgent[RepairDeps]]]:
    """Build the orchestrator and return it alongside the live catalog the host appends to."""
    # Pass a *mutable list* so runtime reveal works; the host keeps the reference in deps.
    catalog: list[WorkflowAgent[RepairDeps]] = [
        WorkflowAgent(
            agent=scout, description='Reproduces the failure; returns {summary, suspect_files, touches_auth}.'
        ),
        WorkflowAgent(
            agent=fixer, description='Given a diagnosis + strategy, returns a proposed {unified_diff, risk}.'
        ),
        WorkflowAgent(agent=referee, description='Scores one fix proposal 0-10; returns {value, reason}.'),
    ]
    orchestrator = Agent(
        MODEL,
        deps_type=RepairDeps,
        instructions=(
            'A test is failing. Use run_workflow to: (1) run scout once for a diagnosis; '
            '(2) fan out fixer in parallel over three distinct strategies; (3) score each proposal '
            'with referee; (4) return the highest-scoring proposal with the smallest diff. '
            'If the diagnosis sets touches_auth, a security_audit sub-agent will appear — also run '
            'it on the chosen proposal and reject anything it scores above 3.'
        ),
        capabilities=[
            DynamicWorkflow(
                agents=catalog,
                id='workflow',
                defer_loading=True,  # ~one line of prompt until orchestration is actually needed
                max_agent_calls=20,  # exact ceiling across the whole fan-out
            )
        ],
    )
    return orchestrator, catalog


# ---- Runtime reveal: a tool the model can call once it knows auth is involved -----------------


async def reveal_security_specialist(ctx: RunContext[RepairDeps]) -> str:
    """Make the security_audit sub-agent callable from the workflow script (idempotent)."""
    if any(a.resolved_name == 'security_audit' for a in ctx.deps.agents):
        return 'security_audit is already available.'
    ctx.deps.agents.append(
        WorkflowAgent(agent=security_audit, description='Audits an auth fix for bypass risk; returns {value, reason}.')
    )
    # The capability announces the newcomer to the model on the next step, and the cached
    # `run_workflow` description stays frozen — so the prompt-cache prefix never changes.
    return 'security_audit revealed; it is callable from the next run_workflow script.'


async def main() -> None:
    """Run the repair tournament against a failing test and print the chosen fix."""
    orchestrator, catalog = make_orchestrator()
    orchestrator.tool(reveal_security_specialist)
    deps = RepairDeps(agents=catalog)

    result = await orchestrator.run(
        'tests/dynamic_workflow/test_dynamic_workflow.py::test_some_failing_case is failing. '
        'Diagnose and propose the best minimal fix.',
        deps=deps,
    )
    logfire.info('done', answer=result.output, requests=result.usage.requests)
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
