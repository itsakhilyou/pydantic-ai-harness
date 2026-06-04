"""Flagship `DynamicWorkflow` showcase: audit a codebase at scale, in one tool call, with a trace.

This is the pattern the capability is built for — **scale plus adversarial convergence**. The
orchestrator is pointed at a Python package and, in a *single* `run_workflow` call, writes a script
that:

1. fans out a **reviewer** sub-agent per file, in parallel — each reads its file and reports
   suspected bugs as typed `Finding`s (so dozens of files are reviewed at once, not one per turn);
2. fans out a **verifier** sub-agent per finding that *tries to refute it* — a finding only survives
   if an independent agent, re-reading the code, agrees it is real (this kills false positives that
   a single pass would wave through); and
3. hands the survivors to a **synthesizer** that dedupes and ranks them into one report.

The fan-out, the filter, and the synthesis are ordinary Python in one model turn. The orchestrator
never sees a single file's contents or the raw findings — only the final report. Every sub-agent is
a full Pydantic AI `Agent` with a typed `output_type` and a **read-only** `FileSystem`, so the audit
cannot modify the code it inspects.

Instrumented with Logfire: one trace shows the orchestrator turn, the `run_workflow` call (its `code`
argument is the exact script the model wrote), and every nested reviewer / verifier / synthesizer
run. Set `LOGFIRE_TOKEN` and the trace is shareable.

Point it at any package (defaults to this harness's `dynamic_workflow` package; try a checkout of
`pydantic_ai` for a bigger run). It only reads:

    export ANTHROPIC_API_KEY=sk-...
    export LOGFIRE_TOKEN=...   # optional, for a shareable public trace
    uv run --extra code-mode --with anthropic --with logfire \
        python examples/dynamic_workflow_audit.py pydantic_ai_harness/dynamic_workflow

With no key it lists the files it would audit and exits, so you can see the plan offline.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness import DynamicWorkflow, FileSystem, WorkflowAgent

logfire.configure(send_to_logfire='if-token-present', service_name='dynamic-workflow-audit')
logfire.instrument_pydantic_ai()

MODEL = 'anthropic:claude-sonnet-4-6'  # or 'anthropic:claude-opus-4-8'
DEFAULT_TARGET = 'pydantic_ai_harness/dynamic_workflow'
MAX_FILES = 12  # bound the fan-out (and the cost) on large packages


class Finding(BaseModel):
    """One suspected bug a reviewer found in one file."""

    file: str
    line: int
    severity: str  # 'low' | 'medium' | 'high'
    issue: str


class Verdict(BaseModel):
    """A verifier's adversarial judgement on a single finding."""

    confirmed: bool
    reason: str


class AuditReport(BaseModel):
    """The synthesized, ranked result of the whole audit."""

    summary: str
    confirmed_findings: list[str]


def python_files(target: Path) -> list[str]:
    """List the package's Python files (relative to `target`), capped at `MAX_FILES`."""
    files = sorted(str(p.relative_to(target)) for p in target.rglob('*.py'))
    return files[:MAX_FILES]


# Given the catalog and instructions below, a Claude model wrote this exact script for the audit
# task (reproduced verbatim, verified to run in the Monty sandbox) — scale (a reviewer per file)
# plus adversarial convergence (a verifier refutes each finding) plus synthesis, in one model turn.
# `extract_workflow_script` prints the script from your own run.
#
#     import asyncio
#     import json
#
#     files = ["__init__.py", "_capability.py", "_toolset.py"]
#
#     # 1. Review every file at once — one reviewer sub-agent per file.
#     reviews = await asyncio.gather(*[reviewer(task=f) for f in files])
#     findings = []
#     for review in reviews:
#         if review:
#             findings.extend(review)
#
#     # 2. Refute every finding at once — it survives only if the verifier confirms it.
#     verdicts = await asyncio.gather(*[verifier(task=json.dumps(f)) for f in findings])
#     confirmed = [f for f, v in zip(findings, verdicts) if v["confirmed"]]
#
#     # 3. Rank the survivors into one report — the only value the orchestrator sees.
#     report = await synthesizer(task=json.dumps(confirmed))
#     report
def build_orchestrator(target: Path) -> Agent[None, str]:
    """Build the orchestrator and its three read-only sub-agents over `target`."""
    read_only = FileSystem(root_dir=target, protected_patterns=['**'])
    reviewer = Agent(
        MODEL,
        name='reviewer',
        output_type=list[Finding],
        instructions=(
            'You audit ONE Python file for real bugs (logic errors, unhandled edge cases, resource '
            'leaks, incorrect types). The task is the file path. Read it and return a list of '
            'concrete Findings with the line number; return an empty list if it is clean.'
        ),
        capabilities=[read_only],
    )
    verifier = Agent(
        MODEL,
        name='verifier',
        output_type=Verdict,
        instructions=(
            'You are given one finding as JSON. Re-read the referenced file and TRY TO REFUTE it. '
            'Set confirmed=True only if, after reading the code, the bug is unmistakably real. When '
            'in doubt, confirmed=False. Give a one-line reason.'
        ),
        capabilities=[read_only],
    )
    synthesizer = Agent(
        MODEL,
        name='synthesizer',
        output_type=AuditReport,
        instructions=(
            'You are given the confirmed findings as JSON. Deduplicate them, rank by severity, and '
            'write a short report: a one-paragraph summary and a ranked list of the real issues.'
        ),
    )
    return Agent(
        MODEL,
        instructions=(
            'Audit the Python files listed below for bugs. Use run_workflow ONCE. In the script: '
            'review every file in parallel with reviewer(task=<path>); collect all findings; verify '
            'each finding in parallel with verifier(task=<json finding>) and keep only the confirmed '
            'ones; then pass the survivors to synthesizer(task=<json findings>) and return its '
            'report. Reviewers and verifiers must run concurrently with asyncio.gather.'
        ),
        capabilities=[
            DynamicWorkflow(
                agents=[
                    WorkflowAgent(
                        agent=reviewer, description='Audits one file; returns a list of {file, line, severity, issue}.'
                    ),
                    WorkflowAgent(
                        agent=verifier,
                        description='Refutes or confirms one finding (given as JSON); returns {confirmed, reason}.',
                    ),
                    WorkflowAgent(
                        agent=synthesizer, description='Ranks confirmed findings (given as JSON) into a report.'
                    ),
                ],
                # Hard cost ceiling on the whole fan-out. Each sub-agent runs sequentially, so its
                # own token limit is enforced exactly; with max_agent_calls=N the tree spends at
                # most N * T tokens — here 60 * 500k = 30M, a provable upper bound. (forward_usage=
                # True would instead share one counter, but its limit is only best-effort under
                # concurrent fan-out, and the default request_limit=50 is far too low for a fan-out
                # this size — each reviewer alone makes several read_file/search_files calls.)
                max_agent_calls=60,
                forward_usage=False,
                sub_agent_usage_limits=UsageLimits(total_tokens_limit=500_000),
            )
        ],
    )


def extract_workflow_script(messages: list[ModelMessage]) -> str | None:
    """Pull the exact Python the model passed to `run_workflow`, for display."""
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == 'run_workflow':
                    args = part.args
                    if isinstance(args, dict):
                        code = args.get('code')
                        if isinstance(code, str):
                            return code
    return None


async def main() -> None:
    """Audit the target package in one tool call and print the script and the report."""
    target = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET).resolve()
    files = python_files(target)
    print(f'Auditing {len(files)} files under {target}:')
    for name in files:
        print(f'  {name}')
    print()

    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('Set ANTHROPIC_API_KEY (and LOGFIRE_TOKEN for a shareable trace) and re-run to audit')
        print('all of these in a single run_workflow call.')
        return

    orchestrator = build_orchestrator(target)
    file_list = '\n'.join(f'- {name}' for name in files)
    result = await orchestrator.run(f'Audit these files for bugs:\n{file_list}')

    script = extract_workflow_script(result.all_messages())
    if script is not None:
        print('The orchestrator wrote this script and ran it in ONE tool call:\n')
        print(script.strip())
        print()
    print(f'Report:\n{result.output}')


if __name__ == '__main__':
    asyncio.run(main())
