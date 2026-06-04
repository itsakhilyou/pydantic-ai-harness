"""Substantial `DynamicWorkflow` example: migrate a small package, at scale, in one tool call.

This is the "port a codebase" shape — many files, coordinated sub-agents — at a size you can run.
The orchestrator is handed a package that still uses `os.path` and, in a *single* `run_workflow`
call, writes a script that:

1. fans out a **migrator** sub-agent per file, in parallel — each one reads its file, rewrites it
   to `pathlib`, and writes it back (real edits, but only inside a throwaway temp dir);
2. fans out a **reviewer** sub-agent per file that re-reads the result and *adversarially* checks it
   — approve only if no `os.path` remains and behaviour is preserved; and
3. loops: any file the reviewer rejects goes back to the migrator with the reviewer's issues, up to
   a couple of rounds, until the whole package converges.

All of that is ordinary Python control flow (`asyncio.gather`, a `while` loop, a list of pending
files) inside one model turn. The orchestrator never sees the file contents or the intermediate
drafts — only the final summary. Each sub-agent is a full Pydantic AI `Agent` with a typed
`output_type` and a *confined* `FileSystem` (the migrator may write; the reviewer is read-only).

The example plants its own source files in a fresh temp dir, so running it never touches your repo.
It prints the exact script the model wrote and every migrated file, so you can see the orchestration
that happened. Set an Anthropic key and run it:

    export ANTHROPIC_API_KEY=sk-...
    uv run --extra code-mode --with anthropic python examples/dynamic_workflow_migrate.py

With no key it still plants the sources and prints the task, so you can see the setup offline.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import textwrap
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness import DynamicWorkflow, FileSystem, WorkflowAgent

MODEL = 'anthropic:claude-sonnet-4-6'  # or 'anthropic:claude-opus-4-8'

# A tiny package that still uses `os.path`. The migrator agents rewrite each file to `pathlib`; the
# files differ, so the parallel fan-out never collides. Kept small so the whole run is cheap.
SOURCES: dict[str, str] = {
    'area.py': textwrap.dedent("""
        import os.path


        def area_table_path(name):
            base = os.path.dirname(__file__)
            return os.path.join(base, 'tables', name + '.csv')


        def has_table(name):
            return os.path.exists(area_table_path(name))
        """).lstrip(),
    'perimeter.py': textwrap.dedent("""
        import os.path


        def config_file():
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(here, 'perimeter.cfg')


        def config_name():
            return os.path.basename(config_file())
        """).lstrip(),
    'io_utils.py': textwrap.dedent("""
        import os.path


        def stem(path):
            return os.path.splitext(os.path.basename(path))[0]


        def ensure_suffix(path, suffix):
            root, ext = os.path.splitext(path)
            return path if ext == suffix else root + suffix
        """).lstrip(),
}


class MigrationReport(BaseModel):
    """One migrator's typed report on the file it rewrote."""

    path: str
    summary: str


class Review(BaseModel):
    """One reviewer's adversarial verdict on a migrated file."""

    approved: bool
    issues: list[str]


def plant_sources() -> Path:
    """Write the `os.path`-using package into a fresh temp dir and return its path."""
    pkg_dir = Path(tempfile.mkdtemp(prefix='dynworkflow-migrate-'))
    for name, content in SOURCES.items():
        (pkg_dir / name).write_text(content)
    return pkg_dir


def build_orchestrator(pkg_dir: Path) -> Agent[None, str]:
    """Build the orchestrator plus its two confined sub-agents over the planted package."""
    migrator = Agent(
        MODEL,
        name='migrator',
        output_type=MigrationReport,
        instructions=(
            'You migrate ONE Python file from os.path to pathlib. The task gives a file path '
            'relative to the package root. Read that file, rewrite it to use pathlib.Path while '
            'preserving behaviour exactly, and write the result back to the same path. If the task '
            'also lists reviewer issues, fix those too. Return the path and a one-line summary.'
        ),
        capabilities=[FileSystem(root_dir=pkg_dir)],  # may read and write inside the temp package
    )
    reviewer = Agent(
        MODEL,
        name='reviewer',
        output_type=Review,
        instructions=(
            'You review ONE migrated file. The task gives its path. Read it and approve only if it '
            'no longer imports `os`/`os.path` or calls any `os.path.*` function, and its behaviour '
            'is preserved. If anything is wrong, set approved=False and list concrete issues.'
        ),
        capabilities=[FileSystem(root_dir=pkg_dir, protected_patterns=['**'])],  # read-only
    )
    return Agent(
        MODEL,
        instructions=(
            'A small package still uses os.path and must move to pathlib. Use run_workflow ONCE to '
            'migrate every file. In the script: for each file, call migrator(task=<path>), then '
            'reviewer(task=<path>); if the review is not approved, call migrator again with the '
            'issues appended to the task, up to 2 extra rounds. Run the files in parallel with '
            'asyncio.gather. Return a short summary listing each file and whether it was approved.'
        ),
        capabilities=[
            DynamicWorkflow(
                agents=[
                    WorkflowAgent(
                        agent=migrator, description='Migrates one file to pathlib in place; returns {path, summary}.'
                    ),
                    WorkflowAgent(agent=reviewer, description='Reviews one migrated file; returns {approved, issues}.'),
                ],
                # Hard cost ceiling: each sub-agent's token limit is exact (its run is sequential),
                # so with max_agent_calls=N the whole tree spends at most N * T tokens. A shared
                # counter (forward_usage=True) would instead cap at the default request_limit=50 —
                # too low for a real fan-out.
                max_agent_calls=24,
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
    """Plant the package, run the migration in one tool call, and show what happened."""
    pkg_dir = plant_sources()
    print(f'Planted a 3-file os.path package in {pkg_dir}\n')
    for name in SOURCES:
        print(f'  {name}:')
        print(textwrap.indent((pkg_dir / name).read_text().rstrip(), '    '))
        print()

    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('Set ANTHROPIC_API_KEY and re-run to watch the orchestrator migrate all three files')
        print('in a single run_workflow call.')
        return

    orchestrator = build_orchestrator(pkg_dir)
    result = await orchestrator.run('Migrate this package from os.path to pathlib.')

    script = extract_workflow_script(result.all_messages())
    if script is not None:
        print('The orchestrator wrote this script and ran it in ONE tool call:\n')
        print(textwrap.indent(script.strip(), '    '))
        print()

    print('Migrated files:\n')
    for name in SOURCES:
        print(f'  {name}:')
        print(textwrap.indent((pkg_dir / name).read_text().rstrip(), '    '))
        print()

    print(f'Summary the orchestrator returned:\n  {result.output}')


if __name__ == '__main__':
    asyncio.run(main())
