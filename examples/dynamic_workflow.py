"""Migrate a package from `os.path` to `pathlib` in a single `run_workflow` call, with a trace.

This is the end-to-end `DynamicWorkflow` example: one orchestrator turn fans work out across many
sub-agents, lets them change code under confinement, retries what fails, and synthesizes a typed
report — all as ordinary Python inside one `run_workflow` call. The orchestrator is handed a package
that still uses `os.path` and writes a script that:

1. migrates every file in parallel — one `migrator` sub-agent per file reads its file, rewrites it
   to `pathlib`, and writes it back (real edits, but only inside a throwaway temp dir);
2. reviews every file in parallel — one read-only `reviewer` sub-agent per file approves the result
   only if no `os.path` remains and behaviour is preserved;
3. loops: any file the reviewer rejects goes back to the migrator with the reviewer's issues, for up
   to two extra rounds, then reports each file's final status; and
4. synthesizes — a `synthesizer` sub-agent turns the per-file outcomes into one typed report.

The retry loop is the part you cannot express as one sub-agent per turn: re-dispatching only the
files that failed review, round after round, is ordinary Python control flow (`asyncio.gather`, a
`while` loop, a list of pending files) in one model turn. The orchestrator's context never gains the
file contents or the intermediate drafts — only the final typed report. Each sub-agent is a Pydantic
AI `Agent` with a typed `output_type` and exactly the filesystem access it needs: the migrator may
write, the reviewer is read-only, and the synthesizer has no filesystem at all.

Logfire instrumentation gives one trace covering the orchestrator turn, the `run_workflow` call
(whose `code` argument is the exact script the model wrote), and every nested migrator, reviewer,
and synthesizer run. Set `LOGFIRE_TOKEN` to make the trace shareable.

The example plants its own source files in a fresh temp dir, so running it never touches your repo.
It prints the exact script the model wrote and every migrated file, so you can see what the
orchestration did:

    export ANTHROPIC_API_KEY=sk-...
    export LOGFIRE_TOKEN=...   # optional, for a shareable public trace
    uv run --extra code-mode --with anthropic --with logfire \
        python examples/dynamic_workflow.py

With no key it still plants the sources and prints the task, so you can see the setup offline.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import textwrap
from pathlib import Path

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness import DynamicWorkflow, FileSystem, WorkflowAgent

logfire.configure(send_to_logfire='if-token-present', service_name='dynamic-workflow')
logfire.instrument_pydantic_ai()

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


class MigrationSummary(BaseModel):
    """The synthesizer's typed report on the whole migration."""

    summary: str
    files: list[str]


def plant_sources() -> Path:
    """Write the `os.path`-using package into a fresh temp dir and return its path."""
    pkg_dir = Path(tempfile.mkdtemp(prefix='dynworkflow-migrate-'))
    for name, content in SOURCES.items():
        (pkg_dir / name).write_text(content)
    return pkg_dir


def build_orchestrator(pkg_dir: Path) -> Agent[None, MigrationSummary]:
    """Build the orchestrator plus its three confined sub-agents over the planted package."""
    migrator = Agent(
        MODEL,
        name='migrator',
        output_type=MigrationReport,
        instructions=(
            'You migrate ONE Python file from os.path to pathlib. The task gives a file path '
            'relative to the package root, and may also list reviewer issues from a previous '
            'attempt. Read that file, rewrite it to use pathlib.Path while preserving behaviour '
            "exactly — including each function's return type, so a function that returns a str "
            'path must keep returning a str (e.g. str(Path(...))). Fix any listed reviewer issues. '
            'Write the result back to the same path and return the path and a one-line summary.'
        ),
        capabilities=[FileSystem(root_dir=pkg_dir)],  # may read and write inside the temp package
    )
    reviewer = Agent(
        MODEL,
        name='reviewer',
        output_type=Review,
        instructions=(
            'You review ONE migrated file. The task gives its path. Read it and approve only if it '
            'no longer imports `os`/`os.path` or calls any `os.path.*` function, and behaviour is '
            "preserved exactly — including each function's return type (a function that returned a "
            'str path must still return a str, not a Path). If anything is wrong, set '
            'approved=False and list concrete issues.'
        ),
        capabilities=[FileSystem(root_dir=pkg_dir, protected_patterns=['**'])],  # read-only
    )
    synthesizer = Agent(
        MODEL,
        name='synthesizer',
        output_type=MigrationSummary,
        instructions=(
            'You are given the per-file migration outcomes as JSON (each has a path, whether it was '
            'approved, and any outstanding issues). Write a short report: a one-paragraph summary '
            'of the migration, and one line per file giving its path and final status.'
        ),
    )
    return Agent(
        MODEL,
        output_type=MigrationSummary,
        instructions=(
            'A small package still uses os.path and must move to pathlib. Use run_workflow ONCE to '
            'migrate every file. In the script: for each file, call migrator(task=<path>), then '
            'reviewer(task=<path>); if the review is not approved, call migrator again with the '
            "reviewer's issues appended to the task, for up to 2 extra rounds. Run the files in "
            'parallel with asyncio.gather. Collect the final outcome for each file, pass them all '
            'to synthesizer(task=<json outcomes>), and return its report unchanged.'
        ),
        capabilities=[
            DynamicWorkflow(
                agents=[
                    WorkflowAgent(
                        agent=migrator, description='Migrates one file to pathlib in place; returns {path, summary}.'
                    ),
                    WorkflowAgent(agent=reviewer, description='Reviews one migrated file; returns {approved, issues}.'),
                    WorkflowAgent(
                        agent=synthesizer,
                        description='Summarizes the per-file outcomes (given as JSON) into a report.',
                    ),
                ],
                # Hard worst-case ceiling on the whole fan-out. Each sub-agent runs sequentially, so
                # its own token limit is enforced exactly; with max_agent_calls=N the tree makes at
                # most N sub-agent calls, spending at most N * 500k tokens. Worst case here is 3
                # files * (3 migrate + 3 review) + 1 synthesize = 19 calls, so 30 leaves headroom.
                # (forward_usage=True would instead share one counter, but its limit is best-effort
                # under concurrent fan-out, and the default request_limit=50 is far too low here.)
                max_agent_calls=30,
                forward_usage=False,
                sub_agent_usage_limits=UsageLimits(total_tokens_limit=500_000),
            )
        ],
    )


def extract_workflow_script(messages: list[ModelMessage]) -> str | None:
    """Pull the Python from the last `run_workflow` call (the one that succeeded), for display."""
    script: str | None = None
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == 'run_workflow':
                    args = part.args
                    if isinstance(args, dict):
                        code = args.get('code')
                        if isinstance(code, str):
                            script = code
    return script


async def main() -> None:
    """Plant the package, run the migration in one tool call, and show what happened."""
    pkg_dir = plant_sources()
    try:
        print(f'Planted a 3-file os.path package in {pkg_dir}\n')
        for name in SOURCES:
            print(f'  {name}:')
            print(textwrap.indent((pkg_dir / name).read_text().rstrip(), '    '))
            print()

        if not os.environ.get('ANTHROPIC_API_KEY'):
            print('Set ANTHROPIC_API_KEY (and LOGFIRE_TOKEN for a shareable trace) and re-run to watch')
            print('the orchestrator migrate all three files in a single run_workflow call.')
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

        # `result.output` is a typed MigrationSummary, not a free-form string — the synthesizer's
        # structured report survives the round-trip back through the orchestrator.
        report = result.output
        print(f'The orchestrator returned a typed {type(report).__name__} in {result.usage.requests} request(s):\n')
        print(f'  {report.summary}\n')
        for line in report.files:
            print(f'  - {line}')
    finally:
        shutil.rmtree(pkg_dir, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(main())
