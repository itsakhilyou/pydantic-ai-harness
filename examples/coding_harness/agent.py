"""A real coding agent composed from pydantic-ai-harness capabilities.

The harness wires together the capabilities that ship (or are in review) in
`pydantic_ai_harness` into a single agent that can navigate a repository, edit
files, run commands and tests, keep a task plan, and delegate focused subtasks
to specialized sub-agents.

The capability set is the product here. `build_coding_agent` only supplies the
glue: a root directory, a system prompt, and the choice of which capabilities to
turn on.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from pydantic_ai_harness import FileSystem, Planning, Shell, SubAgents

CODING_INSTRUCTIONS = """\
You are a coding agent working inside a single repository. You can read, search,
and edit files, run shell commands, and keep a task plan.

Work like a careful engineer:

1. Before changing anything, understand the task. Use `read_file`, `search_files`,
   and `find_files` to inspect the code. Read a file before you edit it.
2. For any task with more than one step, call `write_plan` to record the plan as a
   list of steps, and keep exactly one step `in_progress` as you go. Update the
   plan when the situation changes.
3. Make the smallest change that solves the task. Prefer `edit_file` for targeted
   edits over rewriting a whole file with `write_file`.
4. Verify your work. Run the project's tests or the relevant command with
   `run_command` and read the output. If it fails, read the error, fix it, and
   run again. Do not claim success until you have seen it pass.
5. When the task involves focused investigation or review that would clutter your
   own context, delegate it with `delegate_task` to a sub-agent.

Be concise. Report what you changed and the evidence that it works.
"""

REVIEWER_INSTRUCTIONS = """\
You review code in this repository. Read the files you are asked about and report
concrete issues: bugs, missing edge cases, and risks. Cite file paths and line
content. Do not edit anything; return findings only.
"""

RESEARCHER_INSTRUCTIONS = """\
You answer questions about this repository by searching and reading its files.
Return a focused answer with the file paths and snippets that support it. Do not
edit anything.
"""


def build_coding_agent(
    root: str | Path,
    *,
    model: Model | str = 'anthropic:claude-sonnet-4-6',
    include_subagents: bool = True,
    persist_cwd: bool = True,
) -> Agent[None, str]:
    """Build a coding agent rooted at `root`.

    `model` accepts a model string for real use or a `Model` instance (e.g. a
    scripted `FunctionModel`) for deterministic tests. Sub-agents are enabled by
    default and share the same model and repository root.
    """
    root = Path(root)

    capabilities: list[AbstractCapability[None]] = [
        FileSystem(root_dir=root),
        Shell(cwd=root, persist_cwd=persist_cwd),
        Planning(),
    ]

    if include_subagents:
        reviewer = Agent(
            model,
            name='reviewer',
            description='Reviews code in the repository and reports bugs and risks without editing.',
            instructions=REVIEWER_INSTRUCTIONS,
            capabilities=[FileSystem(root_dir=root)],
        )
        researcher = Agent(
            model,
            name='researcher',
            description='Searches and reads the repository to answer questions, without editing.',
            instructions=RESEARCHER_INSTRUCTIONS,
            capabilities=[FileSystem(root_dir=root)],
        )
        capabilities.append(SubAgents(agents={'reviewer': reviewer, 'researcher': researcher}))

    return Agent(
        model,
        instructions=CODING_INSTRUCTIONS,
        capabilities=capabilities,
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
