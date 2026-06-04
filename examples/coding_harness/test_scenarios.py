"""End-to-end scenarios for the coding harness.

Each test sets up a real scratch repository, drives the harness with a scripted
model (so it runs offline and deterministically), and asserts on real outcomes:
files actually changed on disk and `pytest`/commands actually run. The scripted
model only chooses tools; the harness capabilities do the work.

Run:
    PYTHONPATH=examples uv run pytest examples/coding_harness/test_scenarios.py -v
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings

from coding_harness.agent import build_coding_agent
from pydantic_ai_harness import FileSystem, Planning, Shell, SubAgents

# A turn is either a final text answer or a list of (tool_name, args) to call.
Turn = str | list[tuple[str, dict[str, object]]]


def scripted(turns: list[Turn]) -> FunctionModel:
    """Build a FunctionModel that plays the given turns in order."""
    index = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal index
        turn = turns[index]
        index += 1
        if isinstance(turn, str):
            return ModelResponse(parts=[TextPart(turn)])
        parts = [ToolCallPart(name, args, tool_call_id=f't{index}_{i}') for i, (name, args) in enumerate(turn)]
        return ModelResponse(parts=parts)

    return FunctionModel(respond)


def _run_command_outputs(result_messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in result_messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_command'
    ]


def _tool_names(result_messages: list[ModelMessage]) -> list[str]:
    return [part.tool_name for message in result_messages for part in message.parts if isinstance(part, ToolCallPart)]


def _green(pytest_output: str) -> bool:
    return 'passed' in pytest_output and 'failed' not in pytest_output and 'error' not in pytest_output.lower()


# --------------------------------------------------------------------------- #
# Scenario 1: fix a bug and prove it with the test suite
# --------------------------------------------------------------------------- #
async def _scenario_fix_bug(root: Path) -> None:
    (root / 'calculator.py').write_text(
        'def factorial(n):\n    result = 1\n    for i in range(1, n):  # off-by-one\n        result *= i\n    return result\n'
    )
    (root / 'test_calculator.py').write_text(
        'from calculator import factorial\n\n\ndef test_factorial():\n    assert factorial(0) == 1\n    assert factorial(5) == 120\n'
    )
    model = scripted(
        [
            [
                (
                    'write_plan',
                    {
                        'items': [
                            {'content': 'Reproduce the failure', 'status': 'in_progress'},
                            {'content': 'Fix factorial', 'status': 'pending'},
                            {'content': 'Verify green', 'status': 'pending'},
                        ]
                    },
                )
            ],
            [('run_command', {'command': 'python -m pytest -q'})],
            [('read_file', {'path': 'calculator.py'})],
            [
                (
                    'edit_file',
                    {
                        'path': 'calculator.py',
                        'old_text': '    for i in range(1, n):  # off-by-one\n',
                        'new_text': '    for i in range(1, n + 1):\n',
                    },
                )
            ],
            [('run_command', {'command': 'python -m pytest -q'})],
            'Fixed the off-by-one in factorial; pytest is green.',
        ]
    )
    agent = build_coding_agent(root, model=model, include_subagents=False)
    async with agent:
        result = await agent.run('Find and fix the failing test.')

    outputs = _run_command_outputs(result.all_messages())
    assert not _green(outputs[0]), 'first run should be red'
    assert _green(outputs[-1]), 'final run should be green'
    assert 'range(1, n + 1)' in (root / 'calculator.py').read_text()
    assert 'write_plan' in _tool_names(result.all_messages())


def test_fix_bug() -> None:
    """The harness reproduces a failing test, fixes the bug, and reruns it green."""
    asyncio.run(_scenario_fix_bug(_tmp()))


# --------------------------------------------------------------------------- #
# Scenario 2: update documentation (README)
# --------------------------------------------------------------------------- #
async def _scenario_update_readme(root: Path) -> None:
    (root / 'README.md').write_text('# MyLib\n\nVersion: 1.0.0\n\nA small library.\n')
    model = scripted(
        [
            [('read_file', {'path': 'README.md'})],
            [('edit_file', {'path': 'README.md', 'old_text': 'Version: 1.0.0', 'new_text': 'Version: 2.0.0'})],
            [
                (
                    'edit_file',
                    {
                        'path': 'README.md',
                        'old_text': 'A small library.\n',
                        'new_text': 'A small library.\n\n## Usage\n\n```python\nimport mylib\n```\n',
                    },
                )
            ],
            'Bumped the version to 2.0.0 and added a Usage section.',
        ]
    )
    agent = build_coding_agent(root, model=model, include_subagents=False)
    async with agent:
        await agent.run('Bump the README version to 2.0.0 and add a Usage section.')

    readme = (root / 'README.md').read_text()
    assert 'Version: 2.0.0' in readme
    assert '1.0.0' not in readme
    assert '## Usage' in readme


def test_update_readme() -> None:
    """The harness edits a README: bumps the version and adds a section."""
    asyncio.run(_scenario_update_readme(_tmp()))


# --------------------------------------------------------------------------- #
# Scenario 3: implement a feature from a failing test (solve a problem)
# --------------------------------------------------------------------------- #
async def _scenario_implement_feature(root: Path) -> None:
    (root / 'slugify.py').write_text('def slugify(text):\n    raise NotImplementedError\n')
    (root / 'test_slugify.py').write_text(
        "from slugify import slugify\n\n\ndef test_slugify():\n    assert slugify('Hello World!') == 'hello-world'\n    assert slugify('  Foo_Bar ') == 'foo-bar'\n"
    )
    impl = "import re\n\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
    model = scripted(
        [
            [
                (
                    'write_plan',
                    {
                        'items': [
                            {'content': 'Read the failing test', 'status': 'in_progress'},
                            {'content': 'Implement slugify', 'status': 'pending'},
                            {'content': 'Run tests', 'status': 'pending'},
                        ]
                    },
                )
            ],
            [('run_command', {'command': 'python -m pytest -q'})],
            [('read_file', {'path': 'test_slugify.py'})],
            [
                (
                    'edit_file',
                    {
                        'path': 'slugify.py',
                        'old_text': 'def slugify(text):\n    raise NotImplementedError\n',
                        'new_text': impl,
                    },
                )
            ],
            [('run_command', {'command': 'python -m pytest -q'})],
            'Implemented slugify; both assertions pass.',
        ]
    )
    agent = build_coding_agent(root, model=model, include_subagents=False)
    async with agent:
        result = await agent.run('Implement slugify so the tests pass.')

    outputs = _run_command_outputs(result.all_messages())
    assert not _green(outputs[0])
    assert _green(outputs[-1])
    assert 're.sub' in (root / 'slugify.py').read_text()


def test_implement_feature() -> None:
    """The harness implements a function from a failing test until it passes."""
    asyncio.run(_scenario_implement_feature(_tmp()))


# --------------------------------------------------------------------------- #
# Scenario 4: search the codebase and refactor across files
# --------------------------------------------------------------------------- #
async def _scenario_search_and_refactor(root: Path) -> None:
    (root / 'geometry.py').write_text('def area_of_circle(r):\n    return 3.14 * r * r\n')
    (root / 'app.py').write_text('from geometry import area_of_circle\n\n\ndef main():\n    return area_of_circle(2)\n')
    (root / 'test_geometry.py').write_text(
        'from geometry import circle_area\n\n\ndef test_circle_area():\n    assert circle_area(1) == 3.14\n'
    )
    model = scripted(
        [
            [('search_files', {'pattern': 'area_of_circle'})],
            [
                (
                    'edit_file',
                    {'path': 'geometry.py', 'old_text': 'def area_of_circle(r):', 'new_text': 'def circle_area(r):'},
                )
            ],
            [
                (
                    'edit_file',
                    {
                        'path': 'app.py',
                        'old_text': 'from geometry import area_of_circle',
                        'new_text': 'from geometry import circle_area',
                    },
                )
            ],
            [
                (
                    'edit_file',
                    {'path': 'app.py', 'old_text': 'return area_of_circle(2)', 'new_text': 'return circle_area(2)'},
                )
            ],
            [('run_command', {'command': 'python -m pytest -q'})],
            'Renamed area_of_circle to circle_area across geometry.py and app.py; tests pass.',
        ]
    )
    agent = build_coding_agent(root, model=model, include_subagents=False)
    async with agent:
        result = await agent.run('Rename area_of_circle to circle_area everywhere.')

    assert 'area_of_circle' not in (root / 'geometry.py').read_text()
    assert 'area_of_circle' not in (root / 'app.py').read_text()
    assert _green(_run_command_outputs(result.all_messages())[-1])


def test_search_and_refactor() -> None:
    """The harness searches the repo and renames a symbol across files."""
    asyncio.run(_scenario_search_and_refactor(_tmp()))


# --------------------------------------------------------------------------- #
# Scenario 5: delegate research to a sub-agent, then act on the answer
# --------------------------------------------------------------------------- #
async def _scenario_delegate_to_subagent(root: Path) -> None:
    (root / 'config.py').write_text('# runtime configuration\nMAX_RETRIES = 3\nTIMEOUT = 30\n')

    researcher_model = scripted(
        [
            [('search_files', {'pattern': 'MAX_RETRIES'})],
            [('read_file', {'path': 'config.py'})],
            'MAX_RETRIES is defined in config.py with value 3.',
        ]
    )
    researcher = Agent(
        researcher_model,
        name='researcher',
        description='Searches and reads the repo to answer questions.',
        capabilities=[FileSystem(root_dir=root)],
    )

    main_model = scripted(
        [
            [
                (
                    'delegate_task',
                    {'agent_name': 'researcher', 'task': 'Where is MAX_RETRIES defined and what is its value?'},
                )
            ],
            [('edit_file', {'path': 'config.py', 'old_text': 'MAX_RETRIES = 3', 'new_text': 'MAX_RETRIES = 5'})],
            'The researcher found MAX_RETRIES in config.py (was 3); I bumped it to 5.',
        ]
    )
    agent = Agent(
        main_model,
        capabilities=[
            FileSystem(root_dir=root),
            Shell(cwd=root),
            Planning(),
            SubAgents(agents={'researcher': researcher}),
        ],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    async with agent:
        result = await agent.run('Find MAX_RETRIES via the researcher and bump it to 5.')

    assert 'delegate_task' in _tool_names(result.all_messages())
    assert 'MAX_RETRIES = 5' in (root / 'config.py').read_text()


def test_delegate_to_subagent() -> None:
    """The main agent delegates research to a sub-agent, then acts on the answer."""
    asyncio.run(_scenario_delegate_to_subagent(_tmp()))


# --------------------------------------------------------------------------- #
# Scenario 6: a guard rail -- the agent cannot escape the repository root
# --------------------------------------------------------------------------- #
async def _scenario_root_confinement(root: Path) -> str:
    (root / 'inside.txt').write_text('safe')
    model = scripted(
        [
            [('read_file', {'path': '../../../../etc/passwd'})],
            'I could not read outside the repository root.',
        ]
    )
    agent = build_coding_agent(root, model=model, include_subagents=False)
    async with agent:
        result = await agent.run('Read /etc/passwd.')
    return '\n'.join(
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    )


def test_root_confinement() -> None:
    """The FileSystem capability blocks reads outside the repository root."""
    feedback = asyncio.run(_scenario_root_confinement(_tmp()))
    # The FileSystem capability refuses paths outside the root: the model gets an
    # error back, and no actual /etc/passwd content (a `root:` line) leaks through.
    assert 'root:' not in feedback
    assert 'allowed' in feedback.lower() or 'outside' in feedback.lower() or 'denied' in feedback.lower()


def _tmp() -> Path:
    """A scratch repo directory for one scenario."""
    return Path(tempfile.mkdtemp(prefix='harness-scenario-'))
