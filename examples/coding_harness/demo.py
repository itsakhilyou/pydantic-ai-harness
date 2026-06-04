"""End-to-end demo: the harness fixes a real bug in a scratch repository.

This drives the agent with a scripted `FunctionModel` instead of a hosted model,
so the workflow is deterministic and runs with no API key. The point is to prove
that the *capabilities* wire together into a working coding loop: the scripted
"model" only chooses which tools to call, and every file read, edit, test run,
and plan update is executed by the real harness capabilities against a real repo
on disk.

Run it:
    python -m coding_harness.demo        # from the examples/ directory
    python examples/coding_harness/demo.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .agent import build_coding_agent

# A scratch project: a calculator with an off-by-one bug in `factorial`, plus a
# test that catches it. A correct agent reads the file, fixes the range, and
# reruns the test to green.
BUGGY_SOURCE = """\
def factorial(n):
    result = 1
    for i in range(1, n):  # bug: should be range(1, n + 1)
        result *= i
    return result
"""

TEST_SOURCE = """\
from calculator import factorial


def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120
"""

FIXED_LINE = '    for i in range(1, n + 1):\n'


def _scripted_model() -> FunctionModel:
    """A model that plays out a realistic fix-the-failing-test workflow."""
    step = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'write_plan',
                        {
                            'items': [
                                {'content': 'Run the test to see the failure', 'status': 'in_progress'},
                                {'content': 'Read calculator.py and locate the bug', 'status': 'pending'},
                                {'content': 'Fix factorial and rerun the test', 'status': 'pending'},
                            ]
                        },
                        tool_call_id='plan-1',
                    )
                ]
            )
        if step == 2:
            return ModelResponse(
                parts=[ToolCallPart('run_command', {'command': 'python -m pytest -q'}, tool_call_id='test-1')]
            )
        if step == 3:
            return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'calculator.py'}, tool_call_id='read-1')])
        if step == 4:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'edit_file',
                        {
                            'path': 'calculator.py',
                            'old_text': '    for i in range(1, n):  # bug: should be range(1, n + 1)\n',
                            'new_text': FIXED_LINE,
                        },
                        tool_call_id='edit-1',
                    )
                ]
            )
        if step == 5:
            return ModelResponse(
                parts=[ToolCallPart('run_command', {'command': 'python -m pytest -q'}, tool_call_id='test-2')]
            )
        return ModelResponse(parts=[TextPart('Fixed the off-by-one in factorial; pytest passes (2 passed).')])

    return FunctionModel(respond)


def _make_repo(root: Path) -> None:
    (root / 'calculator.py').write_text(BUGGY_SOURCE)
    (root / 'test_calculator.py').write_text(TEST_SOURCE)


async def run_demo(root: Path) -> tuple[str, list[str], list[str]]:
    """Build the scratch repo, run the harness, and return (report, tool calls, run_command outputs)."""
    _make_repo(root)
    agent = build_coding_agent(root, model=_scripted_model(), include_subagents=False)
    async with agent:
        result = await agent.run('The test suite is failing. Find and fix the bug, and prove the tests pass.')

    tool_calls = [
        f'{part.tool_name}({_short(part.args)})'
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    test_returns = [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_command'
    ]
    return result.output, tool_calls, test_returns


def _short(args: object) -> str:
    text = str(args)
    return text if len(text) <= 60 else text[:57] + '...'


async def main() -> None:
    """Run the demo in a temporary repo and print the workflow, asserting the bug was fixed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output, tool_calls, test_returns = await run_demo(root)

        print('=== tool calls the agent issued (executed by real capabilities) ===')
        for call in tool_calls:
            print(f'  - {call}')

        print('\n=== first test run (before fix) ===')
        print(_indent(test_returns[0]))
        print('=== final test run (after fix) ===')
        print(_indent(test_returns[-1]))

        final_source = (root / 'calculator.py').read_text()
        fixed = FIXED_LINE.strip() in final_source
        passed = 'passed' in test_returns[-1] and 'failed' not in test_returns[-1]

        print('\n=== agent report ===')
        print(output)
        print('\n=== checks ===')
        print(f'  file actually edited on disk: {fixed}')
        print(f'  final pytest run is green:    {passed}')
        if not (fixed and passed):
            raise SystemExit('DEMO FAILED: harness did not fix the bug')
        print('\nDEMO PASSED: the harness fixed a real bug and verified it with the test suite.')


def _indent(text: str) -> str:
    return '\n'.join(f'    {line}' for line in text.splitlines())


if __name__ == '__main__':
    asyncio.run(main())
