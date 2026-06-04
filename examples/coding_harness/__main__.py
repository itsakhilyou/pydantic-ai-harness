"""CLI for the coding harness.

Usage:
    python -m coding_harness "Fix the failing test in calculator.py" --root /path/to/repo

Requires a model API key in the environment (e.g. ANTHROPIC_API_KEY) for real runs.
"""

from __future__ import annotations

import argparse
import asyncio

from pydantic_ai.messages import ToolReturnPart

from .agent import build_coding_agent


async def _run(task: str, root: str, model: str, no_subagents: bool) -> int:
    agent = build_coding_agent(root, model=model, include_subagents=not no_subagents)
    async with agent:
        result = await agent.run(task)

    print('\n=== result ===')
    print(result.output)

    plans = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'write_plan'
    ]
    if plans:
        print('\n=== final plan ===')
        print(plans[-1])

    usage = result.usage()
    print(f'\n=== usage ===\nrequests={usage.requests} tokens={usage.total_tokens}')
    return 0


def main() -> int:
    """Parse CLI arguments and run the coding agent against a repository."""
    parser = argparse.ArgumentParser(prog='coding_harness', description=__doc__)
    parser.add_argument('task', help='the task for the coding agent to perform')
    parser.add_argument('--root', default='.', help='repository root the agent operates in (default: .)')
    parser.add_argument('--model', default='anthropic:claude-sonnet-4-6', help='model to use')
    parser.add_argument('--no-subagents', action='store_true', help='disable the reviewer/researcher sub-agents')
    args = parser.parse_args()
    return asyncio.run(_run(args.task, args.root, args.model, args.no_subagents))


if __name__ == '__main__':
    raise SystemExit(main())
