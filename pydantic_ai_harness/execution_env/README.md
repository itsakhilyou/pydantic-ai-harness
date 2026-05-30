# Execution Environment

Give an agent filesystem and shell access — over a pluggable backend, so the same tools work whether the agent runs against your local machine or an isolated container.

> **Status: in development.** `read_file`, `write_file`/edit, explore (`ls`/`glob`/`grep`), and `shell` are complete and tested against `LocalEnvironment`. `DockerEnvironment` exists as an API skeleton — `environment='docker'` and `DockerEnvironment(...)` typecheck and wire through, but calling any tool today raises `NotImplementedError`. The real backend is coming next. This README describes the intended shape.

## The idea

A coding agent needs to do four things: read files, write/edit files, explore (list/search), and run shell commands. *Where* those happen — your laptop, a Docker container, a remote VM — should not change the agent's tools. `ExecutionEnv` is the capability that exposes those tools; an `Environment` is the swappable backend that actually performs them.

```
ExecutionEnv (capability)  ──provides tools to──▶  Agent
      │ delegates to
      ▼
AbstractEnvironment  ◀── LocalEnvironment | DockerEnvironment | …
```

The capability is written once; each backend implements the same operations its own way.

## Usage

The common case is one line — the agent runs against your **current working directory** with a local backend:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ExecutionEnv

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExecutionEnv()])

result = agent.run_sync('Read pyproject.toml and tell me the project name.')
print(result.output)
```

`ExecutionEnv()` defaults to `environment='local'`, which hands the agent your `cwd`. For container isolation (skeleton today; real implementation coming next), use the matching string:

```python
ExecutionEnv(environment='docker')
```

When you want to **configure** the backend — a different root, a specific Docker image, a custom backend — pass an `AbstractEnvironment` instance instead of a string:

```python
from pydantic_ai_harness.environments import DockerEnvironment, LocalEnvironment

# Local rooted somewhere other than cwd:
ExecutionEnv(environment=LocalEnvironment(root='/path/to/workspace'))

# Docker with a chosen image (skeleton today; real backend coming):
ExecutionEnv(environment=DockerEnvironment(image='python:3.12-slim'))
```

The agent's tools (`read_file`, `write_file`/`edit_file`, `ls`/`glob`/`grep`, `shell`) are the same on every backend — only the backend changes.

## Backends

| Backend | What it is | Use for |
|---|---|---|
| `LocalEnvironment` | Operations run against your local filesystem, rooted at `root`. | Trusted, local development. |
| `DockerEnvironment` *(skeleton; not yet usable)* | Operations run inside a container. The container is the isolation boundary. | Untrusted / model-generated code. |

## Security

**`LocalEnvironment` is not a security boundary.** Its `root` path jail is *advisory* — it catches accidental escapes but is bypassable (shell, symlinks, TOCTOU). Do not point it at a machine you care about while running untrusted code. For real isolation, use `DockerEnvironment`, where the container — not a path check — is the boundary. See [`agent_docs/confinement-security-research.md`](../../agent_docs/confinement-security-research.md).

## Running shell commands

The `shell` tool runs a command string in a real shell (`bash`, falling back to `sh`), so pipes, `&&`, globs, and `$VARS` all work. Each call runs in a **fresh process** rooted at `root` — no state (cwd, env, exported vars) persists between calls, so chain with `cd x && ...` in a single command when you need it.

A command that exits non-zero is **not** an error: the tool returns the output with the exit code noted, so the model can read the failure and react. An optional `timeout` (seconds, fractional allowed) kills the whole process **tree** — not just the top-level shell — so a command that backgrounds children can't leave orphans running; the model is told it timed out and gets whatever output was captured. The only hard failure is the environment being unable to start a shell at all, which surfaces loudly.

## How errors reach the model

The environment translates backend failures into a single `ExecutionEnvironmentError` taxonomy (uniform across backends). The capability then routes them: errors the model can fix by changing its argument (file not found, wrong path, a directory, non-UTF-8) become a `ModelRetry`, so the model gets another try; infrastructure failures propagate and surface loudly.

## Credits

The system prompt and tool descriptions are adopted, largely verbatim, from [**pi**](https://github.com/badlogic/pi-mono) by Mario Zechner ([@badlogic](https://github.com/badlogic)) and the pi-mono team (MIT-licensed). pi's coding-agent prompts are exceptionally well-tuned; rather than reinvent them, we stand on that work and credit it gratefully. The exact sources, pinned to a commit, are recorded in [`agent_docs/pi-prompts.md`](../../agent_docs/pi-prompts.md).
