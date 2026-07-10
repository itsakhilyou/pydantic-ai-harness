#!/usr/bin/env bash
# Bootstrap a Python environment for the CodeMode + pydantic-monty harness.
#
# pydantic-monty betas ship wheels up to cp313 (not cp314), so the beta cannot install on a 3.14
# interpreter -- this defaults to Python 3.13. Pass a monty spec to pin a specific build.
#
# Usage:
#   benchmarks/codemode/setup_env.sh [VENV_DIR] [PYTHON_VERSION]
#
# Environment overrides:
#   MONTY_SPEC   pip specs for monty, space-separated
#                (default: "pydantic-monty>=0.0.19b1 pydantic-monty-runtime>=0.0.19b1")
#   UV_OFFLINE   set to 1 to resolve only from the uv cache (no network)
#
# Examples:
#   benchmarks/codemode/setup_env.sh .venv-monty-beta 3.13
#   MONTY_SPEC="pydantic-monty==0.0.19b2 pydantic-monty-runtime==0.0.19b2" benchmarks/codemode/setup_env.sh
set -euo pipefail

VENV_DIR="${1:-.venv-monty-bench}"
PYTHON_VERSION="${2:-3.13}"
MONTY_SPEC="${MONTY_SPEC:-pydantic-monty>=0.0.19b1 pydantic-monty-runtime>=0.0.19b1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

offline_flag=()
if [[ "${UV_OFFLINE:-0}" == "1" ]]; then
  offline_flag=(--offline)
fi

echo "Creating $VENV_DIR on Python $PYTHON_VERSION ..."
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

# Monty betas carry a prerelease floor, so uv auto-allows prereleases for the frontend; the split
# `pydantic-monty-runtime` dep has no prerelease marker, so it is listed explicitly here. httpx is
# pinned below 1.0 so a global prerelease policy never drags in an httpx dev build.
echo "Installing harness + monty ($MONTY_SPEC) + bench deps ..."
UV_FROZEN=0 uv pip install "${offline_flag[@]}" --python "$VENV_DIR" \
  -e "${REPO_ROOT}[code-mode]" \
  ${MONTY_SPEC} \
  'httpx>=0.28.1,<1' \
  'pydantic-ai-slim[spec]>=2.1.0'

echo
"$VENV_DIR/bin/python" -c "import pydantic_monty; print('pydantic-monty', pydantic_monty.__version__)"
echo "Done. Run the harness with:"
echo "  $VENV_DIR/bin/python benchmarks/codemode/bench.py"
