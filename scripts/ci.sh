#!/usr/bin/env bash
# Local CI runner mirroring .github/workflows/ci.yml.
#
# Usage:
#   bash scripts/ci.sh
#
# Exit code is non-zero if any step fails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() {
    printf '\n\033[1;34m==> %s\033[0m\n' "$1"
}

step "Install package and dev dependencies"
python -m pip install -e ".[dev]"

step "Lint with ruff"
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/

step "Type check with mypy"
python -m mypy src/igasgd

step "Run test suite"
python -m pytest tests/

step "Run demo smoke test"
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation > /dev/null

printf '\n\033[1;32mAll CI steps passed.\033[0m\n'