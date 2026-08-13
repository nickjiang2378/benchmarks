#!/bin/bash
# Per-task container setup: history-free checkout of the base commit + venv.
# Usage: setup_task.sh <base_commit> [tests_commit]
#
# `suite` tasks need source at one commit and tests/ at another (the fix already
# applied, the PR's own tests rolled back). Passing tests_commit swaps tests/ for
# that commit's copy -- still a pure git snapshot, so the state is reproducible
# and the container never sees a patch file.
set -e
BASE="$1"
TESTS_FROM="$2"
mkdir -p /repo
git -C /opt/pydantic-src archive "$BASE" | tar -x -C /repo
if [ -n "$TESTS_FROM" ]; then
  rm -rf /repo/tests
  git -C /opt/pydantic-src archive "$TESTS_FROM" tests | tar -x -C /repo
fi
cd /repo
git init -q
git config user.email bench@local && git config user.name bench
git add -A && git commit -qm base && git tag BASE
# The full clone contains the fixing commit in its history -- remove it so the
# agent's container holds nothing newer than the base snapshot.
rm -rf /opt/pydantic-src
uv venv -q --python 3.12 .venv
DEPS="pytest==8.3.5 pytest-mock dirty-equals cloudpickle email-validator faker \
pytest-benchmark pytest-examples eval-type-backport jsonschema packaging rich \
pytest-run-parallel pytz"
# Fast path: resolve pydantic-core from PyPI wheels (--no-sources skips the
# Rust workspace build at 2025+ commits). Fall back to the workspace build.
uv pip install -q --compile-bytecode --python .venv/bin/python --no-sources -e . $DEPS \
  || uv pip install -q --compile-bytecode --python .venv/bin/python -e . $DEPS
# Warm caches so the agent's dozens of pytest invocations skip recompilation:
# byte-compile the editable source tree, then one throwaway collection to heat
# pytest/plugin/conftest imports and the OS page cache.
.venv/bin/python -m compileall -q /repo/pydantic || true
FIRST_TEST=$(ls tests/test_*.py 2>/dev/null | head -1)
[ -n "$FIRST_TEST" ] && .venv/bin/python -m pytest --co -q "$FIRST_TEST" >/dev/null 2>&1 || true
echo "SETUP_OK $BASE"
