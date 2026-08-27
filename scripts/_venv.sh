#!/usr/bin/env bash
# Interpreter resolution for this repo's drivers. Source it, do not execute it:
#
#     source "$(dirname "$0")/_venv.sh"
#     "$PY" python/espn_cfb_01_pbp_creation.py -s 2026 -e 2026
#
# Why not `uv run`: it re-resolves and can RE-SYNC the environment mid-sweep,
# which on a multi-hour scrape swaps the interpreter under a running job. Keep
# `uv run` for tests and lint, never for a long-running entry point.
#
# Order: SDV_CFB_DATA_PYTHON override -> this repo's .venv -> loud failure.
# Never fall back to a bare `python`: pilots found drivers silently binding a
# SIBLING repo's venv, which is how a scrape runs against the wrong deps.

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${SDV_CFB_DATA_PYTHON:-}" ]; then
  PY="$SDV_CFB_DATA_PYTHON"
elif [ -x "$_repo_root/.venv/Scripts/python.exe" ]; then   # Windows layout
  PY="$_repo_root/.venv/Scripts/python.exe"
elif [ -x "$_repo_root/.venv/bin/python" ]; then           # POSIX layout
  PY="$_repo_root/.venv/bin/python"
else
  echo "::error ::no interpreter: create it with 'uv sync --frozen' in $_repo_root," >&2
  echo "         or set SDV_CFB_DATA_PYTHON to an explicit python." >&2
  return 1 2>/dev/null || exit 1
fi

if [ ! -x "$PY" ]; then
  echo "::error ::interpreter not executable: $PY" >&2
  return 1 2>/dev/null || exit 1
fi

export PY
echo "interpreter: $PY"
