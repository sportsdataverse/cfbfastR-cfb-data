#!/usr/bin/env bash
# Render every docs/models/*.qmd to committed GFM + figures beside the sources.
# docs/models/ is ALSO a Quarto website project (_quarto.yml -> _site html);
# --output-dir . overrides the project output dir so the GFM lands in-tree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QUARTO_PYTHON="$ROOT/.venv/Scripts/python.exe"
QUARTO_BIN="${QUARTO_BIN:-quarto}"
command -v "$QUARTO_BIN" >/dev/null 2>&1 || QUARTO_BIN="$LOCALAPPDATA/Programs/Quarto/bin/quarto.cmd"
for q in "$ROOT"/docs/models/*.qmd; do
  case "$(basename "$q")" in index.qmd) continue;; esac
  echo "== rendering $q"
  "$QUARTO_BIN" render "$q" --to gfm --output-dir .
done
