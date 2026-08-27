#!/usr/bin/env bash
# Build (and optionally publish) the three recruiting datasets.
#
#   cfb_recruits              per-recruit rows          2002+
#   cfb_team_talent           per team-season composite 2005+
#   cfb_returning_production  per team-season shares    2005+
#
# MIN SEASONS ARE MEASURED, NOT ASSUMED:
#   * 247 returns rows back to 1996, but composite ratings only become usable in
#     2002. 2000 and 2001 carry ratings for the top of the class and then
#     collapse (2001: 52% on page 1, 0% by page 4), so a class built from them
#     would be mostly unrated and silently understate every team.
#   * talent accumulates a 4-class window, so its floor is 2002 + 3 = 2005.
#   * returning production needs the season S-1 ESPN player box (floor 2004)
#     and the season S roster (floor 2003), so its floor is 2005. Verified
#     live: 2005 -> 161 teams, off_returning mean 0.638 / sd 0.259.
#
# Usage:
#   scripts/10_build_recruiting.sh                      # build all, no publish
#   scripts/10_build_recruiting.sh --publish            # build + upload
#   scripts/10_build_recruiting.sh --dataset team_talent --start 2020 --end 2024
#
# recruits/team_talent read the raw 247 store from cfbfastR-cfb-raw; point at it
# with CFB_RAW_ROOT if it is not the default sibling checkout.
set -euo pipefail

# Resolve this repo's interpreter once (never `uv run` in a long build --
# it re-syncs the env mid-run). CFB_DATA_PY overrides.
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"

cd "$(dirname "$0")/.."

RAW_ROOT="${CFB_RAW_ROOT:-../cfbfastR-cfb-raw}"
DATASETS=""
START=""
END=""
PUBLISH=""
DRY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASETS="$2"; shift 2 ;;
    --start)   START="$2";    shift 2 ;;
    --end)     END="$2";      shift 2 ;;
    --publish) PUBLISH="--publish"; shift ;;
    --dry-run) DRY="--dry-run";     shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CURRENT_SEASON="$(date +%Y)"
# 10# forces base 10: bash reads a zero-padded "08" as octal and errors out.
[[ "$((10#$(date +%m)))" -lt 8 ]] && CURRENT_SEASON=$((CURRENT_SEASON - 1))

# per-dataset measured floor; a season below it cannot be built, not merely
# "has less data" -- see the header.
floor_for() {
  case "$1" in
    recruits)             echo 2002 ;;
    team_talent)          echo 2005 ;;
    returning_production) echo 2005 ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

[[ -z "$DATASETS" ]] && DATASETS="recruits team_talent returning_production"

rc=0
for ds in $DATASETS; do
  floor="$(floor_for "$ds")"
  s="${START:-$floor}"
  e="${END:-$CURRENT_SEASON}"
  if [[ "$s" -lt "$floor" ]]; then
    echo "!! $ds: start $s is below the measured floor $floor -- clamping" >&2
    s="$floor"
  fi
  if [[ "$e" -lt "$s" ]]; then
    # Clamping the start can invert an explicitly-requested range. An inverted
    # range builds nothing while the loop still reports success, so say so.
    echo "!! $ds: end $e is below start $s -- nothing to build, skipping" >&2
    continue
  fi
  echo "=== $ds ${s}-${e} ==="
  # each season is isolated inside the driver; a bad one is reported, not fatal,
  # so one gap cannot abandon the rest of the sweep
  PYTHONPATH=python "$PY" -m cfb_data_build \
    --dataset "$ds" \
    --start-year "$s" \
    --end-year "$e" \
    --raw-root "$RAW_ROOT" \
    ${PUBLISH} ${DRY} || rc=1
done

exit "$rc"
