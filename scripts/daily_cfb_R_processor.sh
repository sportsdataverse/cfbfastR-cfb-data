#!/bin/bash
# Compile cfbfastR-cfb-data datasets from cfbfastR-cfb-raw final JSON, per season.
# Each R creation script reshapes one block; per-game tryCatch keeps partial output usable.
set -uo pipefail

while getopts s:e: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
  esac
done
END_YEAR=${END_YEAR:-$START_YEAR}

SCRIPTS=(
  R/espn_cfb_01_pbp_creation.R
  R/espn_cfb_02_team_box_creation.R
  R/espn_cfb_03_player_box_creation.R
  R/espn_cfb_04_adv_box_creation.R
  R/espn_cfb_05_play_participants_creation.R
  R/espn_cfb_06_drives_creation.R
  R/espn_cfb_07_game_rosters_creation.R
  R/espn_cfb_09_betting_creation.R
  R/espn_cfb_10_schedules_creation.R
  R/espn_cfb_11_linescores_creation.R
  R/espn_cfb_12_power_index_creation.R
  R/espn_cfb_14_injuries_creation.R
  R/espn_cfb_15_team_summaries_creation.R
)

mkdir -p logs
ANY_FAILED=0

# Commit + push, surviving a remote that moved while the build was running.
#
# The previous form pulled BEFORE staging, which can only ever abort: the build
# has just rewritten the tracked parquet/csv files, so `git pull` refuses with
# "Your local changes would be overwritten by merge". It then committed anyway,
# pushed into a non-fast-forward rejection, and swallowed all of it in
# `>/dev/null` with no rc check -- a GREEN job that published nothing. See
# wehoop-wnba-data runs 32192069433 + 32192069566 (2026-08-18).
#
# Order matters: stage and commit FIRST so the tree is clean, and only then
# reconcile with origin. `rebase --merge` rather than `pull --rebase` because
# git's default am backend base64-encodes every parquet blob it replays.
sdv_commit_push() {
  local msg="$1"; shift
  git add -- "$@" >/dev/null 2>&1 || true
  if git diff --cached --quiet; then
    echo "nothing to commit for: $msg"
    return 0
  fi
  git commit -m "$msg" >/dev/null || { echo "::warning ::commit failed: $msg"; return 1; }

  local attempt
  for attempt in 1 2 3; do
    if git push origin HEAD >/dev/null 2>&1; then
      echo "pushed: $msg (attempt $attempt)"
      return 0
    fi
    echo "push rejected (attempt $attempt); syncing with origin"
    git fetch --quiet origin main || true
    if ! git rebase --merge origin/main >/dev/null 2>&1; then
      git rebase --abort >/dev/null 2>&1 || true
      echo "::error ::cannot rebase onto origin/main for: $msg"
      return 1
    fi
  done
  echo "::error ::push still rejected after 3 attempts: $msg"
  return 1
}

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  LOGFILE="logs/cfbfastR_cfb_data_logfile_${i}.log"
  TMPLOG=$(mktemp "/tmp/cfbfastR_cfb_data_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    SEASON_RC=0
    for SCRIPT in "${SCRIPTS[@]}"; do
      Rscript "$SCRIPT" -s "$i" -e "$i" || {
        rc=$?; echo "::warning ::$SCRIPT for season $i exited with code $rc"; SEASON_RC=$rc
      }
    done
    echo "RSCRIPT_RC=$SEASON_RC" > "/tmp/_rc_${i}"
    sdv_commit_push "CFB Data Updated (Start: $i End: $i)" cfb || PUSH_RC=1
  } 2>&1 | tee "$TMPLOG"

  RSCRIPT_RC=$(sed 's/RSCRIPT_RC=//' "/tmp/_rc_${i}" 2>/dev/null); rm -f "/tmp/_rc_${i}"
  cp "$TMPLOG" "$LOGFILE"
  sdv_commit_push "CFB Data log update (Start: $i End: $i)" "$LOGFILE" || PUSH_RC=1
  rm -f "$TMPLOG"
  if [ "${RSCRIPT_RC:-0}" != "0" ]; then
    echo "::error ::A creation script for season $i exited with code $RSCRIPT_RC"
    ANY_FAILED=1
  fi
done

Rscript R/run_summary.R -s "$START_YEAR" -e "$END_YEAR" || true
# A rejected push is a FAILED run, not a green one. Release assets upload on a
# separate path and can succeed while the repo mirror is left stale.
if [ "${PUSH_RC:-0}" != "0" ]; then
  echo "::error ::At least one commit failed to reach origin; the repo mirror is stale."
  exit 1
fi
[ "${ANY_FAILED:-0}" = "0" ] || exit 1
