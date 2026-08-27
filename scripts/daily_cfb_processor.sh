#!/bin/bash
# Compile cfbfastR-cfb-data datasets, per season.
#
# EVERY dataset is now built by the Python `cfb_data_build` CLI (parity-validated
# port of the R espn_cfb_01..15 creation scripts). The last R producer,
# team_summaries (espn_cfb_15), was retired once the P2 pbp rebuild made the
# Python season-pbp source current -- its stated blocker was a stale
# load_cfb_pbp, which now returns a full current season. Only run_summary.R
# remains, and that only PRINTS a report; it produces no dataset.
#
# Per-dataset failures are GitHub warnings (non-fatal); a non-zero season RC
# fails the job at the end. `pbp` fetches the season's final.json into the
# Python cache; every later dataset reuses it via --no-fetch.
set -uo pipefail

# Resolve this repo's interpreter once (never `uv run` in a long build --
# it re-syncs the env mid-run). CFB_DATA_PY overrides.
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"

while getopts s:e: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
  esac
done
END_YEAR=${END_YEAR:-$START_YEAR}

PY_FIRST="pbp"
PY_REST="play_participants team_box player_box drives game_rosters betting schedules linescores power_index injuries adv_team adv_passing adv_rushing adv_receiving adv_defensive adv_turnover adv_drives adv_situational adv_defensive_players adv_specialists"
# Derived datasets -- each reads an artifact an earlier step produced, so order
# matters: gamelog <- adv_team.
PY_DERIVED="gamelog"
# The UNIFIED schedule published to the `cfb_schedules` tag (what sdv-py
# `load_cfb_schedule` reads): the CFBD all-division superset UNIONed with the
# ESPN-only rows, enriched with the ESPN-native fields. It reads the `schedules`
# artifact this run just wrote, so it must follow PY_REST. Needs CFBD_API_KEY.
PY_UNIFIED_SCHEDULES="cfb_schedules"
# ESPN-native season rosters. Compiled straight from the raw per-game roster
# blocks over HTTP (not from an earlier step), with a REAL resolved position.
# Supersedes the legacy `rosters` dataset and R/espn_cfb_08_rosters_creation.R,
# both of which published href-only positions under a second naming stem on the
# same tag; neither runs from a driver any more.
PY_ROSTERS="cfb_rosters"
# Weekly long-format snapshots read the summaries/ratings output, so they run
# last of all.
PY_WEEKLY="ratings_weekly team_summaries_weekly"
# Roster continuity. `recruits`/`team_talent` read the 247 raw store in
# cfbfastR-cfb-raw; `returning_production` reads the ESPN player box and needs
# no store. `recruiting_proj` consumes talent + returning, so it runs after
# both. run_py does `cd python`, so the raw root must be absolute or resolve
# from there -- CFB_RAW_ROOT is exported below for exactly that reason.
PY_RECRUITING="recruits team_talent returning_production"
export CFB_RAW_ROOT="${CFB_RAW_ROOT:-$(cd "$(dirname "$0")/../../cfbfastR-cfb-raw" 2>/dev/null && pwd)}"

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

    # Build one Python dataset (writes to the repo-root cfb/ via --base ../cfb).
    run_py() {
      local ds="$1"; shift
      (cd python && "$PY" -m cfb_data_build --dataset "$ds" --base ../cfb -s "$i" -e "$i" "$@") || {
        rc=$?; echo "::warning ::cfb_data_build $ds for season $i exited with code $rc"; SEASON_RC=$rc
      }
    }

    run_py "$PY_FIRST" --publish
    for ds in $PY_REST; do run_py "$ds" --no-fetch --publish; done
    for ds in $PY_DERIVED; do run_py "$ds" --no-fetch --publish; done
    for ds in $PY_UNIFIED_SCHEDULES; do run_py "$ds" --publish; done
    for ds in $PY_ROSTERS; do run_py "$ds" --publish; done

    # The 5-table summaries family. This ran on R (espn_cfb_15) because the
    # Python season-pbp source was stale for the current season; the P2 pbp
    # rebuild fixed that -- load_cfb_pbp(2025) now returns 165,850 rows / 956
    # games and the Python driver builds all 5 tables for 2025. R retired here.
    run_py summaries --publish

    for ds in $PY_WEEKLY; do run_py "$ds" --no-fetch --publish; done

    # Recruiting last: it depends on nothing above, and a 247 outage must not
    # cost the game datasets. Per-dataset failures are already non-fatal here.
    # Guard on the input this family ACTUALLY reads (cfb/recruits/json), not on
    # the raw root merely existing. CI now materialises ../cfbfastR-cfb-raw with
    # ONLY cfb_schedule_master.parquet in it (for ratings_weekly /
    # team_summaries_weekly), so `-d "$CFB_RAW_ROOT"` became true while the
    # recruits store was still absent -- and recruits/team_talent went from
    # cleanly skipped to hard failures:
    #   cfb_recruits 2026: FAILED (raw store is missing complete classes [2026])
    # One env var, two unrelated inputs; each guard has to check its own.
    if [[ -n "$CFB_RAW_ROOT" && -d "$CFB_RAW_ROOT/cfb/recruits/json" ]]; then
      for ds in $PY_RECRUITING; do run_py "$ds" --publish; done
    else
      echo "::warning::no 247 recruits store at ${CFB_RAW_ROOT:-<unset>}/cfb/recruits/json; skipping recruiting datasets"
      # returning_production reads no raw store, so it can still run
      run_py returning_production --publish
    fi

    echo "RSCRIPT_RC=$SEASON_RC" > "/tmp/_rc_${i}"
    sdv_commit_push "CFB Data Updated (Start: $i End: $i)" cfb || PUSH_RC=1
  } 2>&1 | tee "$TMPLOG"

  RSCRIPT_RC=$(sed 's/RSCRIPT_RC=//' "/tmp/_rc_${i}" 2>/dev/null); rm -f "/tmp/_rc_${i}"
  cp "$TMPLOG" "$LOGFILE"
  sdv_commit_push "CFB Data log update (Start: $i End: $i)" "$LOGFILE" || PUSH_RC=1
  rm -f "$TMPLOG"
  if [ "${RSCRIPT_RC:-0}" != "0" ]; then
    echo "::error ::A creation step for season $i exited with code $RSCRIPT_RC"
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
