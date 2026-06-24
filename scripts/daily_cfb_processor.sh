#!/bin/bash
# Compile cfbfastR-cfb-data datasets, per season.
#
# The per-game reshape datasets are built by the Python `cfb_data_build` CLI
# (parity-validated port of the R espn_cfb_01..14 creation scripts). Only
# `team_summaries` (espn_cfb_15) stays on R for now -- it is a season-level
# aggregation off the full cfbfastR season pbp, whose Python loader
# (sportsdataverse.cfb.load_cfb_pbp) is stale for the current season.
#
# Per-dataset failures are GitHub warnings (non-fatal); a non-zero season RC
# fails the job at the end. `pbp` fetches the season's final.json into the
# Python cache; every later dataset reuses it via --no-fetch.
set -uo pipefail

while getopts s:e: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
  esac
done
END_YEAR=${END_YEAR:-$START_YEAR}

PY_FIRST="pbp"
PY_REST="play_participants team_box player_box drives game_rosters betting schedules linescores power_index injuries adv_team adv_passing adv_rushing adv_receiving adv_defensive adv_turnover adv_drives adv_situational"
PY_DERIVED="rosters"   # derives from the game_rosters parquet -> must run after it

mkdir -p logs
ANY_FAILED=0
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
      (cd python && uv run python -m cfb_data_build --dataset "$ds" --base ../cfb -s "$i" -e "$i" "$@") || {
        rc=$?; echo "::warning ::cfb_data_build $ds for season $i exited with code $rc"; SEASON_RC=$rc
      }
    }

    run_py "$PY_FIRST" --publish
    for ds in $PY_REST; do run_py "$ds" --no-fetch --publish; done
    run_py "$PY_DERIVED" --no-fetch --publish

    # team_summaries (espn_cfb_15) stays on R until the Python season-pbp source is current.
    Rscript R/espn_cfb_15_team_summaries_creation.R -s "$i" -e "$i" || {
      rc=$?; echo "::warning ::team_summaries (R) for season $i exited with code $rc"; SEASON_RC=$rc
    }

    echo "RSCRIPT_RC=$SEASON_RC" > "/tmp/_rc_${i}"
    git pull >/dev/null
    git add cfb/* >/dev/null 2>&1 || true
    git commit -m "CFB Data Updated (Start: $i End: $i)" || echo "No changes to commit"
    git pull >/dev/null
    git push >/dev/null
  } 2>&1 | tee "$TMPLOG"

  RSCRIPT_RC=$(sed 's/RSCRIPT_RC=//' "/tmp/_rc_${i}" 2>/dev/null); rm -f "/tmp/_rc_${i}"
  cp "$TMPLOG" "$LOGFILE"
  git pull --rebase >/dev/null || true
  git add "$LOGFILE"
  git commit -m "CFB Data log update (Start: $i End: $i)" >/dev/null || true
  git push >/dev/null
  rm -f "$TMPLOG"
  if [ "${RSCRIPT_RC:-0}" != "0" ]; then
    echo "::error ::A creation step for season $i exited with code $RSCRIPT_RC"
    ANY_FAILED=1
  fi
done

Rscript R/run_summary.R -s "$START_YEAR" -e "$END_YEAR" || true
[ "${ANY_FAILED:-0}" = "0" ] || exit 1
