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
  R/espn_cfb_07_rosters_creation.R
  R/espn_cfb_08_betting_creation.R
  R/espn_cfb_09_schedules_creation.R
  R/espn_cfb_10_linescores_creation.R
  R/espn_cfb_11_power_index_creation.R
  R/espn_cfb_13_injuries_creation.R
)

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
    for SCRIPT in "${SCRIPTS[@]}"; do
      Rscript "$SCRIPT" -s "$i" -e "$i" || {
        rc=$?; echo "::warning ::$SCRIPT for season $i exited with code $rc"; SEASON_RC=$rc
      }
    done
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
    echo "::error ::A creation script for season $i exited with code $RSCRIPT_RC"
    ANY_FAILED=1
  fi
done

Rscript R/run_summary.R -s "$START_YEAR" -e "$END_YEAR" || true
[ "${ANY_FAILED:-0}" = "0" ] || exit 1
