#!/usr/bin/env bash
# Droplet cron entry for the daily CFB data build -- its ONE automatic producer
# since 2026-09-29. The crontab chains it after every cfbfastR-cfb-raw scrape,
# so it always reads a finished capture. daily_cfb.yml keeps a manual trigger
# as the fallback; it no longer runs on a schedule, and cfb-raw's push no
# longer dispatches it.
#
#   bash scripts/cron_daily_cfb.sh [-s YYYY -e YYYY]   # default: current season
#
# Guards, each from a real failure:
# - the git_pull sweep's lock: that sweep runs `pull --rebase --autostash` over
#   every repo every 4h; mid-build it stashes the outputs out from under the
#   job (baseballr-data, 2026-09). Holding its lock makes it skip a round.
# - main + no tracked changes: this runs in the shared checkout, and a branch
#   or WIP left there would be committed and pushed as data.
# - uv sync: the driver runs .venv's python directly (_venv.sh), so a merged
#   lock bump never reached it otherwise.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== cron_daily_cfb $(date -u +%FT%TZ)"

# Saturday chains fire ~4h apart; a slow build queues the next instead of racing it.
exec 8>/tmp/cfbfastR-cfb-data-build.lock
flock -w 10800 8 || { echo "::error ::previous build still running after 3h"; echo "EXIT=1"; exit 1; }
exec 9>/tmp/git_pull_sdv.lock
flock -w 3600 9 || { echo "::error ::git_pull sweep held its lock for 1h"; echo "EXIT=1"; exit 1; }
# This shell holds both locks; git children get 8>&- 9>&-. git daemonizes `gc --auto` and
# credential-cache--daemon, and an inherited fd keeps a lock held after this build exits.

# The processor appends coach_careers output to the season log AFTER committing
# it; an Actions runner threw that tail away, here it would persist and trip
# the guard below. Drop stale edits to TRACKED logs only (the lines are in
# this cron log); any other tracked change still refuses the build.
git checkout -q -- logs/ 2>/dev/null || true
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != main ] || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "::error ::checkout is on '$branch' or has tracked changes; refusing to build"
  git status --short --untracked-files=no | head -20
  echo "EXIT=1"; exit 1
fi
git pull -q --ff-only 8>&- 9>&- || { echo "::error ::git pull --ff-only failed"; echo "EXIT=1"; exit 1; }
uv sync -q --frozen --inexact || { echo "::error ::uv sync failed"; echo "EXIT=1"; exit 1; }

# Same season rule as daily_cfb.yml: the CFB season rolls over on Aug 15.
if [ $# -eq 0 ]; then
  Y=$(date -u +%Y); M=$((10#$(date -u +%m))); D=$((10#$(date -u +%d)))
  if [ "$M" -gt 8 ] || { [ "$M" -eq 8 ] && [ "$D" -ge 15 ]; }; then YR=$Y; else YR=$((Y - 1)); fi
  set -- -s "$YR" -e "$YR"
fi

# The raw scrape just wrote this season's finals locally; read them there
# instead of re-fetching ~1,000 files from GitHub.
export CFB_FINAL_CACHE=/mnt/sdv_repos/cfbfastR-cfb-raw/cfb/json/final
# Python never sees ~/.Renviron (only R does); read the key at call time.
CFBD_API_KEY=$(sed -n 's/^CFBD_API_KEY *= *//p' ~/.Renviron | tr -d "\"'" | head -1)
export CFBD_API_KEY

bash scripts/daily_cfb_processor.sh "$@" 8>&- 9>&-
rc=$?
git checkout -q -- logs/ 2>/dev/null || true
echo "EXIT=$rc $(date -u +%FT%TZ)"
exit "$rc"
