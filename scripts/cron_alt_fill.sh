#!/usr/bin/env bash
# Droplet step after cfbfastR-cfb-alt-data's scripts/parse_alt.sh parses a CBS / Fox / Yahoo
# season: rebuild that season's ESPN <-> CBS / Fox / Yahoo player crosswalk
# (python/build_cfb_players_crosswalk.py, offline), publish to the cfb_crosswalk tag ONLY the
# parquets whose sha256 differs from the release's, then commit the changed files.
#
#   bash scripts/cron_alt_fill.sh -s YYYY [-e YYYY] [-n]   # -n = build into a temp tree; no commit, no publish
#
# A season without any parsed alt season is skipped; a range without one exits 0. A rebuild that
# would drop provider ids from the committed file is refused by the builder (exit 1).
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== cron_alt_fill $(date -u +%FT%TZ)"
LG=cfb PREFIX=CFB START="" END="" DRY=0
ALT=${CFB_ALT_DATA_ROOT:-../cfbfastR-cfb-alt-data}
XW=$LG/crosswalk/parquet
while getopts s:e:n flag; do
  case "$flag" in s) START=$OPTARG;; e) END=$OPTARG;; n) DRY=1;; *) echo "usage: $0 -s YYYY [-e YYYY] [-n]" >&2; exit 2;; esac
done
[[ "$START" =~ ^[0-9]{4}$ ]] || { echo "usage: $0 -s YYYY [-e YYYY] [-n]" >&2; exit 2; }
END=${END:-$START}
# A failed build or publish must not leave tracked changes: cron_daily_cfb.sh refuses a dirty tree.
# The release side heals itself: the next run re-uploads whatever still differs from it.
restore() { [ "$DRY" = 1 ] || { git checkout -q -- "$XW" 8>&- 9>&-; git clean -qf -- "$XW/${LG}_players_crosswalk_*.parquet" 8>&- 9>&-; }; }
fail() { echo "::error ::$*"; restore; echo "EXIT=1"; exit 1; }

seasons=()
for season in $(seq "$START" "$END"); do
  if compgen -G "$ALT/*/game_rosters/parquet/${LG}_*_game_rosters_$season.parquet" >/dev/null; then
    seasons+=("$season")
  else
    echo "$season: no parsed CBS / Fox / Yahoo season under $ALT; skipped"
  fi
done
[ ${#seasons[@]} -gt 0 ] || { echo "no parsed alt season in $START-$END"; echo "EXIT=0"; exit 0; }

# One writer in this checkout: cron_daily_cfb.sh's build lock. And the git_pull sweep's lock (it
# would autostash our outputs). Children get 8>&- 9>&- so a daemonized git gc / credential cache
# cannot keep either lock after this exits.
exec 8>/tmp/cfbfastR-cfb-data-build.lock
flock -w 10800 8 || { echo "::error ::cfb build held its lock for 3h"; echo "EXIT=1"; exit 1; }
exec 9>/tmp/git_pull_sdv.lock
flock -w 3600 9 || { echo "::error ::git_pull sweep held its lock for 1h"; echo "EXIT=1"; exit 1; }
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != main ] || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git status --short --untracked-files=no | head -20
  echo "::error ::checkout is on '$branch' or has tracked changes; refusing to fill"; echo "EXIT=1"; exit 1
fi
[ "$DRY" = 1 ] || git pull -q --ff-only 8>&- 9>&- || fail "git pull --ff-only failed"

source scripts/_venv.sh || fail "no interpreter"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
out=$XW upload=--upload
if [ "$DRY" = 1 ]; then
  out=$(mktemp -d)
  # Seed the committed files so the builder's never-drop-a-leg check runs against them.
  cp "$XW"/${LG}_players_crosswalk_*.parquet "$out"/ 2>/dev/null
  upload=
fi
for season in "${seasons[@]}"; do
  "$PY" python/build_cfb_players_crosswalk.py -s "$season" -e "$season" --alt-root "$ALT" --out "$out" $upload 8>&- 9>&- \
    || fail "players crosswalk $season build/publish failed"
done

if [ "$DRY" = 1 ]; then
  for season in "${seasons[@]}"; do
    f=${LG}_players_crosswalk_$season.parquet
    if [ ! -f "$out/$f" ]; then echo "$f: not built"
    elif [ ! -f "$XW/$f" ]; then echo "$f: NEW"
    elif cmp -s "$out/$f" "$XW/$f"; then echo "$f: unchanged"
    else echo "$f: CHANGED"; fi
  done
  echo "dry run: built into $out, no commit, no publish"; echo "EXIT=0"; exit 0
fi

files=()
for season in "${seasons[@]}"; do
  f=$XW/${LG}_players_crosswalk_$season.parquet
  [ -f "$f" ] && files+=("$f")   # one missing pathspec makes git add stage nothing
done
[ ${#files[@]} -gt 0 ] && git add -- "${files[@]}" 8>&- 9>&-
git diff --cached --quiet && { echo "players crosswalks unchanged"; echo "EXIT=0"; exit 0; }
git diff --cached --stat
git commit -q -m "$PREFIX Data Updated (Start: $START End: $END)" 8>&- 9>&- || fail "commit failed"
rc=1
for attempt in 1 2 3; do   # cron_daily_cfb.sh may push meanwhile; rebase --merge (parquet blobs)
  if git push -q origin HEAD 8>&- 9>&-; then rc=0; echo "pushed $PREFIX $START-$END players crosswalk"; break; fi
  git fetch -q origin main 8>&- 9>&- && git rebase -q --merge origin/main 8>&- 9>&- \
    || { git rebase --abort 8>&- 9>&- 2>/dev/null; echo "::error ::cannot rebase onto origin/main"; break; }
  [ "$attempt" = 3 ] && echo "::error ::push rejected 3 times"
done
echo "EXIT=$rc"
exit "$rc"
