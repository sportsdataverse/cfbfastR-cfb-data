#!/bin/bash
# Capture this season's weekly ESPN FPI snapshot (`cfb_fpi_weekly`, stage 13).
#
# Run by .github/workflows/cfb_fpi_weekly.yml on its own in-season schedule,
# deliberately NOT from scripts/daily_cfb_processor.sh. Every other dataset in
# this repo is rebuildable from the raw store whenever the daily job recovers;
# this one is not. ESPN overwrites the week-1 slot with a late-season
# computation (2024 wk1 is stamped 2024-12-15, later than that season's wk16),
# so an as-of-week-N rating only ever exists in a capture taken during week N.
# Coupling an unrecoverable capture to the 25-dataset daily job means one of its
# failures costs a week permanently.
#
# Idempotent by construction: the builder preserves weeks already on disk and
# appends only new ones, so a run that finds no new week writes nothing, commits
# nothing, publishes nothing, and exits 0.
set -uo pipefail

# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh" || exit 1

PUBLISH=""
while getopts s:e:n flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    n) PUBLISH="--dry-run";;   # build + report only
    *) echo "usage: $0 -s <start> [-e <end>] [-n]" >&2; exit 2;;
  esac
done
: "${START_YEAR:?-s <season> is required}"
END_YEAR=${END_YEAR:-$START_YEAR}
[ -n "$PUBLISH" ] || PUBLISH="--publish"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/python" || exit 1

# `git status --porcelain` on the parquet is the ONLY signal that the build
# added a week. The builder's own "no new snapshot" line is informational; the
# tree is the fact. A build failure leaves the tree clean too, so the rc is
# checked first and separately -- otherwise a crashed fetch would read as
# "nothing new" and the job would go green having captured nothing.
"$PY" espn_cfb_13_fpi_weekly_creation.py -s "$START_YEAR" -e "$END_YEAR" --base ../cfb $PUBLISH
BUILD_RC=$?
if [ "$BUILD_RC" != "0" ]; then
  echo "::error ::fpi_weekly build exited with code $BUILD_RC"
  exit "$BUILD_RC"
fi

cd "$REPO_ROOT" || exit 1
# Only when unset. The sibling drivers set this unconditionally because they
# only ever run in CI; this one is also a documented manual entry point, and a
# `git config --local` from a worktree writes the SHARED .git/config -- i.e. it
# would overwrite the human's identity in the primary checkout too.
git config user.email >/dev/null 2>&1 || git config --local user.email "action@github.com"
git config user.name  >/dev/null 2>&1 || git config --local user.name "Github Action"
# Explicit path, and `add` rather than `add -u`: a season's FIRST capture is an
# untracked new file, every later one is a modification, and this catches both.
git add -- cfb/fpi_weekly || exit 1
if git diff --cached --quiet; then
  echo "no new FPI week for ${START_YEAR}-${END_YEAR}; nothing to commit"
  exit 0
fi
git commit -m "CFB FPI weekly snapshot (Start: $START_YEAR End: $END_YEAR)" || {
  echo "::error ::commit failed"; exit 1;
}

# Stage-then-reconcile, matching daily_cfb_processor.sh: pulling before staging
# can only abort ("local changes would be overwritten"), and swallowing that is
# how a green job publishes nothing. `rebase --merge` because git's default am
# backend base64-encodes every parquet blob it replays.
for attempt in 1 2 3; do
  if git push origin HEAD; then
    echo "pushed (attempt $attempt)"
    exit 0
  fi
  echo "push rejected (attempt $attempt); syncing with origin"
  git fetch --quiet origin main || true
  if ! git rebase --merge origin/main; then
    git rebase --abort >/dev/null 2>&1 || true
    echo "::error ::cannot rebase onto origin/main"
    exit 1
  fi
done
echo "::error ::push still rejected after 3 attempts; the repo mirror is stale"
exit 1
