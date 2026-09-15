"""Builder: per-team as-of matchup features (``cfb_matchup_features``).

Thin entrypoint. The build lives in ``cfb_data_build.matchup_build``; this file
exists so the directory listing is the pipeline and each dataset is runnable on
its own. 41 opens the 41-49 block after the ESPN reshapes (01-40); 42 is its
consumer, the matchup line.

One row per FBS team-game: EPA / success / WEPA / scoring-opportunity /
rush-rate-over-expected / pace, over the team's plays dated strictly BEFORE
the game (a team's first game is null -- nothing prior exists). Reads the
``cfbfastR_cfb_pbp`` release and CFBD ``/games`` (needs ``CFBD_API_KEY``).

Example:
    One season::

        uv run python python/espn_cfb_41_matchup_features_creation.py -s 2025 -e 2025 --base ../cfb
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "matchup_features"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
