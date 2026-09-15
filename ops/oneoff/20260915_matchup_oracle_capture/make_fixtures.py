"""Cut the committed matchup-feature fixtures from a full-season oracle capture.

Inputs (from ``capture_oracle.R`` and the release parquet):
  <cache>/cfbfastR_cfb_pbp_<season>.parquet   the sdv-py ``load_cfb_pbp_r`` frame
  <cache>/play_flags_wepa_<season>.csv.gz      R per-play oracle (full season)
  <cache>/drives_<season>.csv.gz               R per-drive oracle (full season)

Outputs under tests/cfb_data_build/fixtures/matchup/: every 30th game id
(sorted) -> the sample input parquet (48 columns), the per-play oracle for
those games, and their drives. Row counts are printed for the README.

    uv run python ops/oneoff/20260915_matchup_oracle_capture/make_fixtures.py --season 2025
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "cfb_data_build" / "fixtures" / "matchup"
CACHE = REPO / "python" / ".cache" / "matchup"

INPUT_COLS = [
    "game_id", "id_play", "game_play_number", "season", "week", "start_date", "season_type",
    "home", "away", "pos_team", "def_pos_team", "offense_play", "defense_play", "team",
    "EPA", "ppa", "success", "rush", "pass", "play_type", "penalty_detail", "sack", "fumble_vec",
    "position_rush", "yards_to_goal", "Goal_To_Go", "wp_before", "wp_after", "score_diff", "period",
    "down", "distance", "drive_id", "new_drive_pts", "drive_start_period", "TimeSecsRem",
    "adj_TimeSecsRem", "drive_end_yards_to_goal", "pos_team_timeouts", "def_pos_team_timeouts",
    "drive_result", "clock_minutes", "clock_seconds", "pos_team_score", "def_pos_team_score",
    "half", "ep_before", "rz_play",
]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--every", type=int, default=30, help="keep every Nth game id")
    args = ap.parse_args()
    s = args.season

    pbp = pl.read_parquet(CACHE / f"cfbfastR_cfb_pbp_{s}.parquet")
    plays = pl.read_csv(
        CACHE / f"play_flags_wepa_{s}.csv.gz", infer_schema_length=10000
    )
    drives = pl.read_csv(CACHE / f"drives_{s}.csv.gz", infer_schema_length=10000)

    keep = sorted(plays["game_id"].unique().to_list())[:: args.every]
    (FIX / "sample_game_ids.json").write_text(json.dumps(keep), encoding="utf-8")

    sub = pbp.filter(pl.col("game_id").is_in(keep)).select(INPUT_COLS)
    sub.write_parquet(FIX / f"pbp_input_{s}_sample.parquet")

    ps = plays.filter(pl.col("game_id").is_in(keep))
    with gzip.open(FIX / f"play_flags_wepa_{s}_sample.csv.gz", "wb") as fh:
        fh.write(ps.write_csv().encode())

    prefix = pl.col("drive_id").cast(pl.Int64).cast(pl.Utf8).str.slice(0, 9)
    ds = drives.filter(prefix.is_in([str(g) for g in keep]))
    ds.write_csv(FIX / f"drives_{s}_sample.csv")

    print(f"games={len(keep)} input={sub.shape} plays={ps.shape} drives={ds.shape}")


if __name__ == "__main__":
    main()
