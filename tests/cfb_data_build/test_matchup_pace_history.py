"""``pace_history`` vs the pipeline's ``pace_hist`` table for 2025.

Oracle: ``fixtures/matchup/pace_hist_2025.csv`` -- the season-2025 slice of the
file ``tools/backfill_pace.R`` wrote when the 2025 season was promoted into
the static master (see the fixtures README). One row per (season, team,
game_id) for every team that appears as offense or defense, FCS included.

Parity bar: 1e-6 on the four pace columns (R long-double running sums vs a
polars cumulative sum), null placement identical. Needs the full season as
input, so the parity check is ``integration``-marked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cfb_data_build.matchup_features import dedupe_plays, pace_history

FIX = Path(__file__).parent / "fixtures" / "matchup"
CACHE = Path(__file__).parents[2] / "python" / ".cache" / "matchup"
PACE_COLS = (
    "off_sec_per_play_mean",
    "off_sec_per_play_median",
    "def_sec_per_play_mean",
    "def_sec_per_play_median",
)


def test_first_game_is_null_and_rows_are_unique() -> None:
    pbp = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    out = pace_history(pbp)
    assert out.columns == ["season", "team", "game_id", *PACE_COLS]
    assert out.unique(subset=["season", "team", "game_id"]).height == out.height
    # a team's earliest game has no prior plays on either side
    first = (
        out.join(
            pbp.select("game_id", date=pl.col("start_date").str.slice(0, 10)).unique(),
            on="game_id",
        )
        .sort(["team", "date"])
        .group_by("team", maintain_order=True)
        .first()
    )
    assert (
        first.select(PACE_COLS).null_count().sum_horizontal().item() == 4 * first.height
    )


@pytest.mark.integration
def test_pace_history_matches_r() -> None:
    pbp = dedupe_plays(pl.read_parquet(CACHE / "cfbfastR_cfb_pbp_2025.parquet"))
    py = pace_history(pbp)
    oracle = pl.read_csv(
        FIX / "pace_hist_2025.csv", infer_schema_length=10000, null_values=["NA", ""]
    )
    keys = ["season", "team", "game_id"]
    py, oracle = py.sort(keys), oracle.sort(keys)
    assert py.height == oracle.height, (py.height, oracle.height)
    assert py.select(keys).equals(oracle.select(keys).cast(py.select(keys).schema))
    for c in PACE_COLS:
        a = py[c].to_numpy()
        b = oracle[c].cast(pl.Float64).to_numpy()
        assert (np.isnan(a) == np.isnan(b)).all(), f"{c}: null placement differs"
        ok = ~np.isnan(a)
        assert np.abs(a[ok] - b[ok]).max() <= 1e-6, c
