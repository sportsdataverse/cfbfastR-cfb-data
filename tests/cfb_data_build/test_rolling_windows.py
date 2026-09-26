"""rolling_windows is a derived stage over the committed pbp tree (real 2023-2024 rows)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from sportsdataverse.rolling_windows import FOOTBALL_PBP_COLUMNS

from cfb_data_build.cli import DERIVED
from cfb_data_build.derived import BUILDERS, SPECS, build_rolling_windows

REPO = Path(__file__).resolve().parents[2]
QB = 4433971


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    b = tmp_path_factory.mktemp("cfb")
    for s in (2023, 2024):
        pbp = pl.read_parquet(
            REPO / "cfb" / "pbp" / "parquet" / f"play_by_play_{s}.parquet",
            columns=list(FOOTBALL_PBP_COLUMNS),
        )
        games = (
            pbp.filter(pl.col("passer_player_id") == QB)["game_id"].unique().implode()
        )
        (b / "pbp" / "parquet").mkdir(parents=True, exist_ok=True)
        pbp.filter(pl.col("game_id").is_in(games)).write_parquet(
            b / "pbp" / "parquet" / f"play_by_play_{s}.parquet"
        )
        (b / "cfb_schedules" / "parquet").mkdir(parents=True, exist_ok=True)
        pl.read_parquet(
            REPO / "cfb" / "cfb_schedules" / "parquet" / f"cfb_schedules_{s}.parquet"
        ).write_parquet(b / "cfb_schedules" / "parquet" / f"cfb_schedules_{s}.parquet")
    return b


def test_reads_every_prior_season_of_the_built_tree(base):
    df = build_rolling_windows(2024, base=str(base))
    r = df.filter(
        (pl.col("entity_id") == str(QB))
        & (pl.col("window_unit") == "dropback")
        & (pl.col("metric") == "epa")
        & (pl.col("window_n") == 100)
    ).row(0, named=True)
    # cur/prev/season_start only need 2023-2024, so they match the sdv-py oracle exactly
    assert r["cur"] == pytest.approx(0.625317, abs=1e-6)
    assert r["prev"] == pytest.approx(0.457801, abs=1e-6)
    assert r["season_start"] == pytest.approx(0.464828, abs=1e-6)
    assert df["season"].unique().to_list() == [2024]


def test_registered_as_a_derived_dataset_with_its_tag():
    assert "rolling_windows" in DERIVED and "rolling_windows" in BUILDERS
    assert SPECS["rolling_windows"].tag == "cfb_rolling_windows"
    assert SPECS["rolling_windows"].stem == "rolling_windows"
