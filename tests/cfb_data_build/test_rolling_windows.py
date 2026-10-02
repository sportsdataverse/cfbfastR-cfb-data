"""rolling_windows is a derived stage over the committed pbp tree (real 2023-2024 rows)."""

from __future__ import annotations

from datetime import date
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
    # (re-pin these three if the 2023-2024 espn_cfb_pbp release is ever republished;
    # last re-pinned 2026-10-02 after the rev-13 reprocess rebuilt 2023 and 2024)
    assert r["cur"] == pytest.approx(0.609368, abs=1e-6)
    assert r["prev"] == pytest.approx(0.430229, abs=1e-6)
    assert r["season_start"] == pytest.approx(0.447179, abs=1e-6)
    assert df["season"].unique().to_list() == [2024]
    # QB's last 2024 game (401729867, Syracuse @ Washington State) kicks off at
    # 2024-12-28T01:00:00Z -- 2024-12-27 8pm ET. A bare UTC date slice would
    # land this on the 28th; the UTC -> America/New_York conversion keeps it on
    # the 27th, the calendar day the game was actually played on.
    assert r["last_event_date"] == date(2024, 12, 27)
    assert r["as_of_date"] == date(2024, 12, 27)


def test_registered_as_a_derived_dataset_with_its_tag():
    assert "rolling_windows" in DERIVED and "rolling_windows" in BUILDERS
    assert SPECS["rolling_windows"].tag == "cfb_rolling_windows"
    assert SPECS["rolling_windows"].stem == "rolling_windows"
