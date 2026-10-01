"""metric_curves (stage 65) is a derived stage over ONE season of the committed pbp tree.

The fixture is the real 2024 release pbp copied into a tmp base; every oracle
below is recomputed from that same file, never pinned.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
from sportsdataverse.metric_curves import OUTPUT_SCHEMA

from cfb_data_build.cli import DERIVED
from cfb_data_build.config import PKG_FUNCTION
from cfb_data_build.derived import (
    AIR_YARDS_FLOOR,
    BUILDERS,
    SPECS,
    build_metric_curves,
)
from cfb_data_build.publish import RELEASE_NOTES

REPO = Path(__file__).resolve().parents[2]
SEASON = 2024
PBP = REPO / "cfb" / "pbp" / "parquet" / f"play_by_play_{SEASON}.parquet"


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    b = tmp_path_factory.mktemp("cfb")
    (b / "pbp" / "parquet").mkdir(parents=True)
    shutil.copyfile(PBP, b / "pbp" / "parquet" / PBP.name)
    return b


@pytest.fixture(scope="module")
def curves(base) -> pl.DataFrame:
    return build_metric_curves(SEASON, base=str(base))


@pytest.fixture(scope="module")
def pbp(base) -> pl.DataFrame:
    # regular season + postseason: the population the sdv-py adapter counts
    return pl.read_parquet(
        base / "pbp" / "parquet" / PBP.name,
        columns=[
            "seasonType",
            "fg_attempt",
            "fg_made",
            "yds_fg",
            "pass_attempt",
            "air_yards",
        ],
    ).filter(pl.col("seasonType").is_in([2, 3]))


def test_league_fg_curve_sums_to_the_seasons_fg_attempts_and_makes(pbp, curves):
    fg = pbp.filter((pl.col("fg_attempt") == True) & pl.col("yds_fg").is_not_null())  # noqa: E712
    league = curves.filter(
        (pl.col("metric") == "fg_pct_by_distance") & (pl.col("entity_type") == "league")
    )
    assert league["attempts"].sum() == fg.height > 0
    assert league["successes"].sum() == fg.filter(pl.col("fg_made") == True).height  # noqa: E712
    assert (league["rate"] == league["successes"] / league["attempts"]).all()


def test_success_by_down_distance_has_exactly_20_league_rows(curves):
    league = curves.filter(
        (pl.col("metric") == "success_by_down_distance")
        & (pl.col("entity_type") == "league")
    )
    assert league.height == 20
    assert sorted(league["down"].unique()) == [1, 2, 3, 4]
    assert league.group_by("down").len()["len"].to_list() == [5] * 4


def test_contract_dtypes_ids_entities_and_the_air_yards_span(pbp, curves):
    assert curves.schema == OUTPUT_SCHEMA
    assert set(curves["id_source"]) == {"espn"}
    assert curves["season"].unique().to_list() == [SEASON]
    assert set(curves["entity_type"]) == {"league", "team", "player"}
    assert (
        curves.filter(pl.col("entity_type") == "league")["entity_id"].null_count()
        == curves.filter(pl.col("entity_type") == "league").height
    )
    assert (
        curves.filter(pl.col("entity_type") != "league")["entity_id"].null_count() == 0
    )
    # ids are text, never a stringified float
    assert not curves["entity_id"].drop_nulls().str.contains(r"\.").any()
    # 2024 carries a stray air-yards pass attempt (1 of ~57k): the function would
    # emit a one-attempt curve for it, so the producer enforces the 2025+ span.
    assert SEASON < AIR_YARDS_FLOOR
    assert (
        pbp.filter(
            (pl.col("pass_attempt") == True) & pl.col("air_yards").is_not_null()
        ).height
        > 0
    )  # noqa: E712
    assert not curves["metric"].str.contains("air_yards").any()
    assert set(curves["metric"]) == {
        "fg_pct_by_distance",
        "fourth_conv_by_ytg",
        "success_by_down_distance",
    }


def test_missing_season_is_the_empty_contract_frame(base):
    df = build_metric_curves(1999, base=str(base))
    assert df.height == 0
    assert df.schema == OUTPUT_SCHEMA


def test_registered_as_a_derived_dataset_with_its_tag():
    assert "metric_curves" in DERIVED and "metric_curves" in BUILDERS
    assert SPECS["metric_curves"].tag == "cfb_metric_curves"
    assert SPECS["metric_curves"].stem == "metric_curves"
    assert PKG_FUNCTION["cfb_metric_curves"].endswith(
        "espn_cfb_65_metric_curves_creation.py"
    )
    assert "2025" in RELEASE_NOTES["cfb_metric_curves"]
