"""Stage 68 (cfb_team_adjustment_ledger): the producer layer over sdv-py's ledger.

Real rows only: tests/fixtures/adjustment_ledger/README.md (the 2025 SEC-vs-SEC games).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from sportsdataverse.cfb import cfb_adjusted_epa

from cfb_data_build import adjustment_ledger as stage

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "adjustment_ledger"


@pytest.fixture(scope="module")
def plays() -> pl.DataFrame:
    return pl.read_parquet(FIX / "sec_2025_plays.parquet")


@pytest.fixture(scope="module")
def table(plays) -> pl.DataFrame:
    return stage.ledger_table(plays, 2025)


def test_schema_and_int64_ids(table) -> None:
    assert dict(table.schema) == stage.OUTPUT_SCHEMA
    assert table.height == 2 * 67  # one row per team per game
    assert table["season"].unique().to_list() == [2025]
    assert table["season_type"].null_count() == 0


def test_contributions_sum_to_the_season_fit(plays, table) -> None:
    season = cfb_adjusted_epa(plays).select(
        pl.col("team_id").cast(pl.Int64), "net_adj_epa"
    )
    sums = table.group_by("team_id").agg(total=pl.col("contribution").sum())
    joined = season.join(sums, on="team_id", how="full", coalesce=True)
    assert joined.height == 16 and joined.null_count().sum_horizontal().item() == 0
    assert (joined["total"] - joined["net_adj_epa"]).abs().max() < 1e-12
    assert (
        table["game_influence"].null_count() == 0
    )  # 8 games each: every refit has a value


def test_a_ledger_that_misses_its_season_value_fails_the_season(
    plays, monkeypatch
) -> None:
    real = stage.cfb_adjusted_epa_ledger

    def off_by_a_hair(frame):
        out = real(frame)
        return out.with_columns(
            pl.col("contribution")
            + pl.when(pl.int_range(pl.len()) == 0).then(1e-6).otherwise(0.0)
        )

    monkeypatch.setattr(stage, "cfb_adjusted_epa_ledger", off_by_a_hair)
    with pytest.raises(ValueError, match="contributions miss net_adj_epa"):
        stage.ledger_table(plays, 2025)


def test_wired_as_a_derived_dataset_on_its_tag() -> None:
    from cfb_data_build.cli import DERIVED
    from cfb_data_build.derived import BUILDERS, SPECS

    assert "adjustment_ledger" in DERIVED
    assert SPECS["adjustment_ledger"].tag == "cfb_team_adjustment_ledger"
    assert SPECS["adjustment_ledger"].stem == "cfb_team_adjustment_ledger"
    assert callable(BUILDERS["adjustment_ledger"])


def test_a_season_with_no_plays_is_the_empty_schema(monkeypatch) -> None:
    import sportsdataverse.cfb as cfb

    monkeypatch.setattr(cfb, "load_cfb_pbp", lambda seasons: pl.DataFrame())
    out = stage.build_adjustment_ledger(2027)
    assert out.height == 0 and dict(out.schema) == stage.OUTPUT_SCHEMA


def test_a_ledger_missing_a_column_fails_the_season(plays, monkeypatch) -> None:
    real = stage.cfb_adjusted_epa_ledger
    monkeypatch.setattr(
        stage,
        "cfb_adjusted_epa_ledger",
        lambda frame: real(frame).drop("game_influence"),
    )
    with pytest.raises(ValueError, match="lacks"):
        stage.ledger_table(plays, 2025)


def test_a_season_where_no_team_has_two_games_is_the_empty_schema(plays) -> None:
    # one game: both teams have a single valid game, so the fit rates nobody yet (an opening week)
    one_game = plays.filter(pl.col("game_id") == plays["game_id"].min())
    out = stage.ledger_table(one_game, 2025)
    assert out.height == 0 and dict(out.schema) == stage.OUTPUT_SCHEMA


def test_the_daily_driver_builds_the_ledger_right_after_summaries() -> None:
    # it must read the same pbp + schedule releases as summaries: after PY_UNIFIED_SCHEDULES and the summaries publish
    import re

    sh = (
        Path(__file__).resolve().parents[2] / "scripts" / "daily_cfb_processor.sh"
    ).read_text(encoding="utf-8")
    assert re.search(r'(?m)^PY_LEDGER="adjustment_ledger"$', sh)
    line = lambda pattern: re.search(pattern, sh, re.M).start()  # noqa: E731  (anchored: a commented-out line fails)
    schedules = line(r"^    for ds in \$PY_UNIFIED_SCHEDULES; do")
    summaries = line(r"^    run_py summaries --publish$")
    ledger = line(
        r'^    for ds in \$PY_LEDGER; do run_py "\$ds" --no-fetch --publish; done$'
    )
    assert schedules < summaries < ledger
    assert "adjustment_ledger" not in re.search(r'(?m)^PY_DERIVED="([^"]*)"', sh).group(
        1
    )
