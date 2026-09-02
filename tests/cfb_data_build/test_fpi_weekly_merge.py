"""The in-season FPI capture must never overwrite an earlier week's snapshot.

Real data, not a synthetic frame: the fixture is this repo's committed
``cfb/fpi_weekly/parquet/cfb_fpi_weekly_2026.parquet`` -- the week-1 capture
taken 2026-08-31, which is the only contemporaneous week-1 snapshot that will
ever exist for 2026. ESPN overwrites the week-1 slot with a late-season
computation (2024's is stamped 2024-12-15, later than that season's week 16), so
the destructive case below is what December's cron run actually returns.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.fpi import merge_preserving_earliest

SEASON = 2026
PARQUET = Path(__file__).resolve().parents[2] / "cfb" / "fpi_weekly" / "parquet" / f"cfb_fpi_weekly_{SEASON}.parquet"


@pytest.fixture(scope="module")
def captured() -> pl.DataFrame:
    if not PARQUET.exists():  # pragma: no cover - guards a partial checkout
        pytest.skip(f"no committed capture at {PARQUET}")
    return pl.read_parquet(PARQUET)


def test_refetch_of_an_already_captured_week_is_a_no_op(captured):
    """The cron runs more often than ESPN posts a week; that must cost nothing."""
    merged, added = merge_preserving_earliest(captured, captured)
    assert added == 0
    assert merged.height == captured.height


def test_a_late_reissue_of_week_1_cannot_overwrite_the_contemporaneous_capture(captured):
    """The whole point: December's fetch carries a rewritten week 1. Ours wins."""
    late_week1 = captured.with_columns(
        last_updated=pl.lit(f"{SEASON}-12-15T04:00Z"),
        run_date_time_key=pl.lit(int(f"{SEASON}1215040000"), dtype=pl.Int64),
    )
    new_week2 = captured.with_columns(
        week=pl.lit(2, dtype=captured.schema["week"]),
        last_updated=pl.lit(f"{SEASON}-09-09T04:00Z"),
        run_date_time_key=pl.lit(int(f"{SEASON}0909040000"), dtype=pl.Int64),
    )
    fetch = pl.concat([late_week1, new_week2], how="diagonal_relaxed")

    merged, added = merge_preserving_earliest(fetch, captured)

    assert added == new_week2.height, "the genuinely new week must be appended"
    kept = merged.filter(pl.col("week") == 1)["last_updated"].unique().to_list()
    assert kept == captured["last_updated"].unique().to_list(), (
        "week 1 was overwritten by the late reissue -- the as-of-week-1 rating is unrecoverable once that happens"
    )
    assert sorted(merged["week"].unique().to_list()) == [1, 2]


def test_cold_start_writes_the_whole_fetch(captured):
    merged, added = merge_preserving_earliest(captured, pl.DataFrame())
    assert added == captured.height
    assert merged.height == captured.height
