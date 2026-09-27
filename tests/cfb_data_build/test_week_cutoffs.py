"""``week_cutoffs`` bounds a ``through_week`` ratings snapshot to exactly its weeks.

``cfb_ratings(as_of_date=)`` keeps games with ``date < as_of_date``, so the bound
must be the day AFTER the week's last kickoff. It used to be the last kickoff
date itself, which dropped every game on that date: the published 2026
``through_week == 4`` held 412 of 430 team-games, missing the Saturday-night
kickoffs that fall on Sunday in UTC.

Fixture: real ``cfb_schedule_master`` rows -- 2026 weeks 3-5 (week 4's last
kickoff is 2026-09-27T03:00Z) and 2020 weeks 13-15, where the COVID reschedules
make week 14 (to Dec 7) overlap week 15 (from Dec 6).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.derived import week_cutoffs

FIX = Path(__file__).parent / "fixtures" / "week_cutoffs_schedule.parquet"


def _games(season: int) -> pl.DataFrame:
    return (
        pl.read_parquet(FIX)
        .filter(pl.col("season") == season)
        .with_columns(pl.col("start_date").str.slice(0, 10).str.to_date().alias("d"))
    )


@pytest.mark.parametrize("season", [2020, 2026])
def test_snapshot_never_sees_a_later_week(season):
    games = _games(season)
    for week, cutoff in week_cutoffs(season, FIX):
        leaked = games.filter(
            (pl.col("week") > week) & (pl.col("d") < dt.date.fromisoformat(cutoff))
        )
        assert leaked.height == 0, f"{season} through_week {week}: {leaked}"


@pytest.mark.parametrize("season", [2020, 2026])
def test_snapshot_holds_its_whole_week(season):
    games = _games(season)
    for week, cutoff in week_cutoffs(season, FIX):
        if (season, week) == (2020, 14):
            continue  # overlaps week 15; the leak test above wins
        missing = games.filter(
            (pl.col("week") <= week) & (pl.col("d") >= dt.date.fromisoformat(cutoff))
        )
        assert missing.height == 0, f"{season} through_week {week}: {missing}"


def test_known_bounds():
    assert dict(week_cutoffs(2026, FIX))[4] == "2026-09-28"
    # capped at week 15's first kickoff, not week 14's last + 1 (Dec 8)
    assert dict(week_cutoffs(2020, FIX))[14] == "2020-12-06"
