"""``week_cutoffs`` bounds a ``through_week`` ratings snapshot to exactly its weeks.

``cfb_ratings(as_of_date=)`` keeps games with ``date < as_of_date``. The bound
used to be the week's last kickoff date itself, which dropped every game on it:
the published 2026 ``through_week == 4`` held 412 of 430 team-games, missing
the Saturday-night kickoffs that fall on Sunday in UTC.

Fixtures are real rows. ``week_cutoffs_schedule`` is the raw schedule master
(the week list); ``week_cutoffs_games`` is ``load_cfb_schedule`` (the bounds),
FBS games only, plus 2020's first FBS bowl. They cover 2026 weeks 3-5 and 13-15
(week 14's championship games are not scheduled yet), 2020 weeks 13-15, 2019
weeks 15-16 (Army-Navy: master week 16, schedule week 15) and 2005 weeks 12-13
(Ohio-Miami (OH) 253250195: master week 12, schedule week 13).
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import polars as pl
import pytest

import cfb_data_build.derived as derived
from cfb_data_build.derived import week_cutoffs

FIX = Path(__file__).parent / "fixtures"
MASTER = FIX / "week_cutoffs_schedule.parquet"
GAMES = pl.read_parquet(FIX / "week_cutoffs_games.parquet")
SEASONS = [2005, 2019, 2020, 2026]


def _fit(season: int) -> pl.DataFrame:
    """The games cfb_ratings fits: FBS-vs-FBS, with a UTC date."""
    return GAMES.filter(
        (pl.col("season") == season)
        & (pl.col("home_division") == "fbs")
        & (pl.col("away_division") == "fbs")
    ).with_columns(pl.col("start_date").str.slice(0, 10).str.to_date().alias("d"))


def _cuts(season: int, games: pl.DataFrame = GAMES) -> dict[int, dt.date]:
    return {
        w: dt.date.fromisoformat(c)
        for w, c in week_cutoffs(season, MASTER, games=games)
    }


@pytest.mark.parametrize("season", SEASONS)
def test_snapshot_never_sees_a_later_game(season):
    fit = _fit(season)
    for week, bound in _cuts(season).items():
        later = fit.filter(
            ((pl.col("season_type_id") == 2) & (pl.col("week") > week))
            | (pl.col("season_type_id") == 3)
        )
        leaked = later.filter(pl.col("d") < bound)
        assert leaked.height == 0, f"{season} through_week {week}: {leaked}"


@pytest.mark.parametrize("season", SEASONS)
def test_snapshot_holds_its_whole_week(season):
    fit = _fit(season).filter(pl.col("season_type_id") == 2)
    for week, bound in _cuts(season).items():
        missing = fit.filter((pl.col("week") <= week) & (pl.col("d") >= bound))
        assert missing.height == 0, f"{season} through_week {week}: {missing}"


def test_known_bounds():
    assert _cuts(2026)[4] == dt.date(2026, 9, 28)
    # 253250195 kicks off 2005-11-22 UTC and is week 13 to every consumer
    assert _cuts(2005)[12] == dt.date(2005, 11, 21)
    # the master's cancelled Dec 6 week-15 row no longer caps week 14
    assert _cuts(2020)[14] == dt.date(2020, 12, 8)
    # master weeks with no FBS game of their own reuse the week before
    assert _cuts(2019)[16] == _cuts(2019)[15] == dt.date(2019, 12, 15)
    assert _cuts(2026)[14] == _cuts(2026)[13]
    # the master week list, regular season only
    assert list(_cuts(2026)) == [3, 4, 5, 13, 14, 15]


def test_next_week_caps_the_bound():
    """A week-15 game moved onto week 14's last date stays out of through_week 14.

    Real FBS weeks never touch (the cap is a guard), so move one: 2020's first
    week-15 game onto the last week-14 date.
    """
    fit = _fit(2020)
    last14 = fit.filter(pl.col("week") == 14)["d"].max()
    first15 = fit.filter(pl.col("week") == 15).sort("start_date")["game_id"][0]
    moved = GAMES.with_columns(
        pl.when(pl.col("game_id") == first15)
        .then(pl.lit(f"{last14}T23:00Z"))
        .otherwise(pl.col("start_date"))
        .alias("start_date")
    )
    assert _cuts(2020, moved)[14] == last14


def test_postseason_caps_the_final_bound():
    """A bowl on the last regular-season day stays out of the final snapshot.

    The real 2020 Myrtle Beach Bowl (Dec 21) moved onto the last week-15 date.
    """
    last = _fit(2020).filter(pl.col("week") == 15)["d"].max()
    moved = GAMES.with_columns(
        pl.when(pl.col("season_type_id") == 3)
        .then(pl.lit(f"{last}T23:00Z"))
        .otherwise(pl.col("start_date"))
        .alias("start_date")
    )
    assert _cuts(2020, moved)[15] == last


def test_build_ratings_weekly_fits_each_week_at_its_bound(monkeypatch, tmp_path):
    """The builder hands every bound, unchanged, to ``cfb_ratings``."""
    (tmp_path / "cfb").mkdir()
    shutil.copy(MASTER, tmp_path / "cfb" / "cfb_schedule_master.parquet")
    monkeypatch.setenv("CFB_RAW_ROOT", str(tmp_path))

    import sportsdataverse.cfb as sdv_cfb

    seen: list[dt.date] = []

    def fake_ratings(season, *, as_of_date):
        seen.append(as_of_date)
        return pl.DataFrame({"season": [season], "team_id": [1]})

    monkeypatch.setattr(sdv_cfb, "cfb_ratings", fake_ratings)
    monkeypatch.setattr(
        sdv_cfb,
        "load_cfb_schedule",
        lambda seasons: GAMES.filter(pl.col("season").is_in(seasons)),
    )

    out = derived.build_ratings_weekly(2026)

    assert seen == list(_cuts(2026).values())
    assert out["through_week"].to_list() == list(_cuts(2026))
