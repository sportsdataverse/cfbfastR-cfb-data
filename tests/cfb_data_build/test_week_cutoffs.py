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


def test_cancelled_game_does_not_cap_the_bound():
    """A cancelled row keeps its original date; it must not end a week early.

    2020's first week-15 game, marked cancelled and moved onto week 14's last
    date, leaves week 14's bound alone.
    """
    fit = _fit(2020)
    last14 = fit.filter(pl.col("week") == 14)["d"].max()
    first15 = fit.filter(pl.col("week") == 15).sort("start_date")["game_id"][0]
    moved = GAMES.with_columns(
        pl.when(pl.col("game_id") == first15)
        .then(pl.lit(f"{last14}T23:00Z"))
        .otherwise(pl.col("start_date"))
        .alias("start_date"),
        pl.when(pl.col("game_id") == first15)
        .then(pl.lit("STATUS_CANCELED"))
        .otherwise(pl.col("status"))
        .alias("status"),
    )
    assert _cuts(2020, moved)[14] == _cuts(2020)[14]


def test_no_schedule_bounds_nothing():
    """Preseason (no cfb_schedules asset yet): every week unbounded, none rated."""
    assert all(c is None for _, c in week_cutoffs(2026, MASTER, games=pl.DataFrame()))


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


#: The 2026-09-28 review: week 4 is over, week 5 has not kicked off.
TODAY = dt.date(2026, 9, 28)
#: A "today" after every 2026 bound, i.e. the season as it will look once played.
AFTER_SEASON = dt.date(2027, 2, 1)


def _raw_root(monkeypatch, tmp_path, games: pl.DataFrame):
    """Point the builders at the fixture master and ``games`` as the schedule."""
    (tmp_path / "cfb").mkdir()
    shutil.copy(MASTER, tmp_path / "cfb" / "cfb_schedule_master.parquet")
    monkeypatch.setenv("CFB_RAW_ROOT", str(tmp_path))

    import sportsdataverse.cfb as sdv_cfb

    monkeypatch.setattr(
        sdv_cfb,
        "load_cfb_schedule",
        lambda seasons: games.filter(pl.col("season").is_in(seasons)),
    )
    return sdv_cfb


def _build(
    monkeypatch,
    tmp_path,
    games: pl.DataFrame,
    *,
    season: int = 2026,
    today: dt.date | None = AFTER_SEASON,
):
    """Run ``build_ratings_weekly(season)``; return (as_of_dates seen, output)."""
    sdv_cfb = _raw_root(monkeypatch, tmp_path, games)

    seen: list = []

    def fake_ratings(season, *, as_of_date):
        seen.append(as_of_date)
        return pl.DataFrame({"season": [season], "team_id": [1]})

    monkeypatch.setattr(sdv_cfb, "cfb_ratings", fake_ratings)
    return seen, derived.build_ratings_weekly(season, today=today)


def test_build_ratings_weekly_fits_each_week_at_its_bound(monkeypatch, tmp_path):
    """The builder hands every bound, unchanged, to ``cfb_ratings``."""
    seen, out = _build(monkeypatch, tmp_path, GAMES)

    assert seen == list(_cuts(2026).values())
    assert out["through_week"].to_list() == list(_cuts(2026))


def test_week_without_a_bound_is_not_rated(monkeypatch, tmp_path):
    """No FBS game at or before week 3: no bound, and never ``as_of_date=None``.

    None would fit the WHOLE season and label it through_week 3.
    """
    games = GAMES.filter(~((pl.col("season") == 2026) & (pl.col("week") == 3)))
    assert dict(week_cutoffs(2026, MASTER, games=games))[3] is None

    seen, out = _build(monkeypatch, tmp_path, games)

    assert None not in seen and len(seen) == 5
    assert 3 not in out["through_week"].to_list()


def test_future_weeks_are_not_built(monkeypatch, tmp_path, capsys):
    """The master lists all 15 weeks before they are played. A week whose bound
    is still ahead has no new game to fit, so it refit the played ones and
    shipped a copy: 2026 through_week 5-15 were week 4 relabelled (review of
    2026-09-28). Those weeks are not gaps either, so nothing is reported MISSING.
    """
    seen, out = _build(monkeypatch, tmp_path, GAMES, today=TODAY)

    assert out["through_week"].to_list() == [3, 4]
    assert seen == [_cuts(2026)[3], _cuts(2026)[4]]
    assert "MISSING" not in capsys.readouterr().out


@pytest.mark.parametrize("days, weeks", [(-1, [3]), (0, [3, 4]), (1, [3, 4])])
def test_week_is_built_from_its_cutoff_day(monkeypatch, tmp_path, days, weeks):
    """Boundary: the bound is EXCLUSIVE (games dated before it), so on the bound's
    own date every game the snapshot holds has already kicked off.
    """
    today = _cuts(2026)[4] + dt.timedelta(days=days)

    _, out = _build(monkeypatch, tmp_path, GAMES, today=today)

    assert out["through_week"].to_list() == weeks


def test_completed_season_keeps_every_week(monkeypatch, tmp_path):
    """Every 2020 bound is in the past on the real clock: the same weeks as ever."""
    _, out = _build(monkeypatch, tmp_path, GAMES, season=2020, today=None)

    assert out["through_week"].to_list() == list(_cuts(2020)) == [13, 14, 15]


@pytest.mark.parametrize(
    "today, weeks", [(TODAY, [3, 4]), (AFTER_SEASON, [3, 4, 5, 13, 14, 15])]
)
def test_team_summaries_weekly_stops_at_the_same_week(
    monkeypatch, tmp_path, today, weeks
):
    """The summaries twin padded the same way (0 changes after 2026 week 4)."""
    _raw_root(monkeypatch, tmp_path, GAMES)
    built = _fake_summaries(monkeypatch)

    out = derived.build_team_summaries_weekly(
        2026, base=str(tmp_path / "out"), today=today
    )

    assert built == weeks
    assert out["through_week"].to_list() == weeks


def _fake_summaries(monkeypatch) -> list[int]:
    """Stub ``build_summaries_season`` to write a one-row snapshot; return the
    ``through_week`` list it is called with."""
    from cfb_data_build import summaries_build
    from cfb_data_build.config import SUMMARIES_REGISTRY

    spec = SUMMARIES_REGISTRY["team_summaries"]
    built: list[int] = []

    def fake_build(season, *, through_week, base, publish):
        built.append(through_week)
        snap = (
            Path(base)
            / "snapshots"
            / f"through_wk{through_week:02d}"
            / spec.dataset
            / "parquet"
            / f"{spec.stem}_{season}.parquet"
        )
        snap.parent.mkdir(parents=True)
        pl.DataFrame({"team_id": [1]}).write_parquet(snap)

    monkeypatch.setattr(summaries_build, "build_summaries_season", fake_build)
    return built


@pytest.mark.parametrize("today", [TODAY, AFTER_SEASON])
def test_schedule_outage_builds_nothing(monkeypatch, tmp_path, capsys, today):
    """``load_cfb_schedule`` gave up (or 2026's schedule is not published yet):
    every week is unbounded. team_summaries_weekly builds by week number, so it
    republished every one of them -- the whole season padded. Nothing is built,
    and the fetch's GAVE UP line still says why.
    """
    sdv_cfb = _raw_root(monkeypatch, tmp_path, GAMES)

    def down(seasons):
        raise ConnectionError("cfb_schedules unreachable")

    monkeypatch.setattr(sdv_cfb, "load_cfb_schedule", down)
    monkeypatch.setattr(derived, "_RETRY_BACKOFF_S", 0)
    built = _fake_summaries(monkeypatch)
    rated: list = []
    monkeypatch.setattr(
        sdv_cfb, "cfb_ratings", lambda season, *, as_of_date: rated.append(as_of_date)
    )

    summaries = derived.build_team_summaries_weekly(
        2026, base=str(tmp_path / "out"), today=today
    )
    ratings = derived.build_ratings_weekly(2026, today=today)

    assert built == [] and summaries.height == 0
    assert rated == [] and ratings.height == 0
    assert "GAVE UP" in capsys.readouterr().out


@pytest.mark.parametrize(
    "today, weeks",
    [
        # week 14's master kickoff (12-04 UTC) is still ahead
        (dt.date(2026, 12, 1), [3, 4, 5, 13]),
        # it has passed, but load_cfb_schedule has no FBS week-14 game yet
        (dt.date(2026, 12, 4), [3, 4, 5, 13]),
        # week 15 has kicked off; its own bound (12-13) is still ahead
        (dt.date(2026, 12, 12), [3, 4, 5, 13]),
        # week 15's own bound: the season has moved past week 14
        (dt.date(2026, 12, 13), [3, 4, 5, 13, 14, 15]),
    ],
)
def test_borrowed_week_waits_until_the_season_moves_past_it(
    monkeypatch, tmp_path, today, weeks
):
    """2026 week 14 has no FBS game until its championship matchups are set, so
    it borrows week 13's bound (11-30); on the bound alone it passed from then
    on as week 13 relabelled. Its master kickoff (12-04 UTC) is not enough
    either: load_cfb_schedule can still lack every week-14 game then, and the
    snapshot is still week 13 relabelled. A borrowed bound waits until a later
    week has its own bound, or the postseason has started.
    """
    _raw_root(monkeypatch, tmp_path, GAMES)
    assert _cuts(2026)[14] == _cuts(2026)[13] < dt.date(2026, 12, 1)

    assert [w for w, _ in derived.played_cutoffs(2026, today)] == weeks


#: 2019's first FBS bowl, a real ``load_cfb_schedule`` row (the fixture has none).
BOWL_2019 = pl.DataFrame(
    {
        "game_id": [401135260],
        "season": [2019],
        "week": [1],
        "season_type_id": [3],
        "start_date": ["2019-12-20T19:00:00.000Z"],
        "status": ["STATUS_FINAL"],
        "home_division": ["fbs"],
        "away_division": ["fbs"],
    },
    schema=GAMES.schema,
)


@pytest.mark.parametrize(
    "today, weeks",
    [
        (dt.date(2019, 12, 19), [15]),
        (dt.date(2019, 12, 20), [15, 16]),
        (None, [15, 16]),
    ],
)
def test_borrowed_final_week_waits_for_the_postseason(
    monkeypatch, tmp_path, today, weeks
):
    """2019 week 16 (Army-Navy, week 15 to load_cfb_schedule) borrows week 15's
    bound and has no later week to move past. The first bowl moves the season
    past it, so a completed season keeps it, as it always has.
    """
    _raw_root(monkeypatch, tmp_path, pl.concat([GAMES, BOWL_2019]))
    assert _cuts(2019)[16] == _cuts(2019)[15]

    assert [w for w, _ in derived.played_cutoffs(2019, today)] == weeks
