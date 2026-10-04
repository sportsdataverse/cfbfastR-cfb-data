"""A ``through_week`` summaries snapshot holds only games played by that week.

ESPN restarts postseason week numbering at 1, so the old bare ``week <= W``
filter put every bowl and CFP game into EVERY ``cfb_team_summaries_weekly``
snapshot: the published 2024 asset has Notre Dame at ``valid_games = 5`` for
``through_week == 1`` (Texas A&M plus its four CFP games).

Fixture: real games keyed exactly as the released ``espn_cfb_pbp`` and
``cfb_schedules`` carry them -- all 16 Notre Dame 2024 games, plus 2009 edge
games (regular-season games whose pbp ``seasonType`` is null, and offseason
all-star games ESPN files as week 1). One "play" per game is enough: the
snapshot filter reads only the game keys.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import cfb_data_build.summaries_build as sb
from cfb_data_build.config import SUMMARIES_REGISTRY

FIX = Path(__file__).parent / "fixtures" / "summaries_snapshot_games.parquet"
ND_CFP_2024 = {"401677179", "401677184", "401677189", "401677192"}


def _snapshot_ids(monkeypatch, tmp_path, season: int, through_week: int | None):
    """Game ids that reach ``build_team_summaries`` for one snapshot."""
    games = pl.read_parquet(FIX).filter(pl.col("season") == season)
    pbp = games.select("game_id", "week", "seasonType")
    schedule = games.select("game_id", "week", "season_type", "season_type_id")
    seen: dict = {}

    def fake_build(plays, _season, *, through_week=None, rosters=None):
        seen["ids"] = set(plays["game_id"].to_list())
        seen["through_week"] = through_week
        return {k: None for k in SUMMARIES_REGISTRY}

    # prepare_plays_input needs ~60 pbp columns; its only effect the snapshot
    # filter depends on is the game_id -> Utf8 cast, reproduced here.
    monkeypatch.setattr(
        sb,
        "prepare_plays_input",
        lambda p, _s, _season: p.with_columns(pl.col("game_id").cast(pl.Utf8)),
    )
    monkeypatch.setattr(sb, "build_team_summaries", fake_build)

    # The Paper Index shares ride the same snapshot filter. Stand in one "share"
    # per game (Int64 ids, as the real table carries them) and record which
    # games reach the luck roll-up.
    def fake_attach(team, shares, _season):
        seen["share_ids"] = set(shares["game_id"].cast(pl.Utf8).to_list())
        return team

    monkeypatch.setattr(sb, "paper_index_games_table", lambda p: p.select("game_id"))
    monkeypatch.setattr(sb, "attach_luck", fake_attach)
    monkeypatch.setattr(sb, "write_dataset", lambda *a, **k: None)
    sb.build_summaries_season(
        season,
        through_week=through_week,
        base=str(tmp_path),
        pbp=pbp,
        schedule=schedule,
        rosters=pl.DataFrame(),
    )
    # the snapshot must reach the builder AS a snapshot: that is what exempts it
    # from the full-season no-op gate (checks.assert_adjustment_is_real)
    assert seen["through_week"] == through_week
    # one id list cuts both: the luck columns cover exactly the plays' games
    assert seen["share_ids"] == seen["ids"]
    return seen["ids"]


@pytest.mark.parametrize("week", range(1, 17))
def test_no_postseason_game_in_any_regular_season_snapshot(monkeypatch, tmp_path, week):
    assert not _snapshot_ids(monkeypatch, tmp_path, 2024, week) & ND_CFP_2024


def test_week_one_snapshot_is_week_one_only(monkeypatch, tmp_path):
    assert _snapshot_ids(monkeypatch, tmp_path, 2024, 1) == {"401628332"}
    assert len(_snapshot_ids(monkeypatch, tmp_path, 2024, 16)) == 12


def test_season_final_build_keeps_the_postseason(monkeypatch, tmp_path):
    """Not vacuous: the CFP plays ARE in the input; only snapshots drop them."""
    ids = _snapshot_ids(monkeypatch, tmp_path, 2024, None)
    assert ND_CFP_2024 <= ids and len(ids) == 16


def test_schedule_not_pbp_decides_the_season_type(monkeypatch, tmp_path):
    """pbp ``seasonType`` is null on some real 2009 regular-season games; the
    schedule is complete. Offseason all-star games (week 1) never belong."""
    assert _snapshot_ids(monkeypatch, tmp_path, 2009, 3) == {
        "292480127",
        "292550166",
        "292620024",
        "292620166",
    }
