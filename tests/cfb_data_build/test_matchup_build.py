"""Offline smoke of the two season builders (stages 41 / 42) on the committed fixtures.

The builders fetch CFBD and the pbp release; here those three fetchers are
replaced with the fixtures (CFBD games 2024/2025 and lines 2025 re-encoded as
the API payload; the 56-game pbp sample), so the whole production path --
load, dedupe, display-name mapping, schedule iteration, prior-season block,
column contract, id widths -- runs without a network. It is a structural
check, not a parity one (parity lives in the unit tests); it exists because
no parity test exercises ``build_matchup_features`` / ``build_matchup_line_season``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cfb_data_build import matchup_build
from cfb_data_build.matchup_build import _GAME_FIELDS
from cfb_data_build.matchup_features import TEAM_FEATURE_COLUMNS
from cfb_data_build.matchup_line import LINE_COLUMNS

FIX = Path(__file__).parent / "fixtures" / "matchup"


def _games_payload(season: int) -> list[dict]:
    inverse = {out: src for src, out in _GAME_FIELDS.items()}
    frame = pl.read_parquet(FIX / f"cfbd_games_elo_{season}.parquet")
    return [{inverse[k]: v for k, v in row.items()} for row in frame.to_dicts()]


def _lines_payload(season: int) -> list[dict]:
    frame = pl.read_parquet(FIX / "cfbd_lines_2025.parquet")
    out: dict[int, dict] = {}
    for row in frame.to_dicts():
        game = out.setdefault(row["game_id"], {"id": row["game_id"], "lines": []})
        game["lines"].append(
            {
                "provider": row["provider"],
                "spread": row["spread"],
                "spreadOpen": row["spread_open"],
                "overUnder": row["over_under"],
                "overUnderOpen": row["over_under_open"],
            }
        )
    return list(out.values())


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    monkeypatch.setattr(matchup_build, "fetch_cfbd_games", _games_payload)
    monkeypatch.setattr(matchup_build, "fetch_cfbd_lines", _lines_payload)
    # the sample stands in for both seasons; a duplicated slice proves the
    # load-time dedupe is on the production path
    monkeypatch.setattr(
        matchup_build, "load_pbp", lambda season: pl.concat([sample, sample.head(50)])
    )


def test_build_matchup_features_offline(offline: None) -> None:
    out = matchup_build.build_matchup_features(2025)
    assert out.columns == [
        "season", "week", "season_type", "game_id", "start_date", "team_id", "team", *TEAM_FEATURE_COLUMNS,
    ]  # fmt: skip
    assert out.schema["game_id"] == pl.Int64 and out.schema["team_id"] == pl.Int64
    # every FBS team in an FBS-vs-FBS game has a row for every scheduled game
    assert out.height > 1_500 and out["team"].n_unique() >= 130
    assert out.unique(subset=["game_id", "team"]).height == out.height
    # no duplicate rows reach the aggregation (the 50 duplicated plays)
    assert out.filter(pl.col("off_plays_per_game") > 200).height == 0


def test_build_matchup_line_offline(offline: None) -> None:
    out = matchup_build.build_matchup_line_season(2025)
    assert out.columns == list(LINE_COLUMNS)
    assert (
        out.height == 773
    )  # the delivered 2025 line's row set (bowls dropped, CFP kept)
    assert out.schema["game_id"] == pl.Int64
    # not-yet-joined groups carry their documented dtype, not Null
    assert out.schema["home_head_coach"] == pl.Utf8
    assert out.schema["home_dome"] == pl.Boolean
    assert out.schema["temperature"] == pl.Float64
    assert out["spread"].null_count() < out.height


@pytest.mark.parametrize(
    "builder", ["build_matchup_features", "build_matchup_line_season"]
)
def test_builders_guard_a_schedule_spelling(
    offline: None, monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    """A schedule name the plays do not carry raises, naming the team -- in BOTH stages.

    Mutation-proven: with the guard removed a build returns rows whose
    features are null for that team instead of failing.
    """
    sample = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    victim_game = int(sample["game_id"][0])
    games = pl.read_parquet(FIX / "cfbd_games_elo_2025.parquet")
    victim = games.filter(pl.col("game_id") == victim_game)["home_team"][0]

    def renamed(season: int) -> list[dict]:
        payload = _games_payload(season)
        for row in payload:
            for k in ("homeTeam", "awayTeam"):
                if row.get(k) == victim:
                    row[k] = victim + " Univ."
        return payload

    monkeypatch.setattr(matchup_build, "fetch_cfbd_games", renamed)
    with pytest.raises(RuntimeError, match="spelling mismatch"):
        getattr(matchup_build, builder)(2025)


def test_fetch_cfbd_games_rejects_an_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 with CFBD's error object is an error, not an empty season.

    Both matchup stages (and cfb_schedules) read this fetcher; `[]` here let a
    rate-limited run build zero rows and report success.
    """
    from cfb_data_build import schedules_unified

    monkeypatch.setenv("CFBD_API_KEY", "test")
    monkeypatch.setattr(
        schedules_unified, "_get", lambda *a, **k: '{"message": "rate limited"}'
    )
    with pytest.raises(RuntimeError, match="unexpected body"):
        schedules_unified.fetch_cfbd_games(2025)
    monkeypatch.setattr(schedules_unified, "_get", lambda *a, **k: "[]")
    assert schedules_unified.fetch_cfbd_games(2025) == []
