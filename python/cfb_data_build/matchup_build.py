"""Season builders for the two matchup datasets (derived-dataset entry points).

``cfb_matchup_features`` -- one row per FBS team-game, the team's as-of
features for that game (``team_game_features`` with the strict rule: a team's
first game has no current-season input and is null).

``cfb_matchup_line`` -- one row per FBS-vs-FBS non-bowl game, the 268-column
matchup line (``build_matchup_line``).

Inputs per season: the ``cfbfastR_cfb_pbp`` release (sdv-py
``load_cfb_pbp_r``), CFBD ``/games`` for the season and the prior one
(pregame ELO, divisions, points -- ``schedules_unified.fetch_cfbd_games``),
CFBD ``/lines``, and the stage-35 artifacts bundled at
``cfb_model_build/cfb_matchup/artifacts/``. The prior season's FULL-season
features and pace (the line's ``prev_*`` and week-1 fills) are computed here
too, from the prior season's release -- so a line build reads two seasons of
pbp. Everything network-facing is isolated in the ``fetch_*`` names.
"""

from __future__ import annotations

import json
import os
from datetime import date
import urllib.request
from pathlib import Path

import polars as pl

from cfb_data_build.matchup_elo import matchup_scope
from cfb_data_build.matchup_features import (
    TEAM_FEATURE_COLUMNS,
    augment_plays,
    load_coefficients,
    load_wepa_weights,
    pace_history,
    team_game_features,
)
from cfb_data_build.matchup_line import PACE_COLS, build_matchup_line
from cfb_data_build.schedules_unified import fetch_cfbd_games

ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "cfb_model_build"
    / "cfb_matchup"
    / "artifacts"
)
CFBD_LINES = "https://api.collegefootballdata.com/lines"

#: the columns the CFBD games payload contributes (``cfbd_games_elo_*`` shape)
_GAME_FIELDS = {
    "id": "game_id",
    "season": "season",
    "week": "week",
    "seasonType": "season_type",
    "startDate": "start_date",
    "neutralSite": "neutral_site",
    "conferenceGame": "conference_game",
    "venueId": "venue_id",
    "notes": "notes",
    "homeId": "home_id",
    "homeTeam": "home_team",
    "homeConference": "home_conference",
    "homeClassification": "home_division",
    "homePoints": "home_points",
    "homePregameElo": "home_pregame_elo",
    "homePostgameElo": "home_postgame_elo",
    "awayId": "away_id",
    "awayTeam": "away_team",
    "awayConference": "away_conference",
    "awayClassification": "away_division",
    "awayPoints": "away_points",
    "awayPregameElo": "away_pregame_elo",
    "awayPostgameElo": "away_postgame_elo",
}


def tidy_cfbd_games(payload: list[dict]) -> pl.DataFrame:
    """CFBD ``/games`` objects -> the frame every matchup unit reads (pure)."""
    rows = [{out: g.get(src) for src, out in _GAME_FIELDS.items()} for g in payload]
    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Int64,
            "season": pl.Int64,
            "week": pl.Int64,
            "season_type": pl.Utf8,
            "start_date": pl.Utf8,
            "neutral_site": pl.Boolean,
            "conference_game": pl.Boolean,
            "venue_id": pl.Int64,
            "notes": pl.Utf8,
            "home_id": pl.Int64,
            "home_team": pl.Utf8,
            "home_conference": pl.Utf8,
            "home_division": pl.Utf8,
            "home_points": pl.Int64,
            "home_pregame_elo": pl.Int64,
            "home_postgame_elo": pl.Int64,
            "away_id": pl.Int64,
            "away_team": pl.Utf8,
            "away_conference": pl.Utf8,
            "away_division": pl.Utf8,
            "away_points": pl.Int64,
            "away_pregame_elo": pl.Int64,
            "away_postgame_elo": pl.Int64,
        },
    )


def tidy_cfbd_lines(payload: list[dict]) -> pl.DataFrame:
    """CFBD ``/lines`` objects -> one row per (game, provider) (pure)."""
    rows = [
        {
            "game_id": g["id"],
            "provider": ln.get("provider"),
            "spread": ln.get("spread"),
            "spread_open": ln.get("spreadOpen"),
            "over_under": ln.get("overUnder"),
            "over_under_open": ln.get("overUnderOpen"),
        }
        for g in payload
        for ln in (g.get("lines") or [])
    ]
    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Int64,
            "provider": pl.Utf8,
            "spread": pl.Float64,
            "spread_open": pl.Float64,
            "over_under": pl.Float64,
            "over_under_open": pl.Float64,
        },
    )


def fetch_cfbd_lines(season: int, *, api_key: str | None = None) -> list[dict]:
    key = api_key or os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set -- cfb_matchup_line reads CFBD /lines"
        )
    req = urllib.request.Request(
        f"{CFBD_LINES}?year={season}&seasonType=both",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https host
        payload = json.loads(resp.read())
    return payload if isinstance(payload, list) else []


def load_pbp(season: int) -> pl.DataFrame:
    from sportsdataverse.cfb import load_cfb_pbp_r

    return load_cfb_pbp_r([season])


def augmented_season(pbp: pl.DataFrame) -> pl.DataFrame:
    return augment_plays(
        pbp,
        load_wepa_weights(ARTIFACTS / "wepa_weights.json"),
        load_coefficients(ARTIFACTS / "rush_expect_coef.json"),
    )


def fbs_teams(games: pl.DataFrame) -> list[str]:
    scope = matchup_scope(games)
    return sorted(set(scope["home_team"]) | set(scope["away_team"]))


def played_games(plays: pl.DataFrame) -> pl.DataFrame:
    """The (game_id, date, teams) list the as-of loop iterates, from the pbp."""
    return plays.unique(subset=["game_id"], maintain_order=True).select(
        "game_id", start_date=pl.col("play_date"), home_team="home", away_team="away"
    )


def full_season_features(plays: pl.DataFrame, teams: list[str]) -> pl.DataFrame:
    """Every team's features over its WHOLE season (the source's synthetic future game)."""
    synth = pl.DataFrame(
        {
            "game_id": list(range(-1, -len(teams) - 1, -1)),
            "start_date": [date(2100, 1, 1)] * len(teams),
            "home_team": teams,
            "away_team": ["SYNTH_FULL_SEASON"] * len(teams),
        }
    )
    # the synthetic game must come AFTER the team's real games: the loop's
    # first-game rule would otherwise see it as an opener with no prior plays
    games = pl.concat([played_games(plays), synth], how="vertical_relaxed")
    scoring = load_coefficients(ARTIFACTS / "scoring_opp_coef.json")
    out = team_game_features(plays, games, teams, scoring)
    return out.filter(pl.col("game_id") < 0).drop("game_id")


def full_season_pace(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per team, the season's pace over ALL its plays (the source's ``team_pace_full``)."""
    from cfb_data_build.matchup_features import play_pace

    paced = play_pace(pbp)
    parts = []
    for side, col in (("off", "offense_play"), ("def", "defense_play")):
        parts.append(
            paced.group_by(pl.col(col).alias("team"))
            .agg(
                pl.col("sec_since_prev").mean().alias(f"{side}_sec_per_play_mean"),
                pl.col("sec_since_prev").median().alias(f"{side}_sec_per_play_median"),
            )
            .drop_nulls("team")
        )
    return (
        parts[0]
        .join(parts[1], on="team", how="full", coalesce=True)
        .select(["team", *PACE_COLS])
    )


def build_matchup_features(season: int, *, base: str = "cfb") -> pl.DataFrame:
    games = tidy_cfbd_games(fetch_cfbd_games(season))
    plays = augmented_season(load_pbp(season))
    scoring = load_coefficients(ARTIFACTS / "scoring_opp_coef.json")
    feats = team_game_features(
        plays, played_games(plays), fbs_teams(games), scoring, first_game_same_day=False
    )
    meta = games.select(
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date",
        "home_id",
        "home_team",
        "away_id",
        "away_team",
    )
    out = feats.join(meta, on="game_id", how="left").with_columns(
        team_id=pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.col("home_id"))
        .otherwise(pl.col("away_id"))
    )
    return out.select(
        [
            "season",
            "week",
            "season_type",
            "game_id",
            "start_date",
            "team_id",
            "team",
            *TEAM_FEATURE_COLUMNS,
        ]
    ).sort(["start_date", "game_id", "team"])


def build_matchup_line_season(season: int, *, base: str = "cfb") -> pl.DataFrame:
    games = tidy_cfbd_games(fetch_cfbd_games(season))
    prev_games = tidy_cfbd_games(fetch_cfbd_games(season - 1))
    lines = tidy_cfbd_lines(fetch_cfbd_lines(season))
    pbp = load_pbp(season)
    plays = augmented_season(pbp)
    scoring = load_coefficients(ARTIFACTS / "scoring_opp_coef.json")
    teams = fbs_teams(games)
    feats = team_game_features(
        plays, played_games(plays), teams, scoring, first_game_same_day=False
    )
    prev_pbp = load_pbp(season - 1)
    prev_plays = augmented_season(prev_pbp)
    prev_feats = full_season_features(prev_plays, fbs_teams(prev_games))
    return build_matchup_line(
        games,
        prev_games,
        feats,
        prev_feats,
        pace_history(pbp),
        full_season_pace(prev_pbp),
        lines,
    )
