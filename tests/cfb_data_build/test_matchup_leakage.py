"""The as-of boundary of the matchup features, proven by mutation.

A leakage test that only re-derives the filter would pass against any filter.
Instead, for each team with TWO games in the committed sample, poison every
play of its second game and assert its second-game row is unchanged -- then
poison its first game and assert the row moves.
The first check is what a ``<`` -> ``<=`` regression in
``team_game_features`` breaks (verified: with the comparison mutated to ``<=``
the second-game poison moves the row and the test goes red); the second
check proves the assertion is not vacuous.

The source's faithful rule (``first_game_same_day=True``) is kept only for the
R-parity tests; the published default reads nothing dated on the game day.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cfb_data_build.matchup_features import (
    TEAM_FEATURE_COLUMNS,
    augment_plays,
    load_coefficients,
    load_wepa_weights,
    team_game_features,
)

FIX = Path(__file__).parent / "fixtures" / "matchup"
POISON = 1000.0


def _inputs():
    pbp = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    aug = augment_plays(
        pbp,
        load_wepa_weights(FIX / "wepa_weights.json"),
        load_coefficients(FIX / "rush_expect_coef.json"),
    )
    games = aug.unique(subset=["game_id"], maintain_order=True).select(
        "game_id", start_date=pl.col("play_date"), home_team="home", away_team="away"
    )
    scoring = load_coefficients(FIX / "scoring_opp_coef.json")
    stacked = pl.concat(
        [
            games.select("game_id", "start_date", team="home_team"),
            games.select("game_id", "start_date", team="away_team"),
        ]
    )
    two = (
        stacked.group_by("team")
        .agg(n=pl.len(), second=pl.col("start_date").max())
        .filter(pl.col("n") == 2)
        .sort("team")
    )
    return aug, games, scoring, two


def _poison(aug: pl.DataFrame, game_id: int) -> pl.DataFrame:
    """Multiply the EPA-family columns by 1000 on every play of one game."""
    when = pl.col("game_id") == game_id
    return aug.with_columns(
        [
            pl.when(when).then(pl.col(c) * POISON).otherwise(pl.col(c)).alias(c)
            for c in ("ppa", "off_wepa", "def_wepa", "rroe")
        ]
    )


def _team_games(games: pl.DataFrame, team: str) -> pl.DataFrame:
    return games.filter(
        (pl.col("home_team") == team) | (pl.col("away_team") == team)
    ).sort("start_date")


def _second_row(
    aug: pl.DataFrame, games: pl.DataFrame, team: str, scoring
) -> pl.DataFrame:
    """The team's second-game row under the published (default) rule."""
    out = team_game_features(aug, games, [team], scoring)
    second = _team_games(games, team)["game_id"][1]
    return out.filter(pl.col("game_id") == second).select(TEAM_FEATURE_COLUMNS)


def test_same_day_plays_never_reach_the_row() -> None:
    aug, games, scoring, two = _inputs()
    assert two.height == 12  # the sample's two-game teams
    for team in two["team"].to_list():
        first, second = _team_games(games, team)["game_id"].to_list()
        clean = _second_row(aug, games, team, scoring)
        assert clean["off_epa"].null_count() == 0, team  # the first game feeds it
        # poison the team's OWN second game: its second-game row must not move
        poisoned = _second_row(_poison(aug, second), games, team, scoring)
        assert clean.equals(poisoned), team


def test_prior_plays_do_reach_the_row() -> None:
    aug, games, scoring, two = _inputs()
    for team in two["team"].to_list():
        first, second = _team_games(games, team)["game_id"].to_list()
        clean = _second_row(aug, games, team, scoring)
        # poison the team's OWN first game: its second-game row must move
        poisoned = _second_row(_poison(aug, first), games, team, scoring)
        assert not clean.equals(poisoned), team
