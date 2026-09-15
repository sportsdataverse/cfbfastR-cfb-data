"""The as-of boundary of the matchup features, proven by mutation.

A leakage test that only re-derives the filter would pass against any filter.
Instead: inject a play ON each team's game day with an absurd EPA and assert
the published rule (``first_game_same_day=False``) leaves every feature of
every row untouched, while the source's faithful rule (``True``) is moved for
the first game -- which is exactly the same-day read the published dataset
refuses to make.
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
    return aug, games, scoring


def _poisoned(aug: pl.DataFrame) -> pl.DataFrame:
    """Every play's EPA multiplied by 1000 -- same rows, same dates, absurd values."""
    return aug.with_columns(
        (pl.col("ppa") * 1000).alias("ppa"),
        (pl.col("off_wepa") * 1000).alias("off_wepa"),
    )


def test_strict_rule_never_reads_the_game_being_predicted() -> None:
    aug, games, scoring = _inputs()
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))[:10]
    clean = team_game_features(aug, games, teams, scoring, first_game_same_day=False)
    # poison ONLY the plays dated on each team's own game days -- with the
    # sample (one game per team), that is every play the strict rule must ignore
    poisoned = team_game_features(
        _poisoned(aug), games, teams, scoring, first_game_same_day=False
    )
    assert clean.equals(poisoned)
    # every row is null: the sample holds one game per team, so nothing is prior
    assert (
        clean.select(TEAM_FEATURE_COLUMNS).null_count().sum_horizontal().item()
        == len(TEAM_FEATURE_COLUMNS) * clean.height
    )


def test_faithful_rule_does_read_the_first_game() -> None:
    aug, games, scoring = _inputs()
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))[:10]
    faithful = team_game_features(aug, games, teams, scoring, first_game_same_day=True)
    poisoned = team_game_features(
        _poisoned(aug), games, teams, scoring, first_game_same_day=True
    )
    assert not faithful.equals(poisoned)
    assert faithful["off_epa"].drop_nulls().len() > 0
