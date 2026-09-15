"""Unit F -- the per-team as-of aggregation -- vs the R loop run today.

Oracles (``fixtures/matchup/README.md``): ``team_features_asof_2025.csv`` (one
row per FBS team-game, the pipeline's loop verbatim over the 2025 release) and
``team_features_full_2025_r.csv`` (synthetic future game -> whole season). Both
were produced by ``capture_team_features.R`` against the SAME release parquet the
Python side reads, so any delta is the port's.

Parity bar: every one of the 42 feature columns within 1e-6 (R ``mean`` /
``median`` in long double vs polars float64 over a few thousand terms), null
placement identical (R ``NaN`` from an empty mean and ``NA`` from an empty
median are both "missing" here).

The full-season pbp is needed as INPUT (a team's as-of window is the whole
season before the game), so parity is ``integration``-marked; the committed
sample supports only the structural test.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cfb_data_build.matchup_features import (
    TEAM_FEATURE_COLUMNS,
    TEAM_NAME_MAPPING,
    augment_plays,
    dedupe_plays,
    load_coefficients,
    load_wepa_weights,
    team_game_features,
)

FIX = Path(__file__).parent / "fixtures" / "matchup"
CACHE = Path(__file__).parents[2] / "python" / ".cache" / "matchup"


def _games_from(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.unique(subset=["game_id"], maintain_order=True).select(
        "game_id", start_date=pl.col("play_date"), home_team="home", away_team="away"
    )


def _augmented(pbp: pl.DataFrame) -> pl.DataFrame:
    return augment_plays(
        pbp,
        load_wepa_weights(FIX / "wepa_weights.json"),
        load_coefficients(FIX / "rush_expect_coef.json"),
    )


def _assert_features_close(
    py: pl.DataFrame, oracle: pl.DataFrame, *, tol: float
) -> None:
    worst: dict[str, float] = {}
    for c in TEAM_FEATURE_COLUMNS:
        a = py[c].cast(pl.Float64).to_numpy()
        b = oracle[c].cast(pl.Float64).to_numpy()
        assert (np.isnan(a) == np.isnan(b)).all(), f"{c}: null placement differs"
        ok = ~np.isnan(a)
        d = np.abs(a[ok] - b[ok])
        worst[c] = float(d.max()) if d.size else 0.0
    bad = {c: d for c, d in worst.items() if d > tol}
    assert not bad, f"columns over {tol}: {bad}"


def test_sample_structure_and_first_game_rule() -> None:
    pbp = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    aug = _augmented(pbp)
    games = _games_from(aug)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))[:12]
    scoring = load_coefficients(FIX / "scoring_opp_coef.json")

    faithful = team_game_features(aug, games, teams, scoring, first_game_same_day=True)
    assert faithful.columns == ["game_id", "team", *TEAM_FEATURE_COLUMNS]
    assert faithful.height == sum(
        games.filter((pl.col("home_team") == t) | (pl.col("away_team") == t)).height
        for t in teams
    )

    strict = team_game_features(aug, games, teams, scoring)
    first_rows = strict.group_by("team", maintain_order=True).first()
    assert first_rows.select(
        TEAM_FEATURE_COLUMNS
    ).null_count().sum_horizontal().item() == len(TEAM_FEATURE_COLUMNS) * len(teams)
    # every later game is identical under both rules -- the rule touches game 1 only
    later = pl.concat(
        [g.slice(1) for _, g in faithful.group_by("team", maintain_order=True)]
    )
    later_strict = pl.concat(
        [g.slice(1) for _, g in strict.group_by("team", maintain_order=True)]
    )
    assert later.equals(later_strict)


@pytest.mark.integration
def test_full_season_features_match_r() -> None:
    pbp = dedupe_plays(pl.read_parquet(CACHE / "cfbfastR_cfb_pbp_2025.parquet"))
    aug = _augmented(pbp)
    oracle = pl.read_csv(
        FIX / "team_features_full_2025_r.csv", infer_schema_length=10000
    ).with_columns(pl.col("team").replace(TEAM_NAME_MAPPING))
    teams = oracle["team"].to_list()
    scoring = load_coefficients(FIX / "scoring_opp_coef.json")
    # the pipeline's synthetic future game: every play of the season is "prior"
    games = pl.DataFrame(
        {
            "game_id": list(range(-1, -len(teams) - 1, -1)),
            "start_date": [date(2026, 2, 15)] * len(teams),
            "home_team": teams,
            "away_team": ["SYNTH_FULL_SEASON"] * len(teams),
        }
    )
    games = pl.concat([_games_from(aug), games], how="vertical_relaxed")
    py = (
        team_game_features(aug, games, teams, scoring, first_game_same_day=True)
        .filter(pl.col("game_id") < 0)
        .sort("team")
    )
    oracle = oracle.sort("team")
    assert py.height == oracle.height == 136
    assert py["team"].to_list() == oracle["team"].to_list()
    _assert_features_close(py, oracle, tol=1e-6)


@pytest.mark.integration
def test_asof_features_match_r() -> None:
    pbp = dedupe_plays(pl.read_parquet(CACHE / "cfbfastR_cfb_pbp_2025.parquet"))
    aug = _augmented(pbp)
    oracle = pl.read_csv(
        FIX / "team_features_asof_2025.csv",
        infer_schema_length=10000,
        null_values=["NA", ""],
    ).with_columns(pl.col("team").replace(TEAM_NAME_MAPPING))
    teams = sorted(oracle["team"].unique().to_list())
    games = _games_from(aug)
    py = team_game_features(
        aug,
        games,
        teams,
        load_coefficients(FIX / "scoring_opp_coef.json"),
        first_game_same_day=True,
    )
    keys = ["team", "game_id"]
    py, oracle = py.sort(keys), oracle.sort(keys)
    assert py.height == oracle.height
    assert py.select(keys).equals(oracle.select(keys).cast(py.select(keys).schema))
    _assert_features_close(py, oracle, tol=1e-6)
