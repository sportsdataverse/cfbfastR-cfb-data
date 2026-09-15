"""WEPA weight search: the scorer vs R, and the search's contract.

Oracles: ``fixtures/matchup/wepa_scores_2025.csv`` -- the source's
``evaluate_weights`` run verbatim over the 2025 release for five rows of its
500-candidate table plus the shipped weights (``capture_wepa_scores.R``, R
4.6.1, 2026-09-15); ``wepa_search_evals.csv`` -- the source's own 500
evaluated candidates with their in-sample adj-R² on its 2014-2023 corpus
(informational only: that corpus is not the current release, so those numbers
are not reproduced here, only the winner's identity).

Bars: adj-R² and residual sigma within 1e-6 of R for every candidate on the
same plays and games (the scorer is an OLS on identical inputs); the shipped
weights are the argmax of the source's table (row 302, adj-R² 0.2509 -- the
selection rule the port keeps). The 2025 parity needs the cached full-season
pbp, so it runs under ``-m integration``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.matchup_features import (
    TEAM_COLUMNS,
    dedupe_plays,
    load_wepa_weights,
    to_display_names,
)
from cfb_model_build.cfb_matchup.wepa import (
    WEIGHT_NAMES,
    load_search_table,
    random_candidates,
    score_frame,
    score_weights,
    search,
    tag_for_search,
    team_game_wepa,
    weights_of,
)

FIX = Path(__file__).parents[1] / "cfb_data_build" / "fixtures" / "matchup"
CACHE = Path(__file__).parents[2] / "python" / ".cache" / "matchup"


def _games(season: int) -> pl.DataFrame:
    g = pl.read_parquet(FIX / f"cfbd_games_elo_{season}.parquet")
    return (
        to_display_names(g, "home_team", "away_team")
        .with_columns(start_date=pl.col("start_date").str.slice(0, 10).str.to_date())
        .filter(
            pl.col("home_points").is_not_null() & pl.col("away_points").is_not_null()
        )
    )


def _tagged(pbp: pl.DataFrame) -> pl.DataFrame:
    return tag_for_search(
        to_display_names(pbp, *[c for c in TEAM_COLUMNS if c in pbp.columns])
    )


def test_search_table_and_shipped_weights_agree() -> None:
    evals = load_search_table(FIX / "wepa_search_evals.csv")
    assert evals.height == 500 and set(WEIGHT_NAMES) <= set(evals.columns)
    best = evals.sort("r2", descending=True).row(0, named=True)
    shipped = load_wepa_weights(FIX / "wepa_weights.json")
    # the CSV round-trip of the table costs one ulp on a few weights
    assert max(abs(weights_of(best)[k] - shipped[k]) for k in WEIGHT_NAMES) <= 1e-12
    assert (
        abs(best["r2"] - 0.250909) < 1e-6
    )  # the source's in-sample score of the winner


def test_random_candidates_are_reproducible_and_bounded() -> None:
    a, b = random_candidates(5, seed=4), random_candidates(5, seed=4)
    assert a.equals(b) and a.columns == ["candidate", *WEIGHT_NAMES]
    vals = a.select(WEIGHT_NAMES).to_numpy()
    assert vals.min() >= -1.0 and vals.max() <= 1.0


def test_search_scores_every_random_candidate() -> None:
    """Every random candidate comes back scored, with the candidates' key dtype.

    ``random_candidates`` numbers rows with ``with_row_index`` (UInt32); the
    scorer's rows are built from Python ints (Int64). polars 1.41 coerces the
    two to a supertype so the raw join already scored every row; the explicit
    cast in ``search`` pins that across versions (the repo's join-key rule),
    and this test pins the contract the cast serves.
    """
    pbp = dedupe_plays(pl.read_parquet(FIX / "pbp_input_2025_sample.parquet"))
    games = _games(2025).filter(
        pl.col("game_id").is_in(pbp["game_id"].unique().to_list())
    )
    cands = random_candidates(3, seed=7)
    out = search(_tagged(pbp), games, cands)
    assert out.height == 3 and out.schema["candidate"] == cands.schema["candidate"]
    assert out["adj_r2"].null_count() == 0 and out["n_games"].min() > 0


def test_team_game_wepa_is_as_of_on_the_sample() -> None:
    pbp = dedupe_plays(pl.read_parquet(FIX / "pbp_input_2025_sample.parquet"))
    tagged = _tagged(pbp)
    from cfb_data_build.matchup_features import apply_wepa

    plays = apply_wepa(tagged, load_wepa_weights(FIX / "wepa_weights.json"))
    games = _games(2025).filter(
        pl.col("game_id").is_in(pbp["game_id"].unique().to_list())
    )
    frame = team_game_wepa(plays, games)
    assert frame.height == games.height
    # a team's first sample game has no prior plays: its mean is null; a second
    # game has them (12 two-game teams in the sample)
    assert frame["home_off_wepa"].null_count() + frame["away_off_wepa"].null_count() > 0
    assert (
        frame["home_off_wepa"].drop_nulls().len()
        + frame["away_off_wepa"].drop_nulls().len()
        >= 12
    )
    # the OLS is well-defined on the rows that have all four means
    score = score_frame(frame)
    assert score.n_games == frame.drop_nulls().height and -1.0 <= score.adj_r2 <= 1.0


@pytest.mark.integration
def test_scorer_matches_r_on_2025() -> None:
    oracle = pl.read_csv(FIX / "wepa_scores_2025.csv", null_values=["NA", ""])
    evals = load_search_table(FIX / "wepa_search_evals.csv")
    pbp = dedupe_plays(pl.read_parquet(CACHE / "cfbfastR_cfb_pbp_2025.parquet"))
    # the scorer's corpus is the games that HAVE plays (as the capture defined it)
    tagged = _tagged(pbp)
    games = _games(2025).filter(
        pl.col("game_id").is_in(pbp["game_id"].unique().to_list())
    )
    shipped = load_wepa_weights(FIX / "wepa_weights.json")
    for row in oracle.iter_rows(named=True):
        cand = row["candidate"]
        w = (
            shipped
            if cand == "shipped"
            else weights_of(evals.row(int(cand.split("_")[1]) - 1, named=True))
        )
        s = score_weights(tagged, games, w)
        assert s.n_games == row["n_games"], cand
        assert abs(s.adj_r2 - row["r2"]) <= 1e-6, (cand, s.adj_r2, row["r2"])
        assert abs(s.sigma - row["sd_err"]) <= 1e-6, (cand, s.sigma, row["sd_err"])
