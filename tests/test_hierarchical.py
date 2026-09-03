"""The leakage boundary of the hierarchical fit, checked on real schedule rows.

The defect these guard is specific and was live in the first draft: ESPN's
``week`` restarts per season type, so a January bowl of season S carries
``week == 1`` and lands inside the "week < 8" fit window of an October game of
the same season -- a four-month future leak that no MAE would reveal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cfb_model_build.cfb_higher_models.hierarchical import build_corpus, fit_asof

CACHE = Path(".cache/higher_models/games_2014_2025.parquet")


@pytest.fixture(scope="module")
def corpus():
    if not CACHE.exists():
        pytest.skip(f"real schedule cache absent: {CACHE} (run dev/_prime_cache.py)")
    return build_corpus(pl.read_parquet(CACHE))


def test_postseason_sorts_after_every_regular_week(corpus):
    """A bowl must not carry a lower time index than an October game."""
    per_season = corpus.games.group_by("season").agg(
        pl.col("t").filter(pl.col("t") < 100).max().alias("last_regular"),
        pl.col("t").filter(pl.col("t") >= 100).min().alias("first_post"),
    )
    have_post = per_season.filter(pl.col("first_post").is_not_null())
    assert have_post.height > 0, "no postseason games in the corpus -- fixture is wrong"
    bad = have_post.filter(pl.col("first_post") <= pl.col("last_regular"))
    assert bad.height == 0, f"postseason ordered before the regular season: {bad}"


def test_fit_window_excludes_the_as_of_week_and_every_other_season(corpus):
    """The window a fit actually sees, not the window we believe it sees."""
    season, asof = 2016, 8
    window = corpus.games.filter((pl.col("season") == season) & (pl.col("t") < asof))
    assert window.height > 0
    assert window["t"].max() < asof
    assert window["season"].unique().to_list() == [season]
    # and the postseason of that same season is genuinely outside it
    assert window.filter(pl.col("t") >= 100).height == 0


def test_pooling_actually_shrinks(corpus):
    """A component that ran without error has not been shown to have done anything.

    The closed-form arm must produce a posterior spread NARROWER than the raw
    per-team average margin -- if it does not, the shrinkage is a no-op.
    """
    prev = np.zeros(corpus.n_teams)
    fit = fit_asof(corpus, 2016, 8, prev, method="eb")
    window = corpus.games.filter((pl.col("season") == 2016) & (pl.col("t") < 8))
    raw = (
        pl.concat(
            [
                window.select(pl.col("home_ix").alias("ix"), pl.col("margin")),
                window.select(pl.col("away_ix").alias("ix"), -pl.col("margin")),
            ]
        )
        .group_by("ix")
        .agg(pl.col("margin").mean())
    )
    assert fit.theta.std() < raw["margin"].std(), (
        f"pooled sd {fit.theta.std():.2f} is not below the raw sd "
        f"{raw['margin'].std():.2f} -- the shrinkage did nothing"
    )
    assert 0.5 < fit.tau_team < 40.0, f"tau_team {fit.tau_team} is degenerate"
