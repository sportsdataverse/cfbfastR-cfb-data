"""`fit_pregame`: the ship fit never sees its holdout, and walk-forward win
probabilities are priced with an sd the training fold could have known.

Both guard the 2026-09-27 refit on leak-free weekly summaries (cfb-data #100).
The previous ship fit ran on every season it was given, 2024 included, while
sdv-py's `test_cfb_prediction_backtest.py` gates the constants on a 2024
"holdout". And the walk-forward Brier priced each probability with the
residual sd of the very outcomes it was scored against.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cfb_model_build.cfb_higher_models.fit_pregame import (
    fit,
    fit_for_ship,
    from_constants,
    walk_forward_fit,
)


def _frame(noise_by_season: dict[int, float], n: int = 400) -> pl.DataFrame:
    """Games whose margin is 20 * rating diff + 3 at home, plus per-season noise."""
    rng = np.random.default_rng(7)
    parts = []
    for season, noise in noise_by_season.items():
        diff = rng.normal(0.0, 0.3, n)
        neutral = rng.random(n) < 0.1
        games = rng.integers(1, 12, n)
        margin = 20.0 * diff + np.where(neutral, 0.0, 3.0) + rng.normal(0, noise, n)
        parts.append(
            pl.DataFrame(
                {
                    "season": [season] * n,
                    "week": (games + 1).tolist(),
                    "rt_adj_net_home": diff.tolist(),
                    "rt_adj_net_away": [0.0] * n,
                    "neutral_site": neutral.tolist(),
                    "valid_games_home": games.tolist(),
                    "valid_games_away": (games + 1).tolist(),
                    "margin": margin.tolist(),
                    "home_won": (margin > 0).tolist(),
                    "total": rng.normal(55, 10, n).tolist(),
                    "playsgame_off_home": rng.normal(70, 5, n).tolist(),
                    "playsgame_off_away": rng.normal(70, 5, n).tolist(),
                    "adj_off_epa_home": rng.normal(0, 0.1, n).tolist(),
                    "adj_off_epa_away": rng.normal(0, 0.1, n).tolist(),
                }
            )
        )
    return pl.concat(parts)


def test_ship_fit_excludes_the_holdout() -> None:
    frame = _frame({s: 10.0 for s in range(2014, 2020)})
    got = fit_for_ship(frame, holdout=[2018, 2019])
    assert got.seasons == [2014, 2015, 2016, 2017]
    assert got.n_train == frame.filter(pl.col("season") < 2018).height
    # The holdout must actually change the fit: a filter that dropped nothing
    # would reproduce the all-season constants.
    assert got.net_points_scale != fit(frame).net_points_scale


def test_ship_fit_refuses_a_holdout_it_cannot_hold_out() -> None:
    """A holdout season absent from the frame would be a vacuous holdout."""
    with pytest.raises(ValueError, match="2031"):
        fit_for_ship(_frame({2014: 10.0, 2015: 10.0}), holdout=[2031])


def test_walk_forward_prices_wp_with_the_training_sd() -> None:
    # The scored season is 4x noisier than training, so its own residual sd
    # (~40) is nowhere near what the training fold measured (~10).
    frame = _frame({2014: 10.0, 2015: 10.0, 2016: 40.0})
    _rows, _flat, _curve, sds, scored = walk_forward_fit(frame, min_train=2)
    assert scored["season"].unique().to_list() == [2016]
    train_sd = fit(frame.filter(pl.col("season") < 2016)).margin_sd
    np.testing.assert_allclose(sds, train_sd)
    assert train_sd < 15.0


def test_from_constants_scores_the_serving_formula() -> None:
    """The shipped arm must be the formula `cfb_game_predict` serves.

    Games-played slope from the WEAKER side, hfa_points at home only.
    """
    from sportsdataverse.cfb.cfb_game_predict import predict_margin
    from sportsdataverse.cfb.cfb_prediction_constants import get_constants

    frame = _frame({2014: 10.0}, n=60)
    got = from_constants(get_constants("modern")).predict(frame)
    want = [
        predict_margin(h, a, neutral=n, games_played=min(gh, ga))
        for h, a, n, gh, ga in frame.select(
            "rt_adj_net_home",
            "rt_adj_net_away",
            "neutral_site",
            "valid_games_home",
            "valid_games_away",
        ).iter_rows()
    ]
    np.testing.assert_allclose(got, want)


def test_fit_reads_the_rating_it_is_told_to() -> None:
    """GOP's sdv.ts runs the flat form on SUMMARIES `net_adj_epa`, not `rt_adj_net`.

    Here the margin is generated from `net_adj_epa` (slope 50) and
    `rt_adj_net` is unrelated, so a fit that ignored `rating` recovers ~0.
    """
    frame = _frame({2014: 5.0, 2015: 5.0})
    rng = np.random.default_rng(3)
    summ = rng.normal(0.0, 0.2, frame.height)
    frame = frame.with_columns(
        pl.Series("net_adj_epa_home", summ),
        pl.lit(0.0).alias("net_adj_epa_away"),
        pl.Series("rt_adj_net_home", rng.normal(0.0, 0.3, frame.height)),
        (50.0 * pl.Series(summ) + rng.normal(0, 5.0, frame.height)).alias("margin"),
    )
    got = fit_for_ship(frame, holdout=[2015], rating="net_adj_epa")
    assert got.rating == "net_adj_epa"
    assert abs(got.net_points_scale - 50.0) < 3.0
    pred = got.predict(frame, use_curve=False)
    hfa = np.where(frame["neutral_site"].to_numpy(), 0.0, got.hfa_points)
    np.testing.assert_allclose(pred, got.net_points_scale * summ + hfa, atol=1e-9)
