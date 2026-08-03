"""Walk-forward backtest runner + the shipped-predictor reference.

Self-validating by construction: :func:`shipped_predictor` scores the
constants currently in ``cfb_prediction_constants`` through this harness. Those
were measured ad-hoc at MAE 15.13 / Brier 0.2219 over 2023-2025, so a harness
that reports anything materially different is itself broken. Run
``python -m cfb_higher_models backtest`` to check.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .data import build_game_frame
from .metrics import Report, evaluate, norm_cdf


def shipped_margin(frame: pl.DataFrame, *, era: str = "modern") -> np.ndarray:
    """The shipped closed-form margin, as one formula in one place.

    Uses ``net_adj_epa`` from the weekly summaries as the rating differential --
    the same opponent-adjusted quantity ``cfb_ratings`` exposes as ``adj_net``.
    HFA is zeroed at neutral sites, which the shipped surface does not do.
    """
    from sportsdataverse.cfb.cfb_prediction_constants import CFB_CONSTANTS

    c = CFB_CONSTANTS[era]
    hfa = np.where(frame["neutral_site"].to_numpy(), 0.0, 2 * c.hfa_epa)
    diff = frame["rt_adj_net_home"].to_numpy() - frame["rt_adj_net_away"].to_numpy()
    return c.net_points_scale * (diff + hfa)


def shipped_predictor(frame: pl.DataFrame, *, era: str = "modern") -> Report:
    """Score the shipped closed-form margin/WP constants on an as-of frame."""
    from sportsdataverse.cfb.cfb_prediction_constants import CFB_CONSTANTS

    c = CFB_CONSTANTS[era]
    pred = shipped_margin(frame, era=era)
    return evaluate(
        "shipped closed-form (cfb_game_predict)",
        boundary="as_of",
        pred_margin=pred,
        actual_margin=frame["margin"].to_numpy(),
        prob=norm_cdf(pred, c.margin_sd),
        won=frame["home_won"].to_numpy(),
    )


def season_splits(
    frame: pl.DataFrame, *, min_train: int = 3
) -> list[tuple[list[int], int]]:
    """Walk-forward (train_seasons, test_season) pairs.

    Forward-only: a season is tested using only seasons strictly before it.
    LOSO would leak future seasons into the past, which for a *predictive*
    head is exactly the error this package exists to prevent. (LOSO stays
    correct for the play-level models, whose boundary is the play.)
    """
    seasons = sorted(frame["season"].unique().to_list())
    return [(seasons[:i], s) for i, s in enumerate(seasons) if i >= min_train]


def walk_forward(
    frame: pl.DataFrame,
    fit_predict,
    *,
    name: str,
    min_train: int = 3,
    margin_sd: float | None = None,
) -> tuple[Report, pl.DataFrame]:
    """Run ``fit_predict(train_df, test_df) -> pred_margin`` forward in time.

    Returns the pooled out-of-sample report and the per-game OOF predictions.
    """
    preds = []
    for train_seasons, test_season in season_splits(frame, min_train=min_train):
        tr = frame.filter(pl.col("season").is_in(train_seasons))
        te = frame.filter(pl.col("season") == test_season)
        if not tr.height or not te.height:
            continue
        p = np.asarray(fit_predict(tr, te), dtype=float)
        preds.append(
            te.select("game_id", "season", "week", "margin", "home_won").with_columns(
                pl.Series("pred_margin", p)
            )
        )
    if not preds:
        raise ValueError("walk_forward produced no folds -- too few seasons?")
    oof = pl.concat(preds)
    pm = oof["pred_margin"].to_numpy()
    sd = (
        margin_sd
        if margin_sd is not None
        else float(np.std(oof["margin"].to_numpy() - pm))
    )
    rep = evaluate(
        name,
        boundary="as_of",
        pred_margin=pm,
        actual_margin=oof["margin"].to_numpy(),
        prob=norm_cdf(pm, sd),
        won=oof["home_won"].to_numpy(),
    )
    return rep, oof


def by_week(frame: pl.DataFrame, pred: np.ndarray) -> str:
    """Week-bucketed MAE -- early weeks rest on 1-2 games of ratings."""
    d = frame.select("week", "margin").with_columns(pl.Series("pred", pred))
    out = ["  week    n     MAE    bias"]
    for lo, hi, lbl in ((2, 4, "2-4"), (5, 8, "5-8"), (9, 12, "9-12"), (13, 25, "13+")):
        s = d.filter(pl.col("week").is_between(lo, hi))
        if s.height:
            e = s["pred"] - s["margin"]
            out.append(
                f"  {lbl:>5} {s.height:>5} {e.abs().mean():>7.2f} {e.mean():>+7.2f}"
            )
    return "\n".join(out)


def main(seasons: list[int] | None = None) -> int:
    seasons = seasons or [2021, 2022, 2023, 2024, 2025]
    frame = build_game_frame(seasons)
    print(
        f"as-of game frame: {frame.height} games, {len(frame.columns)} cols, seasons {min(seasons)}-{max(seasons)}\n"
    )
    print(shipped_predictor(frame))
    print(by_week(frame, shipped_margin(frame)))
    return 0
