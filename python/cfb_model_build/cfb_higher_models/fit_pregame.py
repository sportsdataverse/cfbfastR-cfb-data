"""Phase 1 -- refit the closed-form pregame constants, honestly and reproducibly.

Replaces ``dev/cfb_prediction/fit_pregame.py``, which the shipped constants cite
but which exists nowhere: not on disk, not in git (``dev/`` is gitignored). Its
numbers could not be reproduced or refreshed, so the constants silently rotted
when the ratings scale moved underneath them.

WHY THE SHIPPED SLOPE IS TOO LARGE (the interesting part)
--------------------------------------------------------
Not merely "the scale changed". The shipped 44.54 was fit against FULL-SEASON
ratings but is APPLIED to as-of (through week W-1) ratings. As-of ratings are
the same quantity measured with more noise, and OLS slopes attenuate toward
zero when the predictor is noisy (classical errors-in-variables). So the
correct multiplier for a noisy early-season rating is SMALLER than the one
fit on a clean full-season rating -- measured here as 23.30 (all weeks) vs
31.40 (week >= 5) vs the shipped 44.54.

Corollary the shipped model gets wrong: the right slope is not one number. It
should grow through the season as the rating firms up. ``fit_slope_by_games``
estimates that curve; ``PregameFit.predict`` applies it.

Everything is fit walk-forward (train on seasons < S, score S) so the reported
metrics are out-of-sample. The constants to ship are fit on every season except
``--holdout``, which is then scored with them frozen: sdv-py gates them on 2024.
Output: ``pregame_fit.json``, committed as ``models/pregame_fit.json`` and cited
by sdv-py ``CFB_CONSTANTS`` and GOP ``sdv.ts`` (see ``models/REGISTRY.md``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .backtest import by_week, season_splits, shipped_predictor
from .data import build_game_frame
from .metrics import evaluate, norm_cdf

# Rating the closed form runs on: `cfb_ratings_weekly.adj_net` (what sdv-py's
# cfb_game_predict serves). GOP's sdv.ts runs the same flat form on the team
# summaries' `net_adj_epa`, a different, ~30% narrower rating, so it gets its
# own fit (`rating="net_adj_epa"`) rather than borrowing this slope.
_RATING = "rt_adj_net"
# Games-played buckets for the attenuation curve. Below ~4 games a rating is
# mostly noise; past ~8 it has largely settled.
_GAME_BUCKETS = ((0, 3), (4, 5), (6, 7), (8, 20))


@dataclass
class PregameFit:
    """Refit closed-form constants. Superset of the shipped PredictConfig."""

    net_points_scale: float  # global slope (attenuation-blind)
    hfa_points: float  # home-field advantage, POINTS not EPA
    margin_sd: float  # residual sd -> win probability
    total_intercept: float
    total_pace_scale: float
    total_epa_scale: float
    slope_by_games: dict[str, float]  # "lo-hi" -> slope, the attenuation curve
    n_train: int
    seasons: list[int]
    rating: str = _RATING
    # Residual sd of the CURVE's margin, which is what sdv-py serves, so this
    # (not the flat fit's ``margin_sd``) is its ``PredictConfig.margin_sd``.
    # GOP's flat form keeps ``margin_sd``.
    margin_sd_curve: float = float("nan")

    def predict(self, frame: pl.DataFrame, *, use_curve: bool = True) -> np.ndarray:
        diff = _diff(frame, self.rating)
        hfa = np.where(
            frame["neutral_site"].to_numpy().astype(bool), 0.0, self.hfa_points
        )
        if not use_curve:
            return self.net_points_scale * diff + hfa
        # Fewer games -> noisier rating -> flatter slope.
        games = _games_played(frame)
        slope = np.full(len(diff), self.net_points_scale, dtype=float)
        for (lo, hi), key in zip(_GAME_BUCKETS, (f"{a}-{b}" for a, b in _GAME_BUCKETS)):
            m = (games >= lo) & (games <= hi)
            if m.any() and key in self.slope_by_games:
                slope[m] = self.slope_by_games[key]
        return slope * diff + hfa


def _games_played(frame: pl.DataFrame) -> np.ndarray:
    """Games behind the WEAKER of the two as-of ratings (the binding one)."""
    for c in ("valid_games_home", "games_home"):
        if c in frame.columns and c.replace("home", "away") in frame.columns:
            return np.minimum(
                frame[c].to_numpy(), frame[c.replace("home", "away")].to_numpy()
            )
    return (frame["week"].to_numpy() - 1).astype(float)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _diff(frame: pl.DataFrame, rating: str) -> np.ndarray:
    return frame[f"{rating}_home"].to_numpy() - frame[f"{rating}_away"].to_numpy()


def fit(frame: pl.DataFrame, rating: str = _RATING) -> PregameFit:
    """Fit every constant on ``frame`` (caller guarantees it is training-only)."""
    diff = _diff(frame, rating)
    margin = frame["margin"].to_numpy()
    # Callers may have cast neutral_site to float for a matrix build; `~` on a
    # float array raises. Normalise rather than assume the caller's dtype.
    neutral = frame["neutral_site"].to_numpy().astype(bool)

    # Slope and HFA jointly: margin ~ slope*diff + hfa*(not neutral).
    X = np.column_stack([diff, (~neutral).astype(float)])
    coef, *_ = np.linalg.lstsq(X, margin, rcond=None)
    slope, hfa = float(coef[0]), float(coef[1])
    resid_sd = float(np.std(margin - X @ coef))

    curve = {}
    games = _games_played(frame)
    for lo, hi in _GAME_BUCKETS:
        m = (games >= lo) & (games <= hi)
        if m.sum() >= 100:
            Xb = np.column_stack([diff[m], (~neutral[m]).astype(float)])
            cb, *_ = np.linalg.lstsq(Xb, margin[m], rcond=None)
            curve[f"{lo}-{hi}"] = float(cb[0])

    # Total points: pace (plays/game) and combined offensive strength.
    total = frame["total"].to_numpy()
    pace = (
        frame["playsgame_off_home"].to_numpy() + frame["playsgame_off_away"].to_numpy()
    )
    off = frame["adj_off_epa_home"].to_numpy() + frame["adj_off_epa_away"].to_numpy()
    Xt = np.column_stack([np.ones(len(total)), pace, off])
    ct, *_ = np.linalg.lstsq(Xt, total, rcond=None)

    out = PregameFit(
        net_points_scale=slope,
        hfa_points=hfa,
        margin_sd=resid_sd,
        total_intercept=float(ct[0]),
        total_pace_scale=float(ct[1]),
        total_epa_scale=float(ct[2]),
        slope_by_games=curve,
        n_train=int(frame.height),
        seasons=sorted(int(s) for s in frame["season"].unique().to_list()),
        rating=rating,
    )
    out.margin_sd_curve = float(np.std(margin - out.predict(frame, use_curve=True)))
    return out


def walk_forward_fit(frame: pl.DataFrame, *, min_train: int = 2, rating: str = _RATING):
    """Out-of-sample scoring of the refit, fold by fold.

    Also returns each scored game's TRAINING-fold ``margin_sd``. A win
    probability priced with the scored games' own residual sd would use the
    outcomes it is then graded against.
    """
    rows, preds_flat, preds_curve, sds, sds_curve, tests = [], [], [], [], [], []
    for train_seasons, test_season in season_splits(frame, min_train=min_train):
        tr = frame.filter(pl.col("season").is_in(train_seasons))
        te = frame.filter(pl.col("season") == test_season)
        if not tr.height or not te.height:
            continue
        f = fit(tr, rating)
        preds_flat.append(f.predict(te, use_curve=False))
        preds_curve.append(f.predict(te, use_curve=True))
        sds.append(np.full(te.height, f.margin_sd))
        sds_curve.append(np.full(te.height, f.margin_sd_curve))
        tests.append(te)
        rows.append((test_season, f.net_points_scale, f.hfa_points, f.margin_sd))
    # Return the scored rows themselves, not just targets: the incumbent has to
    # be scored on EXACTLY these games or the comparison is rigged. (The first
    # `min_train` seasons are consumed as training and never scored.)
    return (
        rows,
        np.concatenate(preds_flat),
        np.concatenate(preds_curve),
        np.concatenate(sds),
        np.concatenate(sds_curve),
        pl.concat(tests),
    )


def fit_for_ship(
    frame: pl.DataFrame, holdout: list[int], rating: str = _RATING
) -> PregameFit:
    """The constants to ship: every season in ``frame`` except ``holdout``.

    sdv-py gates the shipped constants on a 2024 holdout
    (``tests/cfb/test_cfb_prediction_backtest.py``), so that season must not be
    in the fit.
    """
    have = set(frame["season"].unique().to_list())
    missing = sorted(set(holdout) - have)
    if missing:
        raise ValueError(
            f"holdout seasons {missing} are not in the frame; holding out a "
            "season that was never there tests nothing"
        )
    train = frame.filter(~pl.col("season").is_in(holdout))
    if train.is_empty():
        raise ValueError(f"holdout {sorted(holdout)} leaves no training seasons")
    return fit(train, rating)


def from_constants(cfg, rating: str = _RATING) -> PregameFit:
    """Wrap a shipped sdv-py ``PredictConfig`` so it scores through ``predict``.

    ``PregameFit.predict(use_curve=True)`` is the formula
    ``cfb_game_predict.predict_margin`` serves (games-played slope from the
    weaker side, ``hfa_points`` off neutral sites). ``backtest.shipped_margin``
    is not: it still routes ``2 * hfa_epa`` through the flat scale.
    """
    nan = float("nan")
    return PregameFit(
        net_points_scale=cfg.net_points_scale,
        hfa_points=cfg.hfa_points,
        margin_sd=cfg.margin_sd,
        total_intercept=nan,
        total_pace_scale=nan,
        total_epa_scale=nan,
        slope_by_games=dict(cfg.slope_by_games),
        n_train=0,
        seasons=[],
        rating=rating,
        margin_sd_curve=cfg.margin_sd,
    )


def _sdv_sha() -> str | None:
    """The sportsdataverse commit the loaders and shipped constants came from."""
    from importlib.metadata import distribution

    raw = distribution("sportsdataverse").read_text("direct_url.json")
    return json.loads(raw).get("vcs_info", {}).get("commit_id") if raw else None


def _report(label: str, pred: np.ndarray, frame: pl.DataFrame, sd) -> str:
    return str(
        evaluate(
            label,
            boundary="as_of",
            pred_margin=pred,
            actual_margin=frame["margin"].to_numpy(),
            prob=norm_cdf(pred, sd),
            won=frame["home_won"].to_numpy(),
        )
    )


def main(
    seasons: list[int] | None = None,
    out_dir: str = "artifacts/higher_models",
    holdout: list[int] | None = None,
) -> int:
    from sportsdataverse.cfb.cfb_prediction_constants import get_constants

    seasons = seasons or list(range(2016, 2026))
    holdout = sorted(holdout or [])
    frame = build_game_frame(seasons)
    print(f"as-of frame: {frame.height} games, seasons {min(seasons)}-{max(seasons)}\n")

    rows, p_flat, p_curve, train_sd, train_sd_curve, scored = walk_forward_fit(frame)
    print("per-fold refit (train = all prior seasons):")
    print(f"  {'test':>6} {'slope':>8} {'hfa_pt':>8} {'resid_sd':>9}")
    for s, sl, h, sd in rows:
        print(f"  {s:>6} {sl:>8.2f} {h:>8.2f} {sd:>9.2f}")

    # Score the incumbent on EXACTLY the games the challengers were scored on.
    # The first `min_train` seasons are consumed as training and never scored,
    # so comparing against the incumbent's number over all seasons would be
    # comparing two different game sets.
    print("\n" + str(shipped_predictor(scored)))
    shipped = from_constants(get_constants("modern"))
    for label, p, sd in (
        (
            "shipped constants, serving formula",
            shipped.predict(scored),
            shipped.margin_sd,
        ),
        ("refit (flat slope)", p_flat, train_sd),
        ("refit (attenuation curve)", p_curve, train_sd_curve),
    ):
        print("\n" + _report(label, p, scored, sd))
        print(by_week(scored, p))

    final = fit_for_ship(frame, holdout)
    print(f"\nship fit on {final.seasons} ({final.n_train} games), holdout {holdout}")
    print(f"slope-by-games-played curve: {final.slope_by_games}")
    print(
        f"margin_sd flat {final.margin_sd:.4f}, curve (served) {final.margin_sd_curve:.4f}"
    )
    train = frame.filter(pl.col("season").is_in(final.seasons))
    # The rating spread these constants assume -- sdv-py's
    # cfb_game_predict._FITTED_ADJ_NET_SD, which warns when ratings drift.
    adj_net = np.concatenate(
        [train["rt_adj_net_home"].to_numpy(), train["rt_adj_net_away"].to_numpy()]
    )
    print(
        f"rt_adj_net sd over {adj_net.size} team-games: {np.std(adj_net, ddof=1):.4f}"
    )
    out_json = asdict(final) | {
        "holdout_seasons": holdout,
        "adj_net_sd": float(np.std(adj_net, ddof=1)),
        "sportsdataverse_sha": _sdv_sha(),
    }
    if holdout:
        ho = frame.filter(pl.col("season").is_in(holdout))
        for label, f in (
            ("HOLDOUT shipped constants", shipped),
            ("HOLDOUT ship fit", final),
        ):
            for curve in (False, True):
                p = f.predict(ho, use_curve=curve)
                tag = f"{label} ({'curve' if curve else 'flat'})"
                sd = f.margin_sd_curve if curve else f.margin_sd
                print("\n" + _report(tag, p, ho, sd))
                print(by_week(ho, p))
    # GOP's sdv.ts: the FLAT form on the summaries' net_adj_epa. Its own fit,
    # scored against the shipped slope applied to that rating (what sdv.ts runs).
    gop = fit_for_ship(frame, holdout, rating="net_adj_epa")
    out_json["gop_net_adj_epa"] = {
        k: getattr(gop, k)
        for k in ("net_points_scale", "hfa_points", "margin_sd", "n_train", "seasons")
    }
    print(
        f"\nGOP arm (flat, net_adj_epa): scale {gop.net_points_scale:.4f} "
        f"hfa {gop.hfa_points:.4f} margin_sd {gop.margin_sd:.4f}"
    )
    _r, g_flat, _c, g_sd, _cs, g_scored = walk_forward_fit(frame, rating="net_adj_epa")
    print("\n" + _report("GOP arm walk-forward (flat)", g_flat, g_scored, g_sd))
    if holdout:
        gop_now = from_constants(get_constants("modern"), rating="net_adj_epa")
        for label, f in (("HOLDOUT sdv.ts today", gop_now), ("HOLDOUT GOP arm", gop)):
            p = f.predict(ho, use_curve=False)
            print("\n" + _report(f"{label} (flat, net_adj_epa)", p, ho, f.margin_sd))
            print(by_week(ho, p))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pregame_fit.json").write_text(json.dumps(out_json, indent=2))
    print(f"\nwrote {out / 'pregame_fit.json'}")
    return 0
