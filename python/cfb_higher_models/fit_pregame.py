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
metrics are out-of-sample.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .backtest import season_splits, shipped_predictor
from .data import build_game_frame
from .metrics import evaluate, norm_cdf

# Rating differential the closed form runs on.
_DIFF = ("rt_adj_net_home", "rt_adj_net_away")
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

    def predict(self, frame: pl.DataFrame, *, use_curve: bool = True) -> np.ndarray:
        diff = frame[_DIFF[0]].to_numpy() - frame[_DIFF[1]].to_numpy()
        hfa = np.where(frame["neutral_site"].to_numpy(), 0.0, self.hfa_points)
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


def fit(frame: pl.DataFrame) -> PregameFit:
    """Fit every constant on ``frame`` (caller guarantees it is training-only)."""
    diff = frame[_DIFF[0]].to_numpy() - frame[_DIFF[1]].to_numpy()
    margin = frame["margin"].to_numpy()
    neutral = frame["neutral_site"].to_numpy()

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

    return PregameFit(
        net_points_scale=slope,
        hfa_points=hfa,
        margin_sd=resid_sd,
        total_intercept=float(ct[0]),
        total_pace_scale=float(ct[1]),
        total_epa_scale=float(ct[2]),
        slope_by_games=curve,
        n_train=int(frame.height),
        seasons=sorted(int(s) for s in frame["season"].unique().to_list()),
    )


def walk_forward_fit(frame: pl.DataFrame, *, min_train: int = 2):
    """Out-of-sample scoring of the refit, plus the fit on all data to ship."""
    rows, preds_flat, preds_curve, tests = [], [], [], []
    for train_seasons, test_season in season_splits(frame, min_train=min_train):
        tr = frame.filter(pl.col("season").is_in(train_seasons))
        te = frame.filter(pl.col("season") == test_season)
        if not tr.height or not te.height:
            continue
        f = fit(tr)
        preds_flat.append(f.predict(te, use_curve=False))
        preds_curve.append(f.predict(te, use_curve=True))
        tests.append(te)
        rows.append((test_season, f.net_points_scale, f.hfa_points, f.margin_sd))
    # Return the scored rows themselves, not just targets: the incumbent has to
    # be scored on EXACTLY these games or the comparison is rigged. (The first
    # `min_train` seasons are consumed as training and never scored.)
    return (
        rows,
        np.concatenate(preds_flat),
        np.concatenate(preds_curve),
        pl.concat(tests),
    )


def main(
    seasons: list[int] | None = None, out_dir: str = "artifacts/higher_models"
) -> int:
    seasons = seasons or list(range(2016, 2026))
    frame = build_game_frame(seasons)
    print(f"as-of frame: {frame.height} games, seasons {min(seasons)}-{max(seasons)}\n")

    rows, p_flat, p_curve, scored = walk_forward_fit(frame)
    print("per-fold refit (train = all prior seasons):")
    print(f"  {'test':>6} {'slope':>8} {'hfa_pt':>8} {'resid_sd':>9}")
    for s, sl, h, sd in rows:
        print(f"  {s:>6} {sl:>8.2f} {h:>8.2f} {sd:>9.2f}")

    # Score the incumbent on EXACTLY the games the challengers were scored on.
    # The first `min_train` seasons are consumed as training and never scored,
    # so comparing against the incumbent's number over all seasons would be
    # comparing two different game sets.
    act = scored["margin"].to_numpy()
    won = scored["home_won"].to_numpy()
    print("\n" + str(shipped_predictor(scored)))

    for label, p in (
        ("refit (flat slope)", p_flat),
        ("refit (attenuation curve)", p_curve),
    ):
        sd = float(np.std(act - p))
        print(
            "\n"
            + str(
                evaluate(
                    label,
                    boundary="as_of",
                    pred_margin=p,
                    actual_margin=act,
                    prob=norm_cdf(p, sd),
                    won=won,
                )
            )
        )

    final = fit(frame)
    print(f"\nslope-by-games-played curve: {final.slope_by_games}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pregame_fit.json").write_text(json.dumps(asdict(final), indent=2))
    print(f"\nwrote {out / 'pregame_fit.json'}")
    return 0
