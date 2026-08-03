"""Metric sets for the four model styles, plus the trivial baselines.

The metric is part of a model's contract, not a reporting detail. An
*expectancy* model and a *predictive* model can share every feature and still
need different numbers to be judged honestly -- which is exactly how a
same-game fit came to be published as "spread MAE 3.23" against an
out-of-sample reality of 15.13.

``calibration_slope`` is promoted to a first-class metric here because it is
what caught the shipped ``net_points_scale`` being ~1.4-1.9x too large: MAE
alone barely moved (a badly over-dispersed model still beats nothing), while
the slope showed the predictions were stretched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def norm_cdf(m, sd: float):
    """Margin -> win probability under a normal residual assumption."""
    a = _arr(m)
    return 0.5 * (1.0 + np.vectorize(math.erf)(a / (sd * math.sqrt(2.0))))


# --------------------------------------------------------------------------
# margin / total (regression)
# --------------------------------------------------------------------------
def margin_metrics(pred, actual) -> dict[str, float]:
    p, a = _arr(pred), _arr(actual)
    # Fail loudly and specifically. numpy's own failure here is
    # "SVD did not converge in Linear Least Squares", which says nothing about
    # the actual cause (unrated teams -> null ratings -> NaN predictions).
    bad = ~(np.isfinite(p) & np.isfinite(a))
    if bad.any():
        raise ValueError(
            f"margin_metrics: {bad.sum()}/{len(a)} rows are NaN/inf "
            f"({np.sum(~np.isfinite(p))} in pred, {np.sum(~np.isfinite(a))} in actual). "
            "Build the frame with require_rating=True, or filter on `rated`."
        )
    err = p - a
    # Slope of actual ~ pred. 1.0 == correctly scaled; <1 == over-dispersed
    # (predictions stretched wider than reality), >1 == compressed.
    slope, intercept = np.polyfit(p, a, 1)
    return {
        "n": float(len(a)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "corr": float(np.corrcoef(p, a)[0, 1]),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "resid_sd": float(np.std(a - (slope * p + intercept))),
    }


# --------------------------------------------------------------------------
# win probability (classification)
# --------------------------------------------------------------------------
def wp_metrics(prob, won, *, bins: int = 10) -> dict[str, float]:
    p = np.clip(_arr(prob), 1e-9, 1 - 1e-9)
    y = _arr(won)
    # Max calibration error over equal-width probability bins: a model can post
    # a fine Brier while being badly wrong in one region.
    max_cal_err = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.sum() >= 20:
            max_cal_err = max(max_cal_err, abs(p[m].mean() - y[m].mean()))
    return {
        "n": float(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
        "max_cal_err": float(max_cal_err),
    }


# --------------------------------------------------------------------------
# baselines -- the bar any model must clear to have earned its complexity
# --------------------------------------------------------------------------
def baselines(actual_margin, won) -> dict[str, dict[str, float]]:
    a, y = _arr(actual_margin), _arr(won)
    const = float(np.mean(a))
    return {
        "margin_zero": {"mae": float(np.mean(np.abs(a))), "brier": float(np.mean((0.5 - y) ** 2))},
        "constant_hfa": {"mae": float(np.mean(np.abs(a - const))), "constant": const},
        "always_home": {"accuracy": float(np.mean(y))},
    }


@dataclass
class Report:
    """One model's out-of-sample scorecard, with its boundary declared."""

    name: str
    boundary: str  # "as_of" | "same_game" | "full_season" -- see data.py
    margin: dict[str, float] = field(default_factory=dict)
    wp: dict[str, float] = field(default_factory=dict)
    total: dict[str, float] = field(default_factory=dict)
    base: dict[str, dict[str, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        L = [f"=== {self.name}  [boundary: {self.boundary}] ==="]
        if self.margin:
            m = self.margin
            L.append(
                f"  margin  n={m['n']:.0f} MAE={m['mae']:.2f} bias={m['bias']:+.2f} "
                f"corr={m['corr']:.3f} slope={m['calibration_slope']:.2f} "
                f"resid_sd={m['resid_sd']:.2f}"
            )
        if self.wp:
            w = self.wp
            L.append(
                f"  winprob brier={w['brier']:.4f} logloss={w['logloss']:.4f} "
                f"acc={w['accuracy']:.3f} max_cal_err={w['max_cal_err']:.3f}"
            )
        if self.total:
            t = self.total
            L.append(f"  total   MAE={t['mae']:.2f} bias={t['bias']:+.2f}")
        if self.base:
            b = self.base
            L.append(
                f"  vs base margin_zero MAE={b['margin_zero']['mae']:.2f} | "
                f"constant({b['constant_hfa']['constant']:+.2f}) "
                f"MAE={b['constant_hfa']['mae']:.2f} | "
                f"always_home acc={b['always_home']['accuracy']:.3f}"
            )
            if self.margin:
                gain = b["constant_hfa"]["mae"] - self.margin["mae"]
                L.append(f"  --> beats the constant baseline by {gain:+.2f} MAE")
        return "\n".join(L)


def evaluate(
    name: str,
    *,
    boundary: str,
    pred_margin=None,
    actual_margin=None,
    prob=None,
    won=None,
    pred_total=None,
    actual_total=None,
) -> Report:
    r = Report(name=name, boundary=boundary)
    if pred_margin is not None and actual_margin is not None:
        r.margin = margin_metrics(pred_margin, actual_margin)
    if prob is not None and won is not None:
        r.wp = wp_metrics(prob, won)
    if pred_total is not None and actual_total is not None:
        r.total = margin_metrics(pred_total, actual_total)
    if actual_margin is not None and won is not None:
        r.base = baselines(actual_margin, won)
    return r
