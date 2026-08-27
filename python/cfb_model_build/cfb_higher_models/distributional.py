"""Predict a DISTRIBUTION over margin, and score it with a proper rule.

WHY THIS IS THE UNBLOCKER
-------------------------
The measured minimum detectable effect for this design is 0.067 MAE, and more
folds will not improve it -- week-level clustering was tested and is strictly
worse (SE 0.0505 vs 0.0237). The bottleneck is not sample size, it is that MAE
on a point estimate discards most of what each game reveals.

A proper scoring rule over the full predictive distribution uses all of it.
CRPS generalises MAE: for a point mass it REDUCES to absolute error, so it is
directly comparable, but for a distribution it also rewards stating the right
uncertainty. Strictly proper means a model cannot improve its score by
misrepresenting its own confidence.

This is also the better product. Season simulations need calibrated
uncertainty, not a point estimate with a hand-fitted global sigma -- and the
sim currently samples Normal(margin, margin_sd) with a single constant sd for
every game, which is exactly the assumption tested here.

CRPS for a Gaussian has a closed form, so none of this needs sampling:

    CRPS(N(mu, s), y) = s * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
    z = (y - mu)/s
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


def crps_gaussian(y, mu, sigma) -> np.ndarray:
    """Per-observation CRPS for a Gaussian forecast. Lower is better.

    Reduces to |y - mu| as sigma -> 0, so CRPS and MAE are on the same scale
    and a CRPS improvement is interpretable in points of margin.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    s = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    z = (y - mu) / s
    return s * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )


def log_score_gaussian(y, mu, sigma) -> np.ndarray:
    """Negative log predictive density. Lower is better.

    Harsher than CRPS on overconfident tails -- a model that assigns near-zero
    density to a 40-point upset is punished severely. Reported alongside CRPS
    because CFB margins have heavy tails and the two disagree exactly where
    that matters.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    s = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    return -stats.norm.logpdf(y, loc=mu, scale=s)


def pit_values(y, mu, sigma) -> np.ndarray:
    """Probability integral transform: F(y) under the forecast distribution.

    If the forecasts are calibrated these are Uniform(0,1). Deviations name the
    failure precisely: a U-shaped histogram means the model is OVERCONFIDENT
    (too many observations in the tails), a hump means UNDERCONFIDENT, and a
    shifted mean means biased. MAE cannot distinguish any of these.
    """
    s = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    return stats.norm.cdf(
        (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / s
    )


def pit_report(y, mu, sigma, *, bins: int = 10) -> dict[str, float]:
    """Calibration summary from the PIT histogram."""
    p = pit_values(y, mu, sigma)
    counts, _ = np.histogram(p, bins=bins, range=(0, 1))
    expected = len(p) / bins
    # chi-square against uniform; and the tail mass, which is the overconfidence tell
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    tail = float(((p < 0.05) | (p > 0.95)).mean())
    return {
        "pit_chi2": chi2,
        "pit_p": float(1 - stats.chi2.cdf(chi2, df=bins - 1)),
        "tail_mass": tail,  # calibrated == 0.10
        "tail_excess": tail - 0.10,
        "pit_mean": float(p.mean()),  # calibrated == 0.50
    }


def oof_residuals(
    train: pl.DataFrame, feats: list[str], head, *, n_folds: int = 4, seed: int = 0
) -> np.ndarray:
    """Out-of-fold residuals for the training rows. REQUIRED for a sigma model.

    The first version of this module fit sigma on IN-SAMPLE residuals -- the
    mean model predicting its own training data, which a GBM fits far too
    well. Measured consequence: learned sigma averaged 8.97 against a true
    residual sd of 16.40 (45% too small), forecasts put 36.4% of outcomes in
    their own 5% tails (PIT p = 0.000), and both CRPS and log score came out
    WORSE than a plain constant.

    A variance model is only as honest as the residuals it is shown, and
    in-sample residuals of a flexible learner are not residuals -- they are a
    measure of how much the model memorised.
    """
    from sklearn.model_selection import KFold

    y = train["margin"].to_numpy().astype(float)
    out = np.empty(len(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(np.arange(len(y))):
        out[te_idx] = np.asarray(
            head(train[tr_idx.tolist()], train[te_idx.tolist()], feats=feats),
            dtype=float,
        )
    return y - out


def fit_sigma_model(train: pl.DataFrame, feats: list[str], resid: np.ndarray, **kw):
    """Model the SPREAD of the outcome, not just its centre.

    A single global sigma assumes every game is equally predictable. Whether
    that is false ENOUGH to matter is an empirical question -- and on this data
    the answer turned out to be no, once the fit was done honestly.

    ``resid`` MUST be out-of-fold (see :func:`oof_residuals`). Passing
    in-sample residuals silently produces a confidently wrong model.

    Fitted on log absolute residual so sigma cannot go negative. Note the
    exponentiation in :func:`predict_sigma` is bias-corrected: E[log|e|] is not
    log E[|e|] (Jensen), so a naive exp(mean) underestimates -- a second
    downward bias that compounded the first in the original version.
    """
    import xgboost as xgb

    from .train_game import _xy

    X, _ = _xy(train, feats)
    target = np.log(np.clip(np.abs(resid), 1.0, None))
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 3,  # deliberately shallow: variance is a smooth function
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 50,
        "reg_lambda": 5.0,
        "nthread": 4,
        **kw,
    }
    return xgb.train(params, xgb.DMatrix(X, label=target), num_boost_round=200)


#: E[log|Z|] for standard normal Z. log|Z| = (1/2)log(Z^2), Z^2 ~ chi^2_1, so
#: E[log|Z|] = (1/2)(psi(1/2) + log 2) = -(gamma + log 2)/2 ~ -0.6352.
_E_LOG_ABS_STDNORM = -(np.euler_gamma + np.log(2.0)) / 2.0


def log_abs_to_sigma(log_abs):
    """E[log|e|] -> sigma, for e ~ N(0, sigma^2). ONE conversion, not two.

    THE SINGLE DEFINITION. `check_sigma_recovery` calls this function rather
    than restating the arithmetic -- the first version of that check carried
    its own copy, so `replace_all` updated the implementation and left the test
    validating the old formula. A self-check that reimplements the logic tests
    the reimplementation, and will happily agree with itself while both are
    wrong.

    Four wrong versions preceded this line, each plausible, each shipping a
    confidently wrong uncertainty:
      1. no correction + IN-SAMPLE residuals -> sigma 8.97 vs true 16.40;
         36.4% of outcomes in their own 5% tails
      2. exp(+(gamma+log2)/2)/sqrt(2/pi) = 2.365 -- right Jensen factor, but
         ALSO converting E|e|->sigma, double-counting -> 22.11, 3.6% tail mass
      3. exp(-(log2-gamma)/2) = 0.944 -- wrong sign and identity; recovered
         exactly half
      4. (3) fixed in the implementation but not in its own duplicated test
    """
    return np.exp(np.asarray(log_abs, dtype=float) - _E_LOG_ABS_STDNORM)


def predict_sigma(model, frame: pl.DataFrame, feats: list[str]) -> np.ndarray:
    import xgboost as xgb

    from .train_game import _xy

    X, _ = _xy(frame, feats)
    return log_abs_to_sigma(model.predict(xgb.DMatrix(X)))


def check_sigma_recovery(
    sigma_true: float = 16.0, n: int = 200_000, seed: int = 0
) -> dict:
    """Round-trip the sigma conversion on synthetic data with a KNOWN sigma.

    Draw e ~ N(0, sigma^2), take the mean of log|e| (exactly what the model
    learns to predict), and confirm the conversion returns sigma. No football,
    no model -- just the arithmetic that four attempts got wrong.

    Calls :func:`log_abs_to_sigma` rather than restating the formula. The
    previous version duplicated it, so a fix to the implementation left this
    check silently validating the old constant.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma_true, size=n)
    mean_log_abs = float(np.mean(np.log(np.abs(e))))
    recovered = float(log_abs_to_sigma(mean_log_abs))
    return {
        "sigma_true": sigma_true,
        "sigma_recovered": recovered,
        "rel_error": (recovered - sigma_true) / sigma_true,
    }


def score_frame(y, mu, sigma, *, label: str) -> dict[str, float]:
    """Every metric at once, so point accuracy and calibration are read together."""
    y = np.asarray(y, dtype=float)
    mae = float(np.mean(np.abs(np.asarray(mu, dtype=float) - y)))
    out = {
        "label": label,
        "mae": mae,
        "crps": float(crps_gaussian(y, mu, sigma).mean()),
        "log_score": float(log_score_gaussian(y, mu, sigma).mean()),
        "mean_sigma": float(np.mean(sigma)),
    }
    out.update(pit_report(y, mu, sigma))
    return out


def format_scores(rows: list[dict]) -> str:
    hdr = f"{'model':<26}{'MAE':>8}{'CRPS':>9}{'logS':>8}{'sigma':>8}{'tail%':>8}{'PIT p':>8}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['label']:<26}{r['mae']:>8.3f}{r['crps']:>9.4f}{r['log_score']:>8.3f}"
            f"{r['mean_sigma']:>8.2f}{r['tail_mass'] * 100:>8.1f}{r['pit_p']:>8.3f}"
        )
    lines.append("  tail% calibrated = 10.0   PIT p > 0.05 == cannot reject uniform")
    return "\n".join(lines)
