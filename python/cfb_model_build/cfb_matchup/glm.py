"""Logistic-glm trainers for the matchup pipeline's two fitted rates.

Both models are plain binomial-logit glms in the source (R ``glm(...,
family = "binomial")``); the port fits the same maximum-likelihood problem
with scikit-learn's unpenalised Newton-Cholesky solver (IRLS, as R's) and
writes a coefficient table in the exact JSON shape
``cfb_data_build.matchup_features.load_coefficients`` reads back, so the
applier and the trainer cannot disagree about the feature order.

* ``scoring_opp ~ half_secs_rem + game_secs_rem + start_yards_to_goal +
  pos_team_timeouts + def_pos_team_timeouts`` on the first-play-of-drive
  frame (``build_drive_frame``), drives starting beyond the 40 only;
* ``rush ~ pos_team_score + def_pos_team_score + score_diff + half + period +
  TimeSecsRem + down + distance + yards_to_goal + ep_before + wp_before +
  rz_play`` on rush/pass plays. ``score_diff`` is exactly
  ``pos_team_score - def_pos_team_score``, so the design is rank-deficient;
  R silently aliases it (NA coefficient, dropped at predict). The port drops it
  EXPLICITLY and records it as aliased -- same predictions, no singular solve.

Rows with a null in any term are dropped (R ``na.omit`` inside ``glm``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from cfb_data_build.matchup_features import (
    RUSH_EXPECT_FEATURES,
    SCORING_OPP_FEATURES,
    GlmCoefficients,
)

#: the source's rank-deficient term; R's predict drops it, so does the port
RUSH_EXPECT_ALIASED = ("score_diff",)


@dataclass(frozen=True)
class FitResult:
    coefficients: GlmCoefficients
    n_obs: int
    log_loss: float
    accuracy: float


def fit_logit(
    frame: pl.DataFrame,
    target: str,
    features: tuple[str, ...],
    *,
    aliased: tuple[str, ...] = (),
) -> FitResult:
    """Unpenalised logistic MLE on ``frame[features] -> frame[target]``."""
    active = [f for f in features if f not in aliased]
    data = frame.select([target, *active]).drop_nulls()
    x = data.select(active).to_numpy().astype(np.float64)
    y = data[target].to_numpy().astype(np.int64)
    model = LogisticRegression(
        penalty=None, solver="newton-cholesky", tol=1e-12, max_iter=500
    )
    model.fit(x, y)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-15, 1 - 1e-15)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    coef = GlmCoefficients(
        intercept=float(model.intercept_[0]),
        coefficients={f: float(c) for f, c in zip(active, model.coef_[0])},
        features=list(features),
        aliased=list(aliased),
        family="binomial",
        link="logit",
    )
    return FitResult(
        coef, int(len(y)), log_loss, float(np.mean((p >= 0.5) == (y == 1)))
    )


def fit_scoring_opp(drives: pl.DataFrame) -> FitResult:
    return fit_logit(drives, "scoring_opp", SCORING_OPP_FEATURES)


def fit_rush_expect(plays: pl.DataFrame) -> FitResult:
    return fit_logit(plays, "rush", RUSH_EXPECT_FEATURES, aliased=RUSH_EXPECT_ALIASED)


def rush_expect_training_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Rush/pass plays with the 12 terms and the target (``tools/run_pass.R``)."""
    return (
        pbp.filter((pl.col("rush") == 1) | (pl.col("pass") == 1))
        .select([*RUSH_EXPECT_FEATURES, "rush"])
        .drop_nulls()
    )


def coefficients_json(coef: GlmCoefficients, *, formula: str) -> dict:
    """The ``*_coef.json`` document, R-capture compatible (aliased terms -> null)."""
    table = {"(Intercept)": coef.intercept}
    for f in coef.features:
        table[f] = coef.coefficients.get(f)  # None for aliased
    return {
        "formula": formula,
        "family": coef.family,
        "link": coef.link,
        "n_coef": len(table),
        "n_aliased": len(coef.aliased),
        "coefficients": table,
    }


def write_fit(
    result: FitResult, out_dir: Path, stem: str, *, formula: str, seasons: list[int]
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_path = out_dir / f"{stem}_coef.json"
    meta_path = out_dir / f"{stem}_meta.json"
    coef_path.write_text(
        json.dumps(coefficients_json(result.coefficients, formula=formula), indent=2),
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(
            {
                "model": stem,
                "features": result.coefficients.features,
                "aliased": result.coefficients.aliased,
                "train_seasons": seasons,
                "n_obs": result.n_obs,
                "train_log_loss": result.log_loss,
                "train_accuracy": result.accuracy,
                "solver": "sklearn LogisticRegression(penalty=None, solver='newton-cholesky', tol=1e-12)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return coef_path, meta_path
