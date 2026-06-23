"""Training pipeline: outlier filter + XGBRegressor fit.

OQ-7 resolution: mu=0.0 (point-differential is symmetric), std = std of full
training-set predictions.  The notebook used test-split statistics which is
non-reproducible without a fixed seed.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats

from .constants import OUTLIER_Z_5FR, OUTLIER_Z_PTS, XGB_N_ESTIMATORS, XGB_SEED, WP_MU


def filter_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows where 5FRDiff or PtsDiff exceeds the z-score thresholds."""
    z_5fr = np.abs(stats.zscore(df["5FRDiff"].to_numpy())) < OUTLIER_Z_5FR
    z_pts = np.abs(stats.zscore(df["PtsDiff"].to_numpy())) < OUTLIER_Z_PTS
    mask = z_5fr & z_pts
    return df.filter(pl.Series(mask))


def train_pgwp_model(
    df: pl.DataFrame,
) -> tuple[xgb.XGBRegressor, float, float]:
    """Train a 10-tree XGBRegressor on 5FRDiff → PtsDiff.

    Returns:
        model: fitted XGBRegressor
        mu: 0.0 (OQ-7: symmetric by construction)
        std: std of full training-set predictions
    """
    # Model boundary stays numpy — XGBoost consumes ndarrays.
    X = df.select("5FRDiff").to_numpy()
    y = df["PtsDiff"].to_numpy()

    model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        seed=XGB_SEED,
        verbosity=0,
    )
    model.fit(X, y)

    preds = model.predict(X)
    mu = WP_MU  # 0.0 — per OQ-7 resolution
    std = float(np.std(preds))

    return model, mu, std


def save_pgwp_model(
    model: xgb.XGBRegressor,
    std: float,
    path: str,
    season_range: tuple[int, int] | None = None,
) -> None:
    """Save model as UBJ + a unified ``model_card.json`` sidecar.

    The card uses the shared ``write_xgb_model_card`` helper (Tracks 1-5 parity)
    and merges the pregame-specific ``mu`` / ``std`` normalization params in at the
    top level via ``extra=``.  Those two keys are load-bearing — both
    ``load_pgwp_model`` and the CLI read them back to reconstruct the
    5FRDiff -> WP Gaussian transform — so the write is intentionally NOT
    best-effort here (a failure must surface, unlike the audit-only cards).
    """
    from pathlib import Path

    from model_training.model_card import write_xgb_model_card

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))

    write_xgb_model_card(
        p,
        model_type="pregame_wp",
        label="PtsDiff",
        features=["5FRDiff"],
        model=model,
        seasons=season_range,
        extra={
            "mu": WP_MU,
            "std": std,
            "n_estimators": XGB_N_ESTIMATORS,
            "note": "pgwp_model — NOT bundled into sdv-py. Track 4 analytic artifact.",
        },
    )


def load_pgwp_model(path: str) -> tuple[xgb.XGBRegressor, float, float]:
    """Load model + sidecar metadata (mu, std)."""
    import json
    from pathlib import Path

    p = Path(path)
    model = xgb.XGBRegressor()
    model.load_model(str(p))
    card = json.loads(p.with_suffix(".json").read_text())
    return model, float(card["mu"]), float(card["std"])
