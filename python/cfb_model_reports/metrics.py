"""Metric runners for model reports — provenance extraction, classification metrics, importance."""
from __future__ import annotations

from pathlib import Path


def provenance_from_card(card: dict) -> dict:
    """Extract provenance fields from a model card, tolerating missing keys.

    Args:
        card: Model card dict with optional keys (features, hyperparameters,
            training_seasons, trained_date, xgboost_version).

    Returns:
        Dict with keys (features, hyperparameters, training_seasons, trained_date,
        xgboost_version), using safe defaults for missing keys.
    """
    return {
        "features": card.get("features") or [],
        "hyperparameters": card.get("hyperparameters") or {},
        "training_seasons": card.get("training_seasons"),
        "trained_date": card.get("trained_date"),
        "xgboost_version": card.get("xgboost_version"),
    }


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Compute classification metrics: log-loss and Brier score (binary only).

    Args:
        y_true: True labels (array-like, 0/1).
        y_pred: Predicted probabilities (array-like, shape same as y_true).

    Returns:
        Dict with keys: n (sample count), log_loss (sklearn), brier_score (binary only).
    """
    import numpy as np
    from sklearn.metrics import log_loss

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    out = {"n": int(yt.shape[0]), "log_loss": float(log_loss(yt, yp))}
    if yp.ndim == 1:  # binary
        out["brier_score"] = float(np.mean((yp - yt) ** 2))
    return out


def xgb_importance(model_path, top_n: int = 15) -> dict:
    """Extract XGBoost model feature importance (gain).

    Args:
        model_path: Path to XGBoost model file (.ubj, .json, etc.).
        top_n: Number of top features to return (default: 15).

    Returns:
        Dict mapping feature names to importance scores (rounded to 4 decimals).
    """
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(model_path))
    score = b.get_score(importance_type="gain")
    top = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {k: round(float(v), 4) for k, v in top}


def rb_eval_metrics(loso_parquet) -> dict:
    """Compute RB evaluation metrics from LOSO cross-validation parquet.

    Reads a parquet file containing exp_rb_epa and target columns,
    bins predictions, and computes weighted R² and calibration error.

    Args:
        loso_parquet: Path to xrepa_loso.parquet (str or Path).

    Returns:
        Dict with keys: weighted_r2, weighted_cal_error (rounded to 4 decimals),
        and n (sample count).
    """
    import polars as pl
    from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2

    cv = pl.read_parquet(loso_parquet)
    tbl = calibration_table(cv)
    return {
        "weighted_r2": round(float(weighted_r2(tbl)), 4),
        "weighted_cal_error": round(float(weighted_cal_error(tbl)), 4),
        "n": int(cv.height),
    }
