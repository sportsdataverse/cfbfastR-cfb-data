"""Leave-One-Season-Out cross-validation for the CFB CP model.

For each held-out season:
  1. Train on all other seasons.
  2. Predict CP probabilities on the held-out season.
  3. Record log-loss, Brier score, and play count.

Input DataFrame must have a ``season`` column plus FEATURE_COLS + TARGET_COL.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss

from .constants import FEATURE_COLS, TARGET_COL, XGB_NROUNDS, XGB_PARAMS
from .train_cp import train_cp_model


def run_loso_cv(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    return_preds: bool = False,
    nrounds: int = XGB_NROUNDS,
    params: dict | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Run LOSO cross-validation.

    Args:
        df: DataFrame with ``season_col``, FEATURE_COLS, and TARGET_COL.
        season_col: Column that identifies the season (int year).
        return_preds: If True, each fold record includes a ``cp_pred``
            array of length ``n_plays``.
        nrounds: Boosting rounds per fold.
        params: XGBoost params dict (defaults to constants.XGB_PARAMS).
        features: Feature columns to fit on (defaults to FEATURE_COLS).

    Returns:
        Dict with keys:
            ``folds``   — list of per-fold result dicts.
            ``summary`` — dict with ``mean_log_loss``, ``mean_brier_score``.

    Raises:
        ValueError: If fewer than 2 distinct seasons are present.
    """
    feats = list(features) if features is not None else list(FEATURE_COLS)
    seasons = sorted(df[season_col].unique())
    if len(seasons) < 2:
        raise ValueError(
            f"run_loso_cv requires at least 2 distinct seasons; got {seasons}."
        )

    folds: list[dict[str, Any]] = []

    for held_out in seasons:
        train_df = df[df[season_col] != held_out]
        test_df = df[df[season_col] == held_out]

        X_train = train_df[feats]
        y_train = train_df[TARGET_COL]
        X_test = test_df[feats]
        y_test = test_df[TARGET_COL].to_numpy()

        booster = train_cp_model(
            X_train, y_train, nrounds=nrounds, params=params, features=feats
        )
        preds = booster.predict(xgb.DMatrix(X_test))

        # clip for numerical safety
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)

        fold: dict[str, Any] = {
            "season": int(held_out),
            "n_plays": len(y_test),
            "log_loss": float(log_loss(y_test, preds_clipped)),
            "brier_score": float(brier_score_loss(y_test, preds_clipped)),
        }
        if return_preds:
            fold["cp_pred"] = preds.tolist()

        folds.append(fold)

    mean_log_loss = float(np.mean([f["log_loss"] for f in folds]))
    mean_brier = float(np.mean([f["brier_score"] for f in folds]))

    return {
        "folds": folds,
        "summary": {
            "mean_log_loss": mean_log_loss,
            "mean_brier_score": mean_brier,
            "n_seasons": len(seasons),
        },
    }


def run_grouped_cv(
    df: pd.DataFrame,
    *,
    group_col: str = "game_id",
    n_splits: int = 5,
    nrounds: int = XGB_NROUNDS,
    params: dict | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """K-fold CV grouped by game, for variants with too few seasons for LOSO.

    The air-yards model trains on 2025+ only (ESPN did not emit catch/target
    spots at scale before then), so LOSO would degenerate to two folds, one of
    which is a partial season. Grouping by game instead keeps the fold count
    useful while still preventing the leak that matters here: passes within one
    game share a QB, an offense, an opponent and the weather, so a row-level
    split would put near-duplicate plays on both sides and flatter the model.

    Args:
        df: DataFrame with ``group_col``, the feature columns, and TARGET_COL.
        group_col: Column identifying the group (default: ``game_id``).
        n_splits: Number of folds.
        nrounds: Boosting rounds per fold.
        params: XGBoost params dict (defaults to constants.XGB_PARAMS).
        features: Feature columns to fit on (defaults to FEATURE_COLS).

    Returns:
        Same shape as :func:`run_loso_cv`: ``folds`` + ``summary``.

    Raises:
        ValueError: If ``group_col`` is absent or there are fewer groups than
            folds.
    """
    from sklearn.model_selection import GroupKFold

    feats = list(features) if features is not None else list(FEATURE_COLS)
    if group_col not in df.columns:
        raise ValueError(f"run_grouped_cv needs a {group_col!r} column; got {list(df.columns)}")
    groups = df[group_col].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"need >= {n_splits} distinct {group_col} values; got {n_groups}")

    X, y = df[feats], df[TARGET_COL].to_numpy()
    folds: list[dict[str, Any]] = []
    for i, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        booster = train_cp_model(
            X.iloc[tr], y[tr], nrounds=nrounds, params=params, features=feats
        )
        preds = booster.predict(xgb.DMatrix(X.iloc[te]))
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        folds.append(
            {
                "fold": i,
                "n_plays": int(len(te)),
                "log_loss": float(log_loss(y[te], preds_clipped)),
                "brier_score": float(brier_score_loss(y[te], preds_clipped)),
            }
        )

    return {
        "folds": folds,
        "summary": {
            "mean_log_loss": float(np.mean([f["log_loss"] for f in folds])),
            "mean_brier_score": float(np.mean([f["brier_score"] for f in folds])),
            "n_groups": int(n_groups),
            "cv": f"GroupKFold({n_splits}) by {group_col}",
        },
    }
