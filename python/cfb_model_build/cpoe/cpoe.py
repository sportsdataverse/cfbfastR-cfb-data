"""Compute CPOE = Completion - Predicted Completion Probability.

CPOE > 0  → QB completed passes at a higher rate than expected.
CPOE < 0  → QB completed passes at a lower rate than expected.

This is purely arithmetic once we have the CP model predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import AIR_YARDS_FEATURE_COLS, FEATURE_COLS, TARGET_COL


def compute_cpoe(
    df: pd.DataFrame,
    booster: xgb.Booster,
    *,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Add ``cp_pred`` and ``cpoe`` columns to a pass-play DataFrame.

    Args:
        df: DataFrame with FEATURE_COLS + TARGET_COL (from
            ``extract_pass_features``).
        booster: Trained XGBoost Booster (from ``train_cp_model``).
        features: Feature columns to score on. Defaults to the booster's own
            ``feature_names`` when it records them, else FEATURE_COLS -- so a
            model and its feature list cannot drift apart at scoring time.

    Returns:
        Copy of ``df`` with two additional columns:
            ``cp_pred`` — predicted completion probability [0, 1].
            ``cpoe``    — actual completion minus ``cp_pred``.
        Empty DataFrame (zero rows) if ``df`` is empty.
    """
    if df.empty:
        return pd.DataFrame()

    if features is not None:
        wanted = list(features)
    elif getattr(booster, "feature_names", None):
        wanted = list(booster.feature_names)
    else:
        wanted = list(FEATURE_COLS)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(
            f"cannot score: frame is missing {missing}. The model was fitted on "
            f"{wanted}; scoring on a subset would silently change its meaning."
        )
    dmat = xgb.DMatrix(df[wanted])
    preds = booster.predict(dmat)

    out = df.copy()
    out["cp_pred"] = preds.astype(float)

    if TARGET_COL in out.columns:
        out["cpoe"] = out[TARGET_COL].astype(float) - out["cp_pred"]
    else:
        out["cpoe"] = np.nan

    return out.reset_index(drop=True)


def compute_cpoe_hybrid(
    df: pd.DataFrame,
    base_booster: xgb.Booster,
    air_booster: xgb.Booster | None = None,
) -> pd.DataFrame:
    """Score CPOE with the air-yards model where it applies, else the base model.

    ESPN only supplies air yards for some plays (38.9% of 2025, 90.2% of 2026 to
    date, ~none before 2025), and the two models are not interchangeable: on
    air-yards rows the game-state model is close to a coin flip (AUC 0.567)
    while the air-yards model reaches AUC 0.764. Neither can replace the other
    -- the air-yards model has nothing to score a 2015 play with, and the base
    model wastes the best available signal on a 2026 play.

    So each play is scored by the best model that can actually see it, and the
    ``cp_model`` column records which one did. That column is not decoration:
    CPOE from the two models is NOT on a comparable scale, so any leaderboard
    or season aggregate must either group by it or restrict to one of them.
    Averaging across both silently mixes two different quantities.

    Args:
        df: Pass-play frame from ``extract_pass_features``.
        base_booster: The game-state model (2004+).
        air_booster: The air-yards model. When None, every row is scored by
            ``base_booster`` and ``cp_model`` is uniformly ``"game_state"``.

    Returns:
        Copy of ``df`` with ``cp_pred``, ``cpoe`` and ``cp_model`` columns.
        Empty DataFrame if ``df`` is empty.
    """
    if df.empty:
        return pd.DataFrame()

    can_air = (
        air_booster is not None
        and "air_yards" in df.columns
        and df["air_yards"].notna().any()
        and all(c in df.columns for c in AIR_YARDS_FEATURE_COLS)
    )
    if not can_air:
        out = compute_cpoe(df, base_booster, features=list(FEATURE_COLS))
        out["cp_model"] = "game_state"
        return out

    air_rows = df["air_yards"].notna()
    # Scoring splits the frame in two, so the row order has to be put back
    # explicitly. Returning air-yards rows first would silently reorder the
    # caller's frame -- anything aligning by position (a .values assignment back
    # onto the source pbp, say) would then attach every CPOE to the wrong play,
    # with no error and a plausible-looking column.
    order = "_cpoe_row_order"
    src = df.copy()
    src[order] = np.arange(len(src))

    parts = []
    for mask, booster, feats, label in (
        (air_rows, air_booster, AIR_YARDS_FEATURE_COLS, "air_yards"),
        (~air_rows, base_booster, FEATURE_COLS, "game_state"),
    ):
        if not mask.any():
            continue
        part = compute_cpoe(
            src[mask.to_numpy()].reset_index(drop=True),
            booster,
            features=list(feats),
        )
        part["cp_model"] = label
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    return (
        out.sort_values(order)
        .drop(columns=[order])
        .reset_index(drop=True)
    )
