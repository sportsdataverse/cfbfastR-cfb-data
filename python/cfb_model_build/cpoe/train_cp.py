"""XGBoost CP model training, save, and load helpers (Track 5, Approach A).

The model mirrors the nflfastR `cpoe_model.R` hyper-parameters exactly
(binary:logistic, eta=0.025, max_depth=4, etc.).  See constants.XGB_PARAMS.
"""

from __future__ import annotations

import pathlib
from typing import Union

import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import FEATURE_COLS, TARGET_COL, XGB_NROUNDS, XGB_PARAMS

ArrayLike = Union[pd.DataFrame, np.ndarray]


def train_cp_model(
    X: ArrayLike,
    y: ArrayLike,
    *,
    nrounds: int = XGB_NROUNDS,
    params: dict | None = None,
    verbose_eval: bool = False,
    features: list[str] | None = None,
) -> xgb.Booster:
    """Train the CP model and return the fitted XGBoost Booster.

    Args:
        X: Feature matrix (n_plays × len(FEATURE_COLS)).  DataFrame or ndarray.
        y: Binary completion labels (0/1).  Series or 1-D ndarray.
        nrounds: Number of boosting rounds (default: XGB_NROUNDS = 560).
        params: XGBoost parameter dict.  Defaults to constants.XGB_PARAMS.
        verbose_eval: Print eval log every round if True.
        features: Feature names, used only when ``X`` is a bare ndarray with no
            column names of its own.  Defaults to the 8 game-state
            FEATURE_COLS; pass AIR_YARDS_FEATURE_COLS to fit the air-yards
            variant.  Ignored when ``X`` is a DataFrame, which carries its own
            names -- so the ndarray path cannot silently mislabel a
            differently-shaped matrix.

    Returns:
        Fitted ``xgb.Booster``.
    """
    _params = dict(XGB_PARAMS if params is None else params)
    names = list(features) if features is not None else list(FEATURE_COLS)
    if isinstance(X, np.ndarray) and X.shape[1] != len(names):
        raise ValueError(
            f"X has {X.shape[1]} columns but {len(names)} feature names were given "
            f"({names}). Pass features= matching the matrix."
        )
    dmat = xgb.DMatrix(
        X, label=y, feature_names=names if isinstance(X, np.ndarray) else None
    )
    booster = xgb.train(
        _params,
        dmat,
        num_boost_round=nrounds,
        verbose_eval=verbose_eval,
    )
    return booster


def save_cp_model(
    booster: xgb.Booster,
    path: pathlib.Path | str,
    *,
    features: list[str] | None = None,
    model_type: str = "cpoe",
) -> None:
    """Save a trained Booster to an UBJ file (+ a model_card.json sidecar).

    Args:
        booster: Fitted XGBoost Booster.
        path: Destination path (conventionally ``cfb_cp_model.ubj``).
        features: Feature names to record on the card. Defaults to the
            game-state FEATURE_COLS; pass AIR_YARDS_FEATURE_COLS for the
            air-yards variant.
        model_type: Card ``model_type``, which is what distinguishes the two CP
            variants in the published artifacts.

    Raises:
        RuntimeError: If the model card could not be written. The card is NOT
            cosmetic: ``cfb_model_reports.discovery.discover_models`` enumerates
            artifacts by globbing ``*.json`` cards and only then looks for the
            sibling ``.ubj``, so a model whose card is missing is silently
            skipped by the publisher. Swallowing this would train a model, save
            it, report success, and never ship it.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(p))
    try:
        from cfb_model_build.model_training.model_card import write_xgb_model_card

        write_xgb_model_card(
            p,
            model_type=model_type,
            label=TARGET_COL,
            model=booster,
            features=list(features) if features is not None else list(FEATURE_COLS),
            hyperparams=dict(XGB_PARAMS),
        )
    except Exception as e:  # noqa: BLE001 - re-raised; see the docstring
        raise RuntimeError(
            f"model saved to {p} but its model card could not be written: {e}. "
            "The publisher discovers models by card, so this model would never "
            "be published."
        ) from e


def load_cp_model(path: pathlib.Path | str) -> xgb.Booster:
    """Load a saved CP model from disk.

    Args:
        path: Path to a UBJ (or JSON/bin) XGBoost model file.

    Returns:
        Loaded ``xgb.Booster``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CP model not found: {p}")
    booster = xgb.Booster()
    booster.load_model(str(p))
    return booster
