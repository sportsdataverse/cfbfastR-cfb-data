"""FG make-probability model trainer (reproduces fg_model.ubj)."""

from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import fg_matrix


def train_fg(df: pl.DataFrame, nrounds: int = C.FG_NROUNDS) -> xgb.Booster:
    # era_onehot=True: the shipped fg_model is era-aware (yards_to_goal + era0..era3).
    X, y, _ = fg_matrix(df, era_onehot=True)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(C.FG_PARAMS, dtrain, num_boost_round=nrounds)
