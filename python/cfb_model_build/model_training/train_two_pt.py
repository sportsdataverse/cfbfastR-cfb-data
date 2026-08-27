"""Two-point-conversion-success model trainer (reproduces two_pt_model.ubj)."""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import two_pt_matrix


def train_two_pt(df: pl.DataFrame, nrounds: int = C.TWO_PT_NROUNDS) -> xgb.Booster:
    X, y, _ = two_pt_matrix(df)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(C.TWO_PT_PARAMS, dtrain, num_boost_round=nrounds)
