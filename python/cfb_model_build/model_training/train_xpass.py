"""xPass (pass-vs-rush tendency) model trainer (reproduces xpass_model.ubj)."""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import xpass_matrix


def train_xpass(df: pl.DataFrame, nrounds: int = C.XPASS_NROUNDS) -> xgb.Booster:
    X, y, _ = xpass_matrix(df)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(C.XPASS_PARAMS, dtrain, num_boost_round=nrounds)
