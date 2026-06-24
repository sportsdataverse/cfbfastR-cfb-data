import json
import numpy as np
import pandas as pd

from model_training import constants as C
from model_training.train_qbr import train_qbr_from_matrix


def test_qbr_model_matches_qbr_features_regression():
    """train_qbr_from_matrix trains on exactly C.QBR_FEATURES.

    ``QBR_FEATURES == qbr_vars`` (from sdv ``model_vars``) is era-aware — it
    includes the era0..era3 one-hot dummies alongside the 6 EPA base features —
    so the shipped qbr_model is a 10-feature regressor, not the legacy 6-feature
    one. Asserting against ``len(C.QBR_FEATURES)`` keeps this test tracking the
    live contract instead of a hard-coded count.
    """
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((300, len(C.QBR_FEATURES))), columns=C.QBR_FEATURES)
    y = rng.random(300) * 100
    m = train_qbr_from_matrix(X, y, nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == len(C.QBR_FEATURES) and cfg["objective"]["name"] == "reg:squarederror"
