import math

import numpy as np
import polars as pl
import pytest

from cfb_model_pbp.build import score_cpoe


def test_score_cpoe_appends_completion_prob_and_cpoe():
    carry = pl.DataFrame({"game_id": [1, 1], "id": [100, 101], "completion": [True, None], "pass": [True, False]})
    plays = pl.DataFrame({"game_id": [1, 1], "id": [100, 101], "completion": [True, None],
                          "type.text": ["Pass Completion", "Rush"], "start.down": [1, 2],
                          "start.distance": [10, 8], "start.yardsToEndzone": [75, 60],
                          "pos_score_diff_start": [0, 0], "start.TimeSecsRem": [1800, 1700],
                          "start.is_home": [True, True], "period": [1, 1], "passing_down": [False, False]},
                         infer_schema_length=None)
    out = score_cpoe(carry, plays, cp_model_path=None, _predict=lambda X: [0.6])  # 1 pass row -> cp 0.6
    pass_row = out.filter(pl.col("id") == 100).row(0, named=True)
    assert abs(pass_row["completion_prob"] - 0.6) < 1e-9
    assert abs(pass_row["cpoe"] - (1.0 - 0.6)) < 1e-9
    assert out.filter(pl.col("id") == 101).row(0, named=True)["completion_prob"] is None


def test_score_cpoe_real_model_path(tmp_path):
    """Real-model path: exercises feats[FEATURE_COLS] pandas indexing (regression for polars API misuse)."""
    import xgboost as xgb
    from cpoe.constants import FEATURE_COLS
    from cpoe.train_cp import train_cp_model

    # Build a tiny synthetic training set (20 rows, 8 features).
    rng = np.random.default_rng(42)
    n = 20
    X_train = {col: rng.integers(0, 5, size=n).tolist() for col in FEATURE_COLS}
    import pandas as pd
    X_df = pd.DataFrame(X_train)
    y = rng.integers(0, 2, size=n)

    booster = train_cp_model(X_df, y, nrounds=5, verbose_eval=False)
    model_path = tmp_path / "cp_test.ubj"
    booster.save_model(str(model_path))

    # Build a carry frame and plays frame with one pass play.
    carry = pl.DataFrame({"game_id": [99], "id": [200], "completion": [True]})
    plays = pl.DataFrame({
        "game_id": [99], "id": [200], "completion": [True],
        "type.text": ["Pass Completion"],
        "start.down": [1], "start.distance": [10], "start.yardsToEndzone": [75],
        "pos_score_diff_start": [0], "start.TimeSecsRem": [1800],
        "start.is_home": [True], "period": [1], "passing_down": [False],
    }, infer_schema_length=None)

    # Call score_cpoe with _predict=None — exercises the real booster load + pandas feats[FEATURE_COLS] path.
    out = score_cpoe(carry, plays, cp_model_path=model_path, _predict=None)

    row = out.filter(pl.col("id") == 200).row(0, named=True)
    cp = row["completion_prob"]
    assert cp is not None, "completion_prob must be non-null for a pass play"
    assert 0.0 <= cp <= 1.0, f"completion_prob out of [0,1]: {cp}"
    assert math.isfinite(row["cpoe"]), f"cpoe is not finite: {row['cpoe']}"
