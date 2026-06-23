"""Matrix + trainer tests for the FG / xPass / two-point models.

Mirrors test_train_ep.py / test_train_wp.py / test_features.py: synthetic frames
exercise the feature order, n-features, label derivation, and the binary:logistic
booster structure (so the trainers stay aligned with the shipped artifacts).
"""
import json

import numpy as np
import polars as pl

from model_training import constants as C
from model_training.features import fg_matrix, two_pt_matrix, xpass_matrix
from model_training.train_fg import train_fg
from model_training.train_two_pt import train_two_pt
from model_training.train_xpass import train_xpass


def _synth_fg_frame(n=400):
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "fg_attempt": [True] * n,
        "start.yardsToEndzone": rng.integers(1, 56, n),
        "fg_made": rng.integers(0, 2, n),
    })


def _synth_xpass_frame(n=600):
    rng = np.random.default_rng(1)
    return pl.DataFrame({
        "rush": rng.integers(0, 2, n).astype(bool),
        "pass": rng.integers(0, 2, n).astype(bool),
        "start.down": rng.integers(1, 5, n),
        "start.distance": rng.integers(1, 20, n),
        "start.yardsToEndzone": rng.integers(1, 99, n),
        "pos_score_diff_start": rng.integers(-21, 22, n),
        "start.TimeSecsRem": rng.integers(0, 3600, n),
        "period": rng.integers(1, 5, n),
        "season": rng.integers(2004, 2026, n),
    })


def _synth_two_pt_frame(n=300):
    rng = np.random.default_rng(2)
    return pl.DataFrame({
        "two_point_conv_result": rng.choice(["success", "failure"], n),
        "start.pos_team_spread": rng.normal(0, 7, n),
        "homeTeamSpread": rng.normal(0, 7, n),
        "overUnder": rng.uniform(40, 70, n),
        "start.is_home": rng.integers(0, 2, n).astype(bool),
        "pos_score_diff_start": rng.integers(-21, 22, n),
        "season": rng.integers(2004, 2026, n),
    })


# --- matrices: feature order + label shape ------------------------------------

def test_fg_matrix_order_and_filter():
    X, y, w = fg_matrix(_synth_fg_frame())
    assert list(X.columns) == C.FG_FEATURES == ["yards_to_goal"]
    assert w is None
    assert set(np.unique(y)).issubset({0, 1})


def test_fg_matrix_drops_out_of_range_attempts():
    df = pl.DataFrame({
        "fg_attempt": [True, True, True, False],
        "start.yardsToEndzone": [0, 30, 56, 30],  # 0 and 56 out of [1,55]; False dropped
        "fg_made": [1, 1, 1, 1],
    })
    X, y, _ = fg_matrix(df)
    assert len(X) == 1 and X["yards_to_goal"].tolist() == [30]


def test_xpass_matrix_order_and_label():
    X, y, w = xpass_matrix(_synth_xpass_frame())
    assert list(X.columns) == C.XPASS_FEATURES
    assert len(C.XPASS_FEATURES) == 7
    assert w is None
    assert set(np.unique(y)).issubset({0, 1})


def test_xpass_era_is_ordinal_0_to_3():
    df = pl.DataFrame({
        "rush": [True, True, True, True],
        "pass": [False, False, False, False],
        "start.down": [1, 1, 1, 1],
        "start.distance": [10, 10, 10, 10],
        "start.yardsToEndzone": [50, 50, 50, 50],
        "pos_score_diff_start": [0, 0, 0, 0],
        "start.TimeSecsRem": [900, 900, 900, 900],
        "period": [1, 1, 1, 1],
        "season": [2005, 2010, 2016, 2022],
    })
    X, _, _ = xpass_matrix(df)
    assert X["era"].tolist() == [0, 1, 2, 3]


def test_two_pt_matrix_order_and_label():
    X, y, w = two_pt_matrix(_synth_two_pt_frame())
    assert list(X.columns) == C.TWO_PT_FEATURES
    assert len(C.TWO_PT_FEATURES) == 4
    assert w is None
    assert set(np.unique(y)).issubset({0, 1})


def test_two_pt_posteam_total_home_vs_away():
    df = pl.DataFrame({
        "two_point_conv_result": ["success", "failure"],
        "start.pos_team_spread": [-3.0, 3.0],
        "homeTeamSpread": [-6.0, -6.0],
        "overUnder": [50.0, 50.0],
        "start.is_home": [True, False],
        "pos_score_diff_start": [0, 0],
        "season": [2020, 2020],
    })
    X, y, _ = two_pt_matrix(df)
    # home: (-6+50)/2 = 22 ; away: (50-(-6))/2 = 28
    assert X["posteam_total"].tolist() == [22.0, 28.0]
    assert y.tolist() == [1, 0]


# --- trainers: structure matches the shipped binary:logistic artifacts --------

def test_train_fg_is_1feat_logistic():
    m = train_fg(_synth_fg_frame(), nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 1
    assert m.feature_names == C.FG_FEATURES
    assert cfg["objective"]["name"] == "binary:logistic"


def test_train_xpass_is_7feat_logistic():
    m = train_xpass(_synth_xpass_frame(), nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 7
    assert m.feature_names == C.XPASS_FEATURES
    assert cfg["objective"]["name"] == "binary:logistic"


def test_train_two_pt_is_4feat_logistic():
    m = train_two_pt(_synth_two_pt_frame(), nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 4
    assert m.feature_names == C.TWO_PT_FEATURES
    assert cfg["objective"]["name"] == "binary:logistic"
