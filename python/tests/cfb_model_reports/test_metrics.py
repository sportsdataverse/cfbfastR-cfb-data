"""Tests for cfb_model_reports.metrics — pure + hermetic."""
import numpy as np
import polars as pl
import pytest

from cfb_model_reports.metrics import (
    binary_loso_metrics,
    compute_classification_metrics,
    ep_loso_metrics,
    provenance_from_card,
    qbr_loso_metrics,
    rb_eval_metrics,
    wp_loso_metrics,
)


def test_provenance_from_card_tolerates_missing():
    """provenance_from_card extracts keys with safe defaults."""
    p = provenance_from_card(
        {
            "features": ["a", "b"],
            "training_seasons": [2014, 2024],
            "trained_date": "2026-06-17",
        }
    )
    assert p["features"] == ["a", "b"]
    assert p["training_seasons"] == [2014, 2024]
    assert p["trained_date"] == "2026-06-17"

    # Empty card returns defaults.
    p_empty = provenance_from_card({})
    assert p_empty == {
        "features": [],
        "hyperparameters": {},
        "training_seasons": None,
        "trained_date": None,
        "xgboost_version": None,
    }


def test_compute_classification_metrics_binary():
    """compute_classification_metrics returns n, log_loss, and brier_score for binary."""
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    m = compute_classification_metrics(y, p)
    assert m["n"] == 4
    assert 0.0 <= m["brier_score"] <= 1.0
    assert m["log_loss"] > 0.0


@pytest.mark.parametrize(
    "y,p,expected_brier",
    [
        (np.array([1, 0]), np.array([1.0, 0.0]), 0.0),  # Perfect predictions
        (np.array([1, 0]), np.array([0.5, 0.5]), 0.25),  # Worst-case binary
    ],
)
def test_compute_classification_metrics_values(y, p, expected_brier):
    """Brier score matches expected values."""
    m = compute_classification_metrics(y, p)
    assert np.isclose(m["brier_score"], expected_brier)


def test_compute_classification_metrics_single_class_no_raise():
    """log_loss with labels=[0,1] doesn't raise when y_true is all-ones."""
    y = np.array([1, 1, 1])
    p = np.array([0.8, 0.9, 0.7])
    m = compute_classification_metrics(y, p)
    assert np.isfinite(m["log_loss"])
    assert m["n"] == 3


def test_compute_classification_metrics_empty():
    """Empty arrays return early with {n: 0}."""
    m = compute_classification_metrics(np.array([]), np.array([]))
    assert m == {"n": 0}


def test_rb_eval_metrics_from_loso(tmp_path):
    """rb_eval_metrics reads parquet and computes weighted_r2 + weighted_cal_error."""
    lp = tmp_path / "xrepa_loso.parquet"
    pl.DataFrame(
        {
            "exp_rb_epa": [0.1, 0.2, 0.3, 0.4],
            "target": [0.12, 0.18, 0.31, 0.39],
        }
    ).write_parquet(lp)
    m = rb_eval_metrics(lp)
    assert "weighted_r2" in m
    assert "weighted_cal_error" in m
    assert "n" in m
    assert m["n"] == 4
    assert isinstance(m["weighted_r2"], float)
    assert isinstance(m["weighted_cal_error"], float)


def test_ep_loso_metrics(tmp_path):
    """ep_loso_metrics computes weighted EP calibration MAE + means from OOF."""
    lp = tmp_path / "loso_ep_oof.parquet"
    pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "y": [0, 1, 2, 6],
            "ep_pred": [2.0, 2.0, 1.0, 1.0],
            "realized": [7.0, -7.0, 3.0, 0.0],
        }
    ).write_parquet(lp)
    m = ep_loso_metrics(lp)
    assert m["n"] == 4
    assert {"ep_cal_mae", "mean_pred_ep", "mean_realized"} <= set(m)
    assert m["mean_pred_ep"] == 1.5


def test_wp_loso_metrics(tmp_path):
    """wp_loso_metrics returns logloss/brier/auc from binary OOF."""
    lp = tmp_path / "loso_wp_oof.parquet"
    pl.DataFrame(
        {"season": [2024] * 4, "y": [1, 0, 1, 0], "wp_pred": [0.9, 0.1, 0.8, 0.2]}
    ).write_parquet(lp)
    m = wp_loso_metrics(lp)
    assert m["n"] == 4
    assert m["logloss"] > 0 and 0 <= m["brier"] <= 1
    assert m["auc"] == 1.0  # perfectly separable


def test_binary_loso_metrics(tmp_path):
    """binary_loso_metrics returns logloss/brier/auc/base_rate/weighted_cal_err."""
    lp = tmp_path / "loso_fg_oof.parquet"
    pl.DataFrame(
        {
            "season": [2024] * 4,
            "yards_to_goal": [18.0, 25.0, 40.0, 50.0],
            "made": [1, 1, 0, 0],
            "fg_pred": [0.9, 0.8, 0.2, 0.1],
        }
    ).write_parquet(lp)
    m = binary_loso_metrics(lp, pred_col="fg_pred", event_col="made")
    assert m["n"] == 4
    assert m["logloss"] > 0 and 0 <= m["brier"] <= 1
    assert m["auc"] == 1.0  # perfectly separable
    assert m["base_rate"] == 0.5
    assert m["weighted_cal_err"] >= 0.0


def test_binary_loso_metrics_custom_cols(tmp_path):
    """The pred/event column names are parameterised (xpass uses is_pass/xpass)."""
    lp = tmp_path / "loso_xpass_oof.parquet"
    pl.DataFrame(
        {"season": [2024] * 4, "down": [1.0, 2.0, 3.0, 1.0], "is_pass": [1, 0, 1, 0], "xpass": [0.8, 0.3, 0.7, 0.2]}
    ).write_parquet(lp)
    m = binary_loso_metrics(lp, pred_col="xpass", event_col="is_pass")
    assert m["n"] == 4 and "weighted_cal_err" in m and "base_rate" in m


def test_qbr_loso_metrics(tmp_path):
    """qbr_loso_metrics returns rmse/mae/r2/corr from regression OOF."""
    lp = tmp_path / "loso_qbr_oof.parquet"
    pl.DataFrame(
        {"season": [2024] * 4, "y": [50.0, 60.0, 70.0, 80.0], "qbr_pred": [52.0, 58.0, 71.0, 79.0]}
    ).write_parquet(lp)
    m = qbr_loso_metrics(lp)
    assert m["n"] == 4
    assert m["rmse"] > 0 and m["mae"] > 0
    assert 0.9 < m["r2"] <= 1.0
    assert 0.9 < m["corr"] <= 1.0
