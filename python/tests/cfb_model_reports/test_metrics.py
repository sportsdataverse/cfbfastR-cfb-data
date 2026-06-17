"""Tests for cfb_model_reports.metrics — pure + hermetic."""
import numpy as np
import polars as pl
import pytest

from cfb_model_reports.metrics import provenance_from_card, compute_classification_metrics, rb_eval_metrics


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
