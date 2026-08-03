"""Tests for the cfbscrapR / nflfastR calibration recipe in cfb_model_reports.figures.

These cover the *recipe* (binning + weighted calibration error) and the defensive
builder contracts (missing parquet -> []), without requiring plotnine to render a
PNG. The faceted figure rendering itself is exercised by the live regeneration in
``python -m cfb_model_reports``.
"""
import numpy as np
import polars as pl

from cfb_model_reports import figures as figmod


def test_binned_calibration_recipe_matches_cfbscrapR():
    """bin_pred_prob = round(p/0.05)*0.05; bin_actual = n_event/n_plays per (facet,bin)."""
    # Two facets, perfectly-calibrated synthetic data: predicted prob == event rate.
    pred = np.array([0.10, 0.10, 0.10, 0.10, 0.90, 0.90, 0.90, 0.90])
    event = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 0% at 0.1 bin? -> mismatch on purpose
    facet = np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)
    tab, cal_err = figmod._binned_calibration(pred, event, facet, bin_size=0.05)
    # Two bins: (A, 0.10) all-zero actual; (B, 0.90) all-one actual.
    assert set(tab["by"].to_list()) == {"A", "B"}
    a = tab.filter(pl.col("by") == "A")
    b = tab.filter(pl.col("by") == "B")
    assert a["bin"][0] == 0.10 and a["actual"][0] == 0.0
    assert b["bin"][0] == 0.90 and b["actual"][0] == 1.0
    assert a["n_plays"][0] == 4 and b["n_plays"][0] == 4
    # |0.10-0.0|=0.10 (A) and |0.90-1.0|=0.10 (B); weighted by n_event:
    # A has 0 events so its facet weight is 0 -> overall error is B's 0.10.
    assert abs(cal_err - 0.10) < 1e-9


def test_binned_calibration_perfect_is_zero_error():
    rng = np.random.default_rng(0)
    # Generate events with rate == bin center so binned actual ~ predicted.
    bins = np.repeat([0.05, 0.25, 0.55, 0.85], 2000)
    pred = bins.copy()
    event = (rng.random(len(bins)) < bins).astype(int)
    facet = np.array(["all"] * len(bins), dtype=object)
    _, cal_err = figmod._binned_calibration(pred, event, facet, bin_size=0.05)
    assert cal_err < 0.01  # well-calibrated by construction


def test_ep_class_label_covers_seven_classes():
    assert set(figmod._EP_CLASS_LABEL) == set(range(7))
    assert figmod._EP_CLASS_LABEL[0].startswith("Touchdown")
    assert figmod._EP_CLASS_LABEL[6].startswith("No Score")


def test_builders_missing_parquet_return_empty(tmp_path):
    missing = tmp_path / "nope.parquet"
    assert figmod.build_ep_class_calibration(missing, tmp_path, tmp_path) == []
    assert figmod.build_wp_quarter_calibration(
        missing, tmp_path, tmp_path, stem="x", title="t", subtitle="s"
    ) == []


def test_ep_class_builder_requires_prob_columns(tmp_path):
    """An OOF without p0..p6 returns [] rather than raising."""
    p = tmp_path / "loso_ep_class_oof.parquet"
    pl.DataFrame({"season": [2024, 2024], "y": [0, 6]}).write_parquet(p)
    assert figmod.build_ep_class_calibration(p, tmp_path, tmp_path) == []


def test_wp_quarter_builder_requires_period(tmp_path):
    p = tmp_path / "wp.parquet"
    pl.DataFrame({"y": [0, 1], "wp_pred": [0.3, 0.7]}).write_parquet(p)
    assert figmod.build_wp_quarter_calibration(
        p, tmp_path, tmp_path, stem="x", title="t", subtitle="s"
    ) == []


def test_new_binary_builders_missing_parquet_return_empty(tmp_path):
    missing = tmp_path / "nope.parquet"
    assert figmod.build_fg_calibration(missing, tmp_path, tmp_path) == []
    assert figmod.build_xpass_calibration(missing, tmp_path, tmp_path) == []
    assert figmod.build_two_pt_calibration(missing, tmp_path, tmp_path) == []
    assert figmod.build_pregame_wp_calibration(missing, tmp_path, tmp_path) == []


def test_new_binary_builders_require_their_columns(tmp_path):
    """A parquet missing the prediction/event columns returns [] not a raise."""
    fg = tmp_path / "loso_fg_oof.parquet"
    pl.DataFrame({"season": [2024, 2024]}).write_parquet(fg)
    assert figmod.build_fg_calibration(fg, tmp_path, tmp_path) == []
    xp = tmp_path / "loso_xpass_oof.parquet"
    pl.DataFrame({"season": [2024], "down": [1.0]}).write_parquet(xp)
    assert figmod.build_xpass_calibration(xp, tmp_path, tmp_path) == []
    tp = tmp_path / "loso_two_pt_oof.parquet"
    pl.DataFrame({"season": [2024], "made": [0]}).write_parquet(tp)
    assert figmod.build_two_pt_calibration(tp, tmp_path, tmp_path) == []
    pg = tmp_path / "loso_pgwp_oof.parquet"
    pl.DataFrame({"season": [2024], "pred_pts": [3.0]}).write_parquet(pg)
    assert figmod.build_pregame_wp_calibration(pg, tmp_path, tmp_path) == []
