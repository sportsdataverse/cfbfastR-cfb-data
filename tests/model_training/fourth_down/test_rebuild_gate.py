"""The fd rebuild gate's rule, metrics and row bookkeeping (the gate itself runs once, on real data)."""

import json
import pathlib

import numpy as np
import polars as pl

from cfb_model_build.model_training.fourth_down import rebuild_gate as g
from cfb_model_build.model_training.fourth_down.features import fd_features

FIX = (
    pathlib.Path(__file__).parent.parent.parent
    / "fixtures"
    / "model_training"
    / "fd_fixture_plays.json"
)


def test_cal_err_bins_and_weights() -> None:
    # bin 0.1 holds 3 plays with rate 1/3; bin 0.5 holds 1 play with rate 1
    p = np.array([0.1, 0.1, 0.11, 0.5])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    assert abs(g.cal_err(p, y) - (3 * abs(0.1 - 1 / 3) + 1 * 0.5) / 4) < 1e-12


def test_paired_is_zero_for_identical_predictions_and_signs_a_better_arm() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600).astype(float)
    games = np.repeat(np.arange(60), 10)
    base = np.full(600, 0.5)
    same = g.paired(y, base, base, games, n_boot=50)
    assert (
        same["brier_diff"] == 0
        and same["brier_diff_hi"] == 0
        and same["cal_err_diff"] == 0
    )
    better = g.paired(y, 0.5 + 0.4 * (y - 0.5), base, games, n_boot=50)
    assert better["brier_diff_hi"] < 0


def test_decide_needs_both_baselines_on_both_metrics() -> None:
    win = {"brier_diff_hi": -1e-4, "cal_err_diff_hi": 0.001}
    rec = {
        "arms": {"R1": {"brier": 0.20}, "R2": {"brier": 0.19}},
        "paired": {
            "R1-A": win,
            "R1-B": win,
            "R2-A": win,
            "R2-B": {**win, "cal_err_diff_hi": g.CAL_MARGIN},
        },
    }
    assert (
        g.decide(rec) == "R1" and rec["arms"]["R2"]["passed"] is False
    )  # R2 fails the margin vs B
    rec["paired"]["R1-B"] = {**win, "brier_diff_hi": 0.0}
    assert g.decide(rec) is None


def test_game_ids_follow_the_kept_rows() -> None:
    plays = pl.DataFrame(json.loads(FIX.read_text()), infer_schema_length=None)
    plays = plays.with_columns(
        pl.Series("_gid", np.arange(plays.height) + 400_000_000, dtype=pl.Int64)
    )
    X, y = fd_features(plays)
    Xg, _ = fd_features(
        plays.with_columns(
            pl.col("_gid").cast(pl.Float64).alias("start.pos_team_spread")
        )
    )
    assert len(Xg) == len(X) and np.array_equal(
        X["distance"].to_numpy(), Xg["distance"].to_numpy()
    )
    kept = set(Xg["posteam_spread"].astype(np.int64))
    assert kept <= set(plays["_gid"].to_list()) and len(kept) == len(X)
