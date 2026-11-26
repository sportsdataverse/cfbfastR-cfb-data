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
    gid = np.arange(plays.height) + 400_000_000
    plays = plays.with_columns(pl.Series("_gid", gid, dtype=pl.Int64))
    raw = dict(
        zip(plays["_gid"].to_list(), plays["start.distance"].to_list(), strict=True)
    )
    for one_hot in (False, True):
        X, y, games = g.holdout_frame(plays, one_hot=one_hot)
        assert len(games) == len(X) == len(y) > 0
        # row order is kept, and each kept row's id points back at a raw row with its distance
        assert np.all(np.diff(games) > 0)
        assert [raw[int(k)] for k in games] == X["distance"].tolist()
        # the rows the gate scores are exactly fd_features' rows
        X0, y0 = fd_features(plays, era_onehot=one_hot)
        assert X.equals(X0) and np.array_equal(y, y0)


def test_first_down_boundary_is_gain_at_least_distance() -> None:
    dist = np.array([5, 5])
    probs = np.zeros((2, g.FD_NUM_CLASS))
    probs[0, 5 + g.FD_LABEL_OFFSET] = 1.0  # gain exactly the distance: converts
    probs[1, 4 + g.FD_LABEL_OFFSET] = 1.0  # one yard short: does not
    assert g.p_first_down(probs, dist).tolist() == [1.0, 0.0]
    labels = np.array([5, 4]) + g.FD_LABEL_OFFSET
    assert g.first_down_event(labels, dist).tolist() == [1.0, 0.0]


def test_run_refuses_a_second_look(tmp_path) -> None:
    (tmp_path / g.GATE).write_text("{}")
    assert g.run(tmp_path, tmp_path / "x", tmp_path / "y", tmp_path) == 2


def test_run_refuses_inputs_other_than_the_preregistered(tmp_path) -> None:
    for name in (g.MANIFEST, "pbp.parquet", "fd.ubj"):
        (tmp_path / name).write_text("not the registered file")
    assert g.run(tmp_path, tmp_path / "pbp.parquet", tmp_path / "fd.ubj", tmp_path) == 2
