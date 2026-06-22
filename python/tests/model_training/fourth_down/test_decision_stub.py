"""Tests for the fourth-down GO win-probability layer (get_go_wp_py).

The real-artifact tests load the shipped boosters (artifacts/ep.ubj,
artifacts/wp_spread.ubj, artifacts/fd_model.ubj) and a handful of real 4th-down
rows from artifacts/pbp_full.parquet. They are skipped (suite stays green) when the
artifacts are absent. A pure-synthetic smoke test trains tiny stand-in boosters so
the wiring is exercised even with no artifacts on disk.
"""
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xgboost as xgb

from model_training.fourth_down.fourth_down_decision import get_go_wp_py

ART = pathlib.Path(__file__).parents[3] / "artifacts"
EP = ART / "ep.ubj"
WP = ART / "wp_spread.ubj"
FD = ART / "fd_model.ubj"
PBP = ART / "pbp_full.parquet"

_HAVE_ARTIFACTS = all(p.exists() for p in (EP, WP, FD, PBP))
_skip = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="EP/WP/fd_model/pbp_full artifacts not on disk"
)


def _load(path: pathlib.Path) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(str(path))
    return b


@pytest.fixture(scope="module")
def boosters():
    return _load(EP), _load(WP), _load(FD)


def _fourth_down_rows(n: int = 60) -> pl.DataFrame:
    """Real 4th-down rows with all state columns non-null."""
    needed = [
        "start.down", "start.distance", "start.yardsToEndzone", "start.pos_team_spread",
        "pos_score_diff_start", "start.TimeSecsRem", "start.adj_TimeSecsRem",
        "start.pos_team_receives_2H_kickoff", "start.posTeamTimeouts",
        "start.defPosTeamTimeouts", "start.is_home", "period", "season",
        "overUnder", "homeTeamSpread",
    ]
    lf = pl.scan_parquet(PBP).filter(pl.col("start.down") == 4)
    lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in needed]))
    return lf.select(needed).head(n).collect()


@_skip
def test_columns_present_and_in_unit_interval(boosters):
    ep, wp, fd = boosters
    rows = _fourth_down_rows()
    out = get_go_wp_py(rows, fd, ep, wp)
    for col in ("go_wp", "first_down_prob", "wp_succeed", "wp_fail"):
        assert col in out.columns, f"missing {col}"
        v = out[col].to_numpy().astype(float)
        assert not np.isnan(v).any(), f"{col} has nulls"
        assert (v >= 0.0).all() and (v <= 1.0).all(), f"{col} out of [0,1]"


@_skip
def test_row_count_preserved(boosters):
    ep, wp, fd = boosters
    rows = _fourth_down_rows()
    out = get_go_wp_py(rows, fd, ep, wp)
    assert len(out) == len(rows)


@_skip
def test_first_down_prob_monotonic_short_vs_long(boosters):
    """4th-and-1 near midfield converts more often than 4th-and-20 at the same spot."""
    ep, wp, fd = boosters
    base = {
        "start.yardsToEndzone": 50, "start.pos_team_spread": 0.0,
        "pos_score_diff_start": 0, "start.TimeSecsRem": 1800,
        "start.adj_TimeSecsRem": 1800, "start.pos_team_receives_2H_kickoff": False,
        "start.posTeamTimeouts": 3, "start.defPosTeamTimeouts": 3,
        "start.is_home": True, "period": 2, "season": 2022,
        "overUnder": 55.0, "homeTeamSpread": 0.0,
    }
    short = {**base, "start.down": 4, "start.distance": 1}
    longg = {**base, "start.down": 4, "start.distance": 20}
    df = pl.DataFrame([short, longg])
    out = get_go_wp_py(df, fd, ep, wp)
    fdp = out["first_down_prob"].to_numpy()
    assert fdp[0] > fdp[1], f"4th-and-1 ({fdp[0]:.3f}) should beat 4th-and-20 ({fdp[1]:.3f})"


@_skip
def test_goal_to_go_short_high_conversion(boosters):
    """4th-and-goal at the 2 should convert at a believable (>0.25) rate."""
    ep, wp, fd = boosters
    row = pl.DataFrame([{
        "start.down": 4, "start.distance": 2, "start.yardsToEndzone": 2,
        "start.pos_team_spread": -3.0, "pos_score_diff_start": 0,
        "start.TimeSecsRem": 1800, "start.adj_TimeSecsRem": 1800,
        "start.pos_team_receives_2H_kickoff": False, "start.posTeamTimeouts": 3,
        "start.defPosTeamTimeouts": 3, "start.is_home": True, "period": 2,
        "season": 2022, "overUnder": 55.0, "homeTeamSpread": -3.0,
    }])
    out = get_go_wp_py(row, fd, ep, wp)
    assert 0.25 < float(out["first_down_prob"].iloc[0]) <= 1.0


def test_empty_input_returns_empty_with_columns():
    df = pl.DataFrame({c: pl.Series([], dtype=pl.Int64) for c in (
        "start.down", "start.distance", "start.yardsToEndzone", "period", "season"
    )})
    out = get_go_wp_py(df, None, None, None)
    for col in ("go_wp", "first_down_prob", "wp_succeed", "wp_fail"):
        assert col in out.columns
    assert len(out) == 0


def test_synthetic_smoke_without_artifacts():
    """End-to-end wiring with tiny stand-in boosters (no artifacts needed)."""
    rng = np.random.default_rng(0)

    # tiny fd_model: 6-feat, 76-class softprob
    fd_X = pd.DataFrame({
        "down": rng.integers(3, 5, 300).astype(float),
        "distance": rng.integers(1, 25, 300).astype(float),
        "yards_to_goal": rng.integers(2, 99, 300).astype(float),
        "posteam_total": rng.uniform(20, 70, 300),
        "posteam_spread": rng.uniform(-30, 30, 300),
        "era": rng.integers(0, 4, 300).astype(float),
    })
    fd_y = rng.integers(0, 76, 300)
    fd = xgb.train(
        {"objective": "multi:softprob", "num_class": 76, "max_depth": 2},
        xgb.DMatrix(fd_X, label=fd_y), num_boost_round=3,
    )

    # tiny EP: 8-feat, 7-class softprob
    ep_X = pd.DataFrame(rng.random((300, 8)), columns=[
        "TimeSecsRem", "yards_to_goal", "distance",
        "down_1", "down_2", "down_3", "down_4", "pos_score_diff_start",
    ])
    ep = xgb.train(
        {"objective": "multi:softprob", "num_class": 7, "max_depth": 2},
        xgb.DMatrix(ep_X, label=rng.integers(0, 7, 300)), num_boost_round=3,
    )

    # tiny WP: 13-feat binary:logistic
    wp_cols = [
        "pos_team_receives_2H_kickoff", "spread_time", "TimeSecsRem", "adj_TimeSecsRem",
        "ExpScoreDiff_Time_Ratio", "pos_score_diff_start", "down", "distance",
        "yards_to_goal", "is_home", "pos_team_timeouts_rem_before",
        "def_pos_team_timeouts_rem_before", "period",
    ]
    wp_X = pd.DataFrame(rng.random((300, 13)), columns=wp_cols)
    wp = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2},
        xgb.DMatrix(wp_X, label=rng.integers(0, 2, 300)), num_boost_round=3,
    )

    rows = pl.DataFrame([
        {
            "start.down": 4, "start.distance": d, "start.yardsToEndzone": ytg,
            "start.pos_team_spread": -2.5, "pos_score_diff_start": 0,
            "start.TimeSecsRem": 1500, "start.adj_TimeSecsRem": 1500,
            "start.pos_team_receives_2H_kickoff": False, "start.posTeamTimeouts": 3,
            "start.defPosTeamTimeouts": 3, "start.is_home": True, "period": 2,
            "season": 2022, "overUnder": 55.0, "homeTeamSpread": -2.5,
        }
        for d, ytg in [(1, 50), (5, 50), (10, 60), (2, 2)]
    ])
    out = get_go_wp_py(rows, fd, ep, wp)
    assert len(out) == 4
    for col in ("go_wp", "first_down_prob", "wp_succeed", "wp_fail"):
        v = out[col].to_numpy().astype(float)
        ok = v[~np.isnan(v)]
        assert ((ok >= 0.0) & (ok <= 1.0)).all(), f"{col} out of [0,1]"
    # first_down_prob must be well-defined (not nan) for every play
    assert not np.isnan(out["first_down_prob"].to_numpy().astype(float)).any()
