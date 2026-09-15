"""Matchup-feature units vs the R oracle: play tagging, WEPA, model application,
drive frame, pace.

Oracle provenance: ``tests/cfb_data_build/fixtures/matchup/README.md``. The
committed fixtures cover 56 games of the 2025 season (every 30th game id); the
``integration``-marked tests re-run the same assertions over the full season
from ``python/.cache/matchup/``.

Parity bars (observed on the full 2025 season, 283,172 plays):
* the 60 situational flags -- exact, INCLUDING null placement (R ``ifelse`` on
  an NA condition yields NA; ``%in%`` / ``grepl`` on NA yield FALSE);
* ``off_wepa`` / ``def_wepa`` -- |diff| <= 1e-9 (a 61-term product; only the
  multiplication order differs);
* ``rp_prediction`` / ``scoring_opp_prediction`` -- |diff| <= 1e-9 vs R
  ``predict.glm(type = "response")`` on the frozen coefficient tables;
* drive frame -- keys and integer columns exact, floats <= 1e-9.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cfb_data_build.matchup_features import (
    FLAG_NAMES,
    RUSH_EXPECT_FEATURES,
    SCORING_OPP_FEATURES,
    apply_logit,
    apply_wepa,
    build_drive_frame,
    dedupe_plays,
    load_coefficients,
    load_wepa_weights,
    play_pace,
    tag_plays,
)

FIX = Path(__file__).parent / "fixtures" / "matchup"
CACHE = Path(__file__).parents[2] / "python" / ".cache" / "matchup"


@pytest.fixture(scope="module")
def pbp() -> pl.DataFrame:
    return dedupe_plays(pl.read_parquet(FIX / "pbp_input_2025_sample.parquet"))


@pytest.fixture(scope="module")
def oracle_plays() -> pl.DataFrame:
    return pl.read_csv(
        FIX / "play_flags_wepa_2025_sample.csv.gz", infer_schema_length=10000
    )


def _aligned(
    py: pl.DataFrame, oracle: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    keys = ["game_id", "id_play", "game_play_number"]
    py = py.sort(keys)
    oracle = oracle.sort(keys)
    assert py.height == oracle.height, (py.height, oracle.height)
    assert py.select(keys).equals(oracle.select(keys).cast(py.select(keys).schema))
    return py, oracle


def _assert_close(py: pl.Series, oracle: pl.Series, *, tol: float, name: str) -> None:
    a = py.cast(pl.Float64).to_numpy()
    b = oracle.cast(pl.Float64).to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert (np.isnan(a) == np.isnan(b)).all(), f"{name}: null placement differs"
    d = np.abs(a[~both_nan] - b[~both_nan])
    assert d.size == 0 or d.max() <= tol, f"{name}: max |diff| {d.max():.3e} > {tol}"


def test_flag_inventory_matches_oracle(oracle_plays: pl.DataFrame) -> None:
    oracle_flags = [c for c in oracle_plays.columns if c.endswith("_weight")]
    assert list(FLAG_NAMES) == [c.removesuffix("_weight") for c in oracle_flags]
    assert len(FLAG_NAMES) == 60


def test_tag_plays_matches_r(pbp: pl.DataFrame, oracle_plays: pl.DataFrame) -> None:
    tagged = tag_plays(pbp)
    py, oracle = _aligned(tagged, oracle_plays)
    for name in FLAG_NAMES:
        off = py[f"off_{name}_weight"]
        # the def_ copy is by construction identical to the off_ flag
        assert off.equals(py[f"def_{name}_weight"]), name
        _assert_close(off, oracle[f"{name}_weight"], tol=0.0, name=name)


def test_wepa_matches_r(pbp: pl.DataFrame, oracle_plays: pl.DataFrame) -> None:
    weights = load_wepa_weights(FIX / "wepa_weights.json")
    assert len(weights) == 120
    out = apply_wepa(tag_plays(pbp), weights)
    py, oracle = _aligned(out, oracle_plays)
    _assert_close(py["off_wepa"], oracle["off_wepa"], tol=1e-9, name="off_wepa")
    _assert_close(py["def_wepa"], oracle["def_wepa"], tol=1e-9, name="def_wepa")


def test_rush_expect_apply_matches_r(
    pbp: pl.DataFrame, oracle_plays: pl.DataFrame
) -> None:
    coef = load_coefficients(FIX / "rush_expect_coef.json")
    assert coef.features == list(RUSH_EXPECT_FEATURES)
    assert coef.aliased == ["score_diff"]  # rank-deficient fit: R drops it in predict
    out = pbp.with_columns(rp_prediction=apply_logit(coef))
    py, oracle = _aligned(out, oracle_plays)
    _assert_close(
        py["rp_prediction"], oracle["rp_prediction"], tol=1e-9, name="rp_prediction"
    )
    rroe = (py["rush"] - py["rp_prediction"]).alias("rroe")
    _assert_close(rroe, oracle["rroe"], tol=1e-9, name="rroe")


def test_drive_frame_matches_r(pbp: pl.DataFrame) -> None:
    coef = load_coefficients(FIX / "scoring_opp_coef.json")
    assert coef.features == list(SCORING_OPP_FEATURES)
    drives = build_drive_frame(pbp.filter(pl.col("ppa").is_not_null()), coef)
    oracle = pl.read_csv(FIX / "drives_2025_sample.csv", infer_schema_length=10000)
    keys = ["season", "start_date", "drive_id", "pos_team", "def_pos_team"]
    assert drives.height == oracle.height, (drives.height, oracle.height)
    # R's summarise() emits groups in key order; the frame order IS the contract
    # (prev_drive_result lags over it), so compare positionally.
    assert drives.select(keys).equals(
        oracle.select(keys).cast(drives.select(keys).schema)
    )
    for c in (
        "new_drive_pts",
        "start_yards_to_goal",
        "pos_team_timeouts",
        "def_pos_team_timeouts",
        "scoring_opp_ind_no_td",
        "scoring_opp",
        "half_secs_rem",
        "game_secs_rem",
    ):
        _assert_close(drives[c], oracle[c], tol=0.0, name=c)
    for c in ("scoring_opp_prediction", "scoring_opp_oe"):
        _assert_close(drives[c], oracle[c], tol=1e-9, name=c)
    # prev_drive_result lags over the season-wide frame in the oracle. Wherever
    # the previous SAMPLE row is a drive of the same game, the previous oracle
    # row is that same drive (the sample holds every drive of its games and the
    # key sort keeps a game's drives contiguous), so those rows must agree; a
    # game's first surviving drive sees a different neighbour in each frame.
    game_of = drives["drive_id"].cast(pl.Int64) // 100
    same_game_prev = (game_of.shift(1) == game_of).fill_null(False)
    assert same_game_prev.sum() > 1_000
    assert drives.filter(same_game_prev)["prev_drive_result"].equals(
        oracle.filter(same_game_prev)["prev_drive_result"]
    )


def test_play_pace_is_same_drive_and_non_negative(pbp: pl.DataFrame) -> None:
    paced = play_pace(pbp)
    assert "sec_since_prev" in paced.columns
    s = paced["sec_since_prev"].drop_nulls()
    assert s.min() >= 0
    # the first rush/pass play of every drive carries no interval
    first = paced.group_by(["game_id", "drive_id"], maintain_order=True).first()
    assert first["sec_since_prev"].null_count() == first.height


@pytest.mark.integration
def test_full_season_plays_match_r() -> None:
    pbp = dedupe_plays(pl.read_parquet(CACHE / "cfbfastR_cfb_pbp_2025.parquet"))
    assert pbp.height == 283_172  # R's distinct(game_id, id_play, game_play_number)
    oracle = pl.read_csv(
        CACHE / "play_flags_wepa_2025.csv.gz", infer_schema_length=10000
    )
    weights = load_wepa_weights(FIX / "wepa_weights.json")
    coef = load_coefficients(FIX / "rush_expect_coef.json")
    out = apply_wepa(tag_plays(pbp), weights).with_columns(
        rp_prediction=apply_logit(coef)
    )
    py, oracle = _aligned(out, oracle)
    for name in FLAG_NAMES:
        _assert_close(
            py[f"off_{name}_weight"], oracle[f"{name}_weight"], tol=0.0, name=name
        )
    for c in ("off_wepa", "def_wepa", "rp_prediction"):
        _assert_close(py[c], oracle[c], tol=1e-9, name=c)


@pytest.mark.integration
def test_full_season_drives_match_r() -> None:
    pbp = dedupe_plays(pl.read_parquet(CACHE / "cfbfastR_cfb_pbp_2025.parquet"))
    coef = load_coefficients(FIX / "scoring_opp_coef.json")
    drives = build_drive_frame(pbp.filter(pl.col("ppa").is_not_null()), coef)
    oracle = pl.read_csv(CACHE / "drives_2025.csv.gz", infer_schema_length=10000)
    assert drives.height == oracle.height == 36_151
    assert drives["prev_drive_result"].equals(oracle["prev_drive_result"])
    for c in ("scoring_opp", "scoring_opp_prediction"):
        _assert_close(drives[c], oracle[c], tol=1e-9, name=c)


def test_coefficient_json_roundtrip() -> None:
    raw = json.loads((FIX / "scoring_opp_coef.json").read_text(encoding="utf-8"))
    coef = load_coefficients(FIX / "scoring_opp_coef.json")
    assert coef.intercept == raw["coefficients"]["(Intercept)"]
    assert coef.family == "binomial" and coef.link == "logit"
