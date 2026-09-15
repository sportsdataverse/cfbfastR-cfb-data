"""Model family 35: trainer parity vs R and the level gates a wrong fit cannot pass.

Oracles: ``fixtures/matchup/scoring_opp_coef.json`` / ``rush_expect_coef.json``
(the source's fitted glm coefficients), the training frames extracted from
those same objects (``python/.cache/matchup/*_training_frame.parquet`` via
``ops/oneoff/20260915_matchup_oracle_capture/dump_training_frames.R``; sha256
recorded in the artifact metas), and ``rp_features_sample.csv`` (5,000 rush /
pass plays with R's own ``predict.glm`` value).

Gates (observed 2026-09-15; never lowered):
* refit on R's rows: max relative coefficient delta 5.7e-12 (scoring_opp,
  262,065 rows, 4 Newton iterations) and 1.8e-11 (rush_expect, 1,304,773
  rows, 5 iterations) -- bar 1e-6;
* shipped applier vs R ``predict.glm`` on the 5,000-play sample: max |diff|
  1.0e-15 -- bar 1e-9;
* a sample fit must beat the intercept-only log-loss by a margin: observed
  0.0449 (rush, 7,395 plays) and 0.0272 (scoring_opp, 1,225 drives) -- bars
  0.03 / 0.015. An all-zero-coefficient fit posts ln 2 = 0.693 and fails.
The R-row refits need the cached frames and run under ``-m integration``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cfb_data_build.matchup_features import (
    RUSH_EXPECT_FEATURES,
    SCORING_OPP_FEATURES,
    apply_logit,
    build_drive_frame,
    load_coefficients,
)
from cfb_model_build.cfb_matchup.glm import (
    coefficients_json,
    fit_rush_expect,
    fit_scoring_opp,
    rush_expect_training_frame,
    write_fit,
)

FIX = Path(__file__).parents[1] / "cfb_data_build" / "fixtures" / "matchup"
CACHE = Path(__file__).parents[2] / "python" / ".cache" / "matchup"
REGEN = (
    "regenerate with: Rscript ops/oneoff/20260915_matchup_oracle_capture/"
    "dump_training_frames.R <snapshot_dir> python/.cache/matchup"
)


def _frame(stem: str) -> pl.DataFrame:
    path = CACHE / f"{stem}_training_frame.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} absent -- {REGEN}")
    return pl.read_parquet(path)


def _assert_coefs_close(py, oracle_path: Path, *, rel: float) -> None:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))["coefficients"]
    assert abs(py.intercept - oracle["(Intercept)"]) <= rel * abs(oracle["(Intercept)"])
    for name, value in oracle.items():
        if name == "(Intercept)":
            continue
        if value is None:
            assert name in py.aliased, f"{name} should be aliased"
            continue
        got = py.coefficients[name]
        assert abs(got - value) <= rel * max(1e-3, abs(value)), (
            f"{name}: {got} vs {value}"
        )


def _intercept_only_log_loss(y: np.ndarray) -> float:
    q = y.mean()
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


@pytest.mark.integration
def test_scoring_opp_fit_matches_r() -> None:
    result = fit_scoring_opp(_frame("scoring_opp"))
    assert result.n_obs == 262_065  # R's nobs after na.omit
    _assert_coefs_close(result.coefficients, FIX / "scoring_opp_coef.json", rel=1e-6)


@pytest.mark.integration
def test_rush_expect_fit_matches_r() -> None:
    result = fit_rush_expect(_frame("rush_expect"))
    assert result.n_obs == 1_304_773
    assert result.coefficients.aliased == ["score_diff"]
    _assert_coefs_close(result.coefficients, FIX / "rush_expect_coef.json", rel=1e-6)


def test_shipped_rush_expect_applier_matches_r_predict() -> None:
    sample = pl.read_csv(FIX / "rp_features_sample.csv", null_values=["NA", ""])
    assert sample.height == 5_000
    p = sample.with_columns(
        p=apply_logit(load_coefficients(FIX / "rush_expect_coef.json"))
    )
    assert np.abs(p["p"].to_numpy() - sample["rp"].to_numpy()).max() <= 1e-9


def test_rush_expect_sample_fit_beats_intercept_only(tmp_path: Path) -> None:
    pbp = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    frame = rush_expect_training_frame(pbp)
    assert frame.columns == [*RUSH_EXPECT_FEATURES, "rush"]
    result = fit_rush_expect(frame)
    base = _intercept_only_log_loss(frame["rush"].to_numpy().astype(float))
    assert base - result.log_loss >= 0.03, (base, result.log_loss)  # observed 0.0449
    coef_path, meta_path = write_fit(
        result,
        tmp_path / "bundle",
        "rush_expect",
        formula="rush ~ ...",
        seasons=[2025],
        frame=frame,
        frame_dir=tmp_path / "cache",
    )
    back = load_coefficients(coef_path)
    assert back.features == list(RUSH_EXPECT_FEATURES) and back.aliased == [
        "score_diff"
    ]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["n_obs"] == result.n_obs and meta["n_iter"] == result.n_iter
    # the training frame is persisted content-addressed in frame_dir (the CLI
    # points it at the gitignored cache), never beside the bundle artifacts
    import hashlib

    frame_path = Path(meta["training_frame"])
    assert frame_path.parent == tmp_path / "cache" and frame_path.exists()
    assert not list((tmp_path / "bundle").glob("*.parquet"))
    digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
    assert meta["training_frame_sha256"] == digest and digest[:12] in frame_path.name
    assert meta["training_frame_rows"] == frame.height


def test_scoring_opp_sample_fit_beats_intercept_only() -> None:
    pbp = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    drives = build_drive_frame(
        pbp.filter(pl.col("ppa").is_not_null()),
        load_coefficients(FIX / "scoring_opp_coef.json"),
    )
    result = fit_scoring_opp(drives)
    base = _intercept_only_log_loss(drives["scoring_opp"].to_numpy().astype(float))
    assert base - result.log_loss >= 0.015, (base, result.log_loss)  # observed 0.0272
    doc = coefficients_json(result.coefficients, formula="scoring_opp ~ ...")
    assert list(doc["coefficients"]) == ["(Intercept)", *SCORING_OPP_FEATURES]
    assert doc["n_aliased"] == 0 and result.n_obs == drives.height
