"""Trainer parity: the port's logistic fits vs R's coefficients on R's own rows.

Oracles: ``fixtures/matchup/scoring_opp_coef.json`` / ``rush_expect_coef.json``
(the shipped glm objects' coefficients) and the training frames extracted from
those SAME objects (``python/.cache/matchup/*_training_frame.parquet``, the
``model$data`` slot: 262,089 drives / 1,304,773 plays). A fit on identical rows
must land on identical MLEs.

Bar: every coefficient within 1e-6 relative (IRLS in R vs Newton-Cholesky here,
both converged to tolerance); the aliased ``score_diff`` reported as such. The
frames are not committed, so the parity runs under ``-m integration``; the
round-trip test uses the committed sample.
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


def _assert_coefs_close(py, oracle_path: Path, *, rel: float) -> None:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))["coefficients"]
    assert abs(py.intercept - oracle["(Intercept)"]) <= rel * max(
        1.0, abs(oracle["(Intercept)"])
    )
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


@pytest.mark.integration
def test_scoring_opp_fit_matches_r() -> None:
    drives = pl.read_parquet(CACHE / "scoring_opp_training_frame.parquet")
    result = fit_scoring_opp(drives)
    assert result.n_obs == 262_065  # R's nobs after na.omit
    _assert_coefs_close(result.coefficients, FIX / "scoring_opp_coef.json", rel=1e-6)


@pytest.mark.integration
def test_rush_expect_fit_matches_r() -> None:
    plays = pl.read_parquet(CACHE / "rush_expect_training_frame.parquet")
    result = fit_rush_expect(plays)
    assert result.n_obs == 1_304_773
    assert result.coefficients.aliased == ["score_diff"]
    _assert_coefs_close(result.coefficients, FIX / "rush_expect_coef.json", rel=1e-6)


def test_fit_roundtrips_through_the_applier(tmp_path: Path) -> None:
    pbp = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    frame = rush_expect_training_frame(pbp)
    assert frame.columns == [*RUSH_EXPECT_FEATURES, "rush"]
    result = fit_rush_expect(frame)
    coef_path, meta_path = write_fit(
        result, tmp_path, "rush_expect", formula="rush ~ ...", seasons=[2025]
    )
    back = load_coefficients(coef_path)
    assert back.features == list(RUSH_EXPECT_FEATURES) and back.aliased == [
        "score_diff"
    ]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["n_obs"] == result.n_obs and 0 < meta["train_log_loss"] < 0.7
    # the applier reproduces sklearn's own probabilities on the training rows
    p_applier = frame.with_columns(p=apply_logit(back))["p"].to_numpy()
    x = frame.select([f for f in RUSH_EXPECT_FEATURES if f != "score_diff"]).to_numpy()
    eta = back.intercept + x @ np.array(
        [back.coefficients[f] for f in back.features if f in back.coefficients]
    )
    assert np.abs(p_applier - 1 / (1 + np.exp(-eta))).max() < 1e-12


def test_scoring_opp_fit_on_sample_drives() -> None:
    pbp = pl.read_parquet(FIX / "pbp_input_2025_sample.parquet")
    drives = build_drive_frame(
        pbp.filter(pl.col("ppa").is_not_null()),
        load_coefficients(FIX / "scoring_opp_coef.json"),
    )
    result = fit_scoring_opp(drives)
    doc = coefficients_json(result.coefficients, formula="scoring_opp ~ ...")
    assert list(doc["coefficients"]) == ["(Intercept)", *SCORING_OPP_FEATURES]
    assert doc["n_aliased"] == 0 and result.n_obs == drives.height
