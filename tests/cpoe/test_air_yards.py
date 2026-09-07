"""Air-yards CP variant: feature derivation, model routing, and save contract.

These are structural tests -- they check that rows reach the right model and
that the contracts hold. The *statistical* claim (air yards lifts AUC 0.567 ->
0.764) is validated against real published play-by-play in
``ClaudeCowork/notes/2026-09-07-cfb-xcp-air-yards/measure_air_yards.py``, not
here; a synthetic frame can't evidence it and shouldn't pretend to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from cfb_model_build.cpoe.constants import (
    AIR_YARDS_FEATURE_COLS,
    FEATURE_COLS,
    TARGET_COL,
)
from cfb_model_build.cpoe.cpoe import compute_cpoe, compute_cpoe_hybrid
from cfb_model_build.cpoe.features import extract_pass_features
from cfb_model_build.cpoe.train_cp import save_cp_model, train_cp_model


def _fit(features: list[str], n: int = 200) -> xgb.Booster:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.normal(size=n) for c in features})
    y = rng.integers(0, 2, size=n)
    return train_cp_model(X, y, nrounds=3, features=features)


def _frame(n: int = 6, *, air: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({c: np.arange(n, dtype=float) for c in FEATURE_COLS})
    df[TARGET_COL] = [1, 0] * (n // 2)
    if air:
        # first half has air yards, second half does not
        df["air_yards"] = [5.0] * (n // 2) + [np.nan] * (n - n // 2)
        df["pass_is_middle"] = [1.0] * (n // 2) + [np.nan] * (n - n // 2)
        df["qb_hurry"] = [0.0] * (n // 2) + [np.nan] * (n - n // 2)
    return df


# --------------------------------------------------------------------------
# feature derivation
# --------------------------------------------------------------------------


def test_pass_is_middle_is_null_when_direction_is_unknown():
    """A missing pass_direction must NOT be recorded as "not middle".

    Filling it 0 would tell the model every pre-2025 play was a non-middle
    throw, which is a fabricated observation rather than a missing one.
    """
    raw = pd.DataFrame(
        {
            "type.text": ["Pass Reception"] * 3,
            "start.down": [1, 2, 3],
            "start.distance": [10, 5, 8],
            "start.yardsToEndzone": [70, 40, 55],
            "completion": [1, 0, 1],
            "pass_direction": ["middle", "left", None],
        }
    )
    out = extract_pass_features(raw)
    assert out["pass_is_middle"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(out["pass_is_middle"].iloc[2])


def test_air_yards_survives_extraction_and_stays_nullable():
    raw = pd.DataFrame(
        {
            "type.text": ["Pass Reception", "Pass Incompletion"],
            "start.down": [1, 2],
            "start.distance": [10, 7],
            "start.yardsToEndzone": [70, 40],
            "completion": [1, 0],
            "air_yards": [12, None],
        }
    )
    out = extract_pass_features(raw)
    assert out["air_yards"].iloc[0] == 12.0
    assert pd.isna(out["air_yards"].iloc[1])


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def test_hybrid_routes_each_row_to_the_model_that_can_see_it():
    base, air = _fit(FEATURE_COLS), _fit(AIR_YARDS_FEATURE_COLS)
    out = compute_cpoe_hybrid(_frame(6), base, air)

    assert set(out["cp_model"]) == {"air_yards", "game_state"}
    assert (out.loc[out["air_yards"].notna(), "cp_model"] == "air_yards").all()
    assert (out.loc[out["air_yards"].isna(), "cp_model"] == "game_state").all()
    assert len(out) == 6, "every row must be scored exactly once"
    assert out["cp_pred"].between(0, 1).all()


def test_hybrid_without_an_air_model_falls_back_wholesale():
    out = compute_cpoe_hybrid(_frame(4), _fit(FEATURE_COLS), None)
    assert (out["cp_model"] == "game_state").all()
    assert out["cp_pred"].notna().all()


def test_hybrid_on_a_pre_2025_frame_uses_the_base_model():
    """No air-yards columns at all -- the historical case, must not raise."""
    out = compute_cpoe_hybrid(
        _frame(4, air=False), _fit(FEATURE_COLS), _fit(AIR_YARDS_FEATURE_COLS)
    )
    assert (out["cp_model"] == "game_state").all()


# --------------------------------------------------------------------------
# contracts that would otherwise fail silently
# --------------------------------------------------------------------------


def test_scoring_a_frame_missing_a_fitted_feature_raises():
    """Silently dropping to the intersection would change what CP means."""
    air = _fit(AIR_YARDS_FEATURE_COLS)
    with pytest.raises(ValueError, match="missing"):
        compute_cpoe(_frame(4, air=False), air, features=AIR_YARDS_FEATURE_COLS)


def test_ndarray_feature_count_mismatch_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    with pytest.raises(ValueError, match="feature names"):
        train_cp_model(X, rng.integers(0, 2, 20), nrounds=2, features=FEATURE_COLS)


def test_save_raises_when_the_model_card_cannot_be_written(tmp_path, monkeypatch):
    """A card failure must be loud: the publisher discovers models BY card.

    ``discover_models`` globs ``*.json`` cards and only then looks for the
    sibling ``.ubj``, so a model saved without one is silently never published.
    """
    import cfb_model_build.model_training.model_card as mc

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(mc, "write_xgb_model_card", boom)
    with pytest.raises(RuntimeError, match="model card"):
        save_cp_model(_fit(FEATURE_COLS), tmp_path / "m.ubj", features=FEATURE_COLS)


def test_save_writes_a_card_the_publisher_can_discover(tmp_path):
    from cfb_model_build.cfb_model_reports.discovery import discover_models

    save_cp_model(
        _fit(AIR_YARDS_FEATURE_COLS),
        tmp_path / "cfb_cp_model_air_yards.ubj",
        features=AIR_YARDS_FEATURE_COLS,
        model_type="cpoe_air_yards",
    )
    found = discover_models(tmp_path)
    assert [m.model_type for m in found] == ["cpoe_air_yards"]
    assert found[0].model_path.name == "cfb_cp_model_air_yards.ubj"


# --------------------------------------------------------------------------
# ingest season filtering
# --------------------------------------------------------------------------


def _write_game(d, name: str, season: int, *, head_pad: str = "") -> None:
    import json

    (d / name).write_text(
        json.dumps(
            {
                "pad": head_pad,
                "season": season,
                "plays": [
                    {
                        "type.text": "Pass Reception",
                        "start.down": 1,
                        "start.distance": 10,
                        "start.yardsToEndzone": 70,
                        "completion": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_loader_filters_by_season_through_the_prefilter(tmp_path):
    from cfb_model_build.cpoe.ingest import load_season_pass_plays

    _write_game(tmp_path, "a.json", 2025)
    _write_game(tmp_path, "b.json", 2024)
    out = load_season_pass_plays(tmp_path, seasons=[2025])
    assert len(out) == 1
    assert out["season"].tolist() == [2025]


def test_hybrid_preserves_input_row_order():
    """Scoring splits the frame in two; the output must come back in order.

    A caller assigning cpoe back onto its source frame by position would
    otherwise attach every value to the wrong play -- silently, with a
    plausible-looking column.
    """
    base, air = _fit(FEATURE_COLS), _fit(AIR_YARDS_FEATURE_COLS)
    df = _frame(6)
    # interleave so a naive concat would visibly reorder
    df["air_yards"] = [5.0, np.nan, 7.0, np.nan, 9.0, np.nan]
    df["down"] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]  # positional fingerprint

    out = compute_cpoe_hybrid(df, base, air)

    assert out["down"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert out["cp_model"].tolist() == [
        "air_yards", "game_state", "air_yards",
        "game_state", "air_yards", "game_state",
    ]
    assert "_cpoe_row_order" not in out.columns


def test_grouped_cv_survives_a_single_class_fold():
    """GroupKFold does not stratify, so a fold can be all-complete or all-incomplete.

    With the label set left to inference, log_loss raises ValueError on such a
    fold and the whole CV dies on the luck of the split. Five games, one of them
    entirely incompletions, reproduces it.
    """
    from cfb_model_build.cpoe.loso import run_grouped_cv

    rng = np.random.default_rng(0)
    rows = 20
    frames = []
    for game in range(5):
        f = pd.DataFrame({c: rng.normal(size=rows) for c in FEATURE_COLS})
        f["game_id"] = game
        # game 0 is entirely incompletions -> its fold is single-class
        f[TARGET_COL] = 0 if game == 0 else rng.integers(0, 2, rows)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)

    res = run_grouped_cv(df, n_splits=5, nrounds=2)
    assert len(res["folds"]) == 5
    assert all(np.isfinite(f["log_loss"]) for f in res["folds"])
