"""Phase 6 smoke test: full CLI pipeline on synthetic final.json data."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from cpoe.constants import FEATURE_COLS, TARGET_COL


def _make_plays(n: int, game_id: str, rng: np.random.Generator) -> list[dict]:
    pass_types = ["Pass Reception", "Pass Incompletion", "Passing Touchdown"]
    return [
        {
            "game_id": game_id,
            "playType": rng.choice(pass_types),
            "start.down": int(rng.integers(1, 5)),
            "start.distance": int(rng.integers(1, 20)),
            "start.yardsToEndzone": int(rng.integers(1, 99)),
            "pos_score_diff_start": int(rng.integers(-21, 22)),
            "start.TimeSecsRem": int(rng.integers(0, 3600)),
            "start.is_home": bool(rng.integers(0, 2)),
            "period": int(rng.integers(1, 5)),
            "passing_down": int(rng.integers(0, 2)),
            "completion": int(rng.integers(0, 2)),
        }
        for _ in range(n)
    ]


@pytest.fixture()
def synthetic_raw_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    rng = np.random.default_rng(42)
    for season in (2021, 2022, 2023):
        for game_id in (f"{season}0001", f"{season}0002"):
            plays = _make_plays(30, game_id, rng)
            payload = {"season": season, "game_id": game_id, "plays": plays}
            (tmp_path / f"{game_id}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    return tmp_path


def test_cli_smoke_no_loso(synthetic_raw_dir, tmp_path):
    from cpoe.cli import main
    rc = main([
        "--final-dir", str(synthetic_raw_dir),
        "--out-dir", str(tmp_path / "out"),
        "--seasons", "2021", "2022", "2023",
    ])
    assert rc == 0
    assert (tmp_path / "out" / "cfb_cp_model.ubj").exists()
    assert not (tmp_path / "out" / "loso_cv.json").exists()


def test_cli_smoke_with_loso(synthetic_raw_dir, tmp_path):
    from cpoe.cli import main
    rc = main([
        "--final-dir", str(synthetic_raw_dir),
        "--out-dir", str(tmp_path / "out_loso"),
        "--seasons", "2021", "2022", "2023",
        "--loso",
    ])
    assert rc == 0
    cv_path = tmp_path / "out_loso" / "loso_cv.json"
    assert cv_path.exists()
    cv = json.loads(cv_path.read_text())
    assert "folds" in cv
    assert len(cv["folds"]) == 3
    assert (tmp_path / "out_loso" / "cfb_cp_model.ubj").exists()


def test_cli_no_seasons_returns_nonzero(synthetic_raw_dir, tmp_path):
    from cpoe.cli import main
    rc = main([
        "--final-dir", str(synthetic_raw_dir),
        "--out-dir", str(tmp_path / "out_fail"),
    ])
    assert rc != 0


def test_cli_missing_raw_dir_returns_nonzero(tmp_path):
    from cpoe.cli import main
    rc = main([
        "--final-dir", str(tmp_path / "nonexistent"),
        "--out-dir", str(tmp_path / "out_fail"),
        "--seasons", "2021",
    ])
    assert rc != 0
