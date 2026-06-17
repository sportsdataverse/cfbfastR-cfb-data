"""Phase 2 Task 2.3 — ingest module tests (final.json layout)."""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from cpoe.constants import FEATURE_COLS, TARGET_COL

# ---------------------------------------------------------------------------
# Minimal fixture: two pass plays + one rush, matching final.json structure
# (top-level keys: season, plays).
# ---------------------------------------------------------------------------

_PLAYS_FIXTURE = [
    {
        "game_id": "401628455",
        "playType": "Pass Reception",
        "start.down": 1,
        "start.distance": 10,
        "start.yardsToEndzone": 65,
        "pos_score_diff_start": 7,
        "start.TimeSecsRem": 1800,
        "start.is_home": True,
        "period": 2,
        "passing_down": 0,
        "completion": 1,
    },
    {
        "game_id": "401628455",
        "playType": "Pass Incompletion",
        "start.down": 3,
        "start.distance": 8,
        "start.yardsToEndzone": 30,
        "pos_score_diff_start": -3,
        "start.TimeSecsRem": 400,
        "start.is_home": False,
        "period": 4,
        "passing_down": 1,
        "completion": 0,
    },
    {
        "game_id": "401628455",
        "playType": "Rush",
        "start.down": 1,
        "start.distance": 10,
        "start.yardsToEndzone": 50,
        "pos_score_diff_start": 0,
        "start.TimeSecsRem": 3600,
        "start.is_home": True,
        "period": 1,
        "passing_down": 0,
        "completion": 0,
    },
]


def _write_final_json(directory: pathlib.Path, game_id: str, season: int, plays: list) -> None:
    """Write a minimal final.json file for a game."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"season": season, "game_id": game_id, "plays": plays}
    (directory / f"{game_id}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def season_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A mock final_dir with one game's final.json."""
    _write_final_json(tmp_path, "401628455", 2024, _PLAYS_FIXTURE)
    return tmp_path


def test_ingest_imports():
    from cpoe.ingest import load_season_pass_plays  # noqa: F401


def test_load_season_returns_dataframe(season_dir):
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    assert isinstance(df, pd.DataFrame)


def test_load_season_only_pass_plays(season_dir):
    """Rush play must be excluded; only 2 pass plays."""
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    assert len(df) == 2


def test_load_season_has_feature_cols(season_dir):
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing: {col}"


def test_load_season_has_target_col(season_dir):
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    assert TARGET_COL in df.columns


def test_load_season_empty_dir_returns_empty(tmp_path):
    from cpoe.ingest import load_season_pass_plays
    empty = tmp_path / "empty_season"
    empty.mkdir()
    df = load_season_pass_plays(empty)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_load_season_multiple_games(tmp_path: pathlib.Path):
    """Two game JSON files → combined DataFrame."""
    from cpoe.ingest import load_season_pass_plays
    _write_final_json(tmp_path, "401628455", 2024, _PLAYS_FIXTURE)
    _write_final_json(tmp_path, "401628456", 2024, _PLAYS_FIXTURE)
    df = load_season_pass_plays(tmp_path)
    assert len(df) == 4  # 2 pass plays × 2 games


def test_load_season_filters_by_season(tmp_path: pathlib.Path):
    """Season filter excludes games from other seasons."""
    from cpoe.ingest import load_season_pass_plays
    _write_final_json(tmp_path, "401628455", 2024, _PLAYS_FIXTURE)
    _write_final_json(tmp_path, "401620001", 2023, _PLAYS_FIXTURE)
    df = load_season_pass_plays(tmp_path, seasons=[2024])
    assert len(df) == 2  # only the 2024 game's 2 pass plays
