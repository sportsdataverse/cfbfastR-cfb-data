import json

import numpy as np
import polars as pl
import pytest

from cfb_model_build.cfb_model_pbp.cli import build_parser, main
from cfb_model_build.cfb_model_pbp.schema import MODEL_PBP_COLUMNS


def test_parser_requires_out_and_cp_model():
    ns = build_parser().parse_args(["--final-dir", ".cache/cfb_final", "--cp-model", "m.ubj", "--out", "o.parquet"])
    assert ns.out == "o.parquet" and ns.cp_model == "m.ubj"


def test_contract_columns(tmp_path):
    """Offline contract test: main() produces a parquet whose columns satisfy the schema contract.

    Verifies that scored_date and all four *_version provenance columns are
    written, and that every output column is declared in MODEL_PBP_COLUMNS.
    Uses a tiny synthetic cache + locally-trained CP booster; no network calls.
    """
    import pandas as pd
    import xgboost as xgb
    from cfb_model_build.cpoe.constants import FEATURE_COLS
    from cfb_model_build.cpoe.train_cp import train_cp_model

    # --- build tiny synthetic cache dir ---
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    season = 2021
    game_id = 401012345

    # Two plays: one pass (completion=True), one non-pass.
    plays = [
        {
            "id": 1001,
            "game_id": game_id,
            "season": season,
            "week": 1,
            "period": 1,
            "sequenceNumber": 1,
            "game_play_number": 1,
            "drive.id": 10,
            "type.text": "Pass Reception",
            "text": "QB pass complete",
            "pass": True,
            "rush": False,
            "completion": True,
            "scoring_play": False,
            "statYardage": 8,
            "start.down": 1,
            "start.distance": 10,
            "start.yardsToEndzone": 65,
            "start.TimeSecsRem": 1800,
            "start.is_home": True,
            "pos_score_diff_start": 0,
            "passing_down": False,
            "EP_start": 1.5,
            "EP_end": 2.0,
            "EPA": 0.5,
            "wp_before": 0.55,
            "wp_after": 0.60,
            "wpa": 0.05,
            "pos_team": "TeamA",
            "def_pos_team": "TeamB",
            "start.pos_team.name": "TeamA",
            "homeTeamId": 1,
            "awayTeamId": 2,
            "homeTeamName": "TeamA",
            "awayTeamName": "TeamB",
            "passer_player_name": "QB1",
            "passer_player_id": 4690158,
            "receiver_player_name": "WR1",
            "receiver_player_id": 4690159,
        },
        {
            "id": 1002,
            "game_id": game_id,
            "season": season,
            "week": 1,
            "period": 1,
            "sequenceNumber": 2,
            "game_play_number": 2,
            "drive.id": 10,
            "type.text": "Rush",
            "text": "RB run for 3",
            "pass": False,
            "rush": True,
            "completion": False,
            "scoring_play": False,
            "statYardage": 3,
            "start.down": 2,
            "start.distance": 7,
            "start.yardsToEndzone": 57,
            "start.TimeSecsRem": 1770,
            "start.is_home": True,
            "pos_score_diff_start": 0,
            "passing_down": False,
            "EP_start": 2.0,
            "EP_end": 1.8,
            "EPA": -0.2,
            "wp_before": 0.60,
            "wp_after": 0.58,
            "wpa": -0.02,
            "pos_team": "TeamA",
            "def_pos_team": "TeamB",
            "start.pos_team.name": "TeamA",
            "homeTeamId": 1,
            "awayTeamId": 2,
            "homeTeamName": "TeamA",
            "awayTeamName": "TeamB",
            "passer_player_name": None,
            "passer_player_id": None,
            "rusher_player_name": "RB1",
            "rusher_player_id": 5083848,
        },
    ]

    game_file = cache_dir / f"{game_id}.json"
    game_file.write_text(json.dumps({"season": season, "plays": plays}), encoding="utf-8")

    # --- train a tiny CP booster ---
    rng = np.random.default_rng(0)
    n = 20
    X_train = pd.DataFrame({col: rng.integers(0, 5, size=n).tolist() for col in FEATURE_COLS})
    y_train = rng.integers(0, 2, size=n)
    booster = train_cp_model(X_train, y_train, nrounds=5, verbose_eval=False)
    model_path = tmp_path / "cp_test.ubj"
    booster.save_model(str(model_path))

    # --- run the CLI end-to-end ---
    out_path = tmp_path / "model_pbp.parquet"
    rc = main([
        "--final-dir", str(cache_dir),
        "--cp-model", str(model_path),
        "--out", str(out_path),
        "--seasons", str(season),
    ])
    assert rc == 0, "main() must return 0"
    assert out_path.exists(), "output parquet must be written"

    # --- verify schema contract ---
    df = pl.read_parquet(out_path)
    cols = set(df.columns)

    # Every output column must be in the declared contract.
    assert cols.issubset(set(MODEL_PBP_COLUMNS)), (
        f"columns not in contract: {cols - set(MODEL_PBP_COLUMNS)}"
    )

    # scored_date and all four version provenance columns must be present.
    required_provenance = {"scored_date", "model_pbp_version", "cp_model_version",
                           "ep_model_version", "wp_model_version"}
    assert required_provenance.issubset(cols), (
        f"missing provenance columns: {required_provenance - cols}"
    )

    # scored_date must be a non-null ISO date string.
    assert df["scored_date"][0] is not None


def test_refuses_when_athlete_ids_are_absent(tmp_path):
    """An upstream rename / re-scrape that drops the id keys must fail loudly, not publish all-null ids.

    Rush-only play so no CP booster is loaded; names present, id keys absent -> coverage 0 < 0.9.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    play = {"id": 1, "game_id": 5, "season": 2025, "week": 1, "period": 1, "sequenceNumber": 1,
            "game_play_number": 1, "drive.id": 1, "type.text": "Rush", "pass": False, "rush": True,
            "completion": False, "EP_start": 1.0, "EP_end": 1.2, "EPA": 0.2, "wp_before": 0.5,
            "wp_after": 0.5, "wpa": 0.0, "passer_player_name": "QB1", "rusher_player_name": "RB1"}
    (cache / "5.json").write_text(json.dumps({"season": 2025, "plays": [play]}), encoding="utf-8")
    out = tmp_path / "o.parquet"
    with pytest.raises(ValueError, match="athlete id coverage"):
        main(["--final-dir", str(cache), "--cp-model", "unused.ubj", "--out", str(out), "--seasons", "2025"])
    assert not out.exists()


def test_refuses_when_a_role_has_neither_ids_nor_names(tmp_path):
    """A rename that drops BOTH keys for a role the season HAS plays for must still fail the gate.

    _with_athlete_cols materializes the missing columns as all-null, so a role scored as
    "unmeasurable" (no names) used to slip past check_athlete_ids and publish an all-null id
    column. Zero names on an applicable role now scores 0.0 instead of being skipped; a role
    the frame has no plays for (no completion -> no receiver) is still skipped.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    # A rush play (so no CP booster is loaded): the rusher role APPLIES, but BOTH rusher keys
    # are absent -- names 0 and ids 0, which used to be scored "unmeasurable" and skipped.
    play = {"id": 1, "game_id": 5, "season": 2025, "week": 1, "period": 1, "sequenceNumber": 1,
            "game_play_number": 1, "drive.id": 1, "type.text": "Rush", "pass": False, "rush": True,
            "completion": False, "EP_start": 1.0, "EP_end": 1.2, "EPA": 0.2, "wp_before": 0.5,
            "wp_after": 0.5, "wpa": 0.0}
    (cache / "5.json").write_text(json.dumps({"season": 2025, "plays": [play]}), encoding="utf-8")
    out = tmp_path / "o.parquet"
    with pytest.raises(ValueError, match="athlete id coverage"):
        main(["--final-dir", str(cache), "--cp-model", "unused.ubj", "--out", str(out), "--seasons", "2025"])
    assert not out.exists()
