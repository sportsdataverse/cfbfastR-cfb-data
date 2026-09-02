import json

import polars as pl

from cfb_model_build.cfb_model_pbp.build import build_carry_frame  # noqa: F401  (import order: cpoe deps)
from cfb_model_build.cpoe.constants import FEATURE_COLS as CP_FEATURES
from cfb_model_build.model_training import constants as C
from cfb_model_build.model_training.analysis import MODELS, build_frames, export_analysis_frames


def _frame() -> pl.DataFrame:
    """Four plays: pass, rush, punt (not xpass/cp), pass with a null down (dropped by xpass)."""
    base = {src: 1.0 for src in set(C.EP_SOURCE.values()) | set(C.WP_SOURCE.values())}
    base.update(
        {
            "game_id": 1,
            "season": 2024,
            "week": 3,
            "pos_team": "A",
            "start.pos_team.name": "A",
            "homeScore": 21,
            "homeTeamName": "A",
            "awayTeamName": "B",
            "awayScore": 17,
            "label": 0,
            "ScoreDiff_W": 0.5,
            "start.down": 1,
            "start.distance": 10,
            "start.yardsToEndzone": 60,
            "period": 1,
            "start.is_home": True,
            "passing_down": False,
            "completion": False,
        }
    )
    rows = [
        {**base, "id": 1, "type.text": "Pass Reception", "pass": True, "rush": False, "completion": True},
        {**base, "id": 2, "type.text": "Rush", "pass": False, "rush": True},
        {**base, "id": 3, "type.text": "Punt", "pass": False, "rush": False},
        {**base, "id": 4, "type.text": "Pass Incompletion", "pass": True, "rush": False, "start.down": None},
    ]
    return pl.DataFrame(rows)


def test_frames_carry_ids_then_the_trainer_feature_order_then_label():
    out = build_frames(_frame())
    assert set(out) == set(MODELS)
    ids = ["game_id", "id", "season", "week", "pos_team"]
    assert out["ep"].columns == ids + C.EP_FEATURES + ["label"]
    assert out["wp"].columns == ids + C.WP_SPREAD_FEATURES + ["label"]
    assert out["xpass"].columns == ids + C.XPASS_FEATURES + ["label"]
    assert out["cp"].columns == ["game_id", "id", "season"] + CP_FEATURES + ["label"]
    # unfiltered builders keep every play; xpass drops the punt + the null-down pass; cp keeps passes
    assert out["ep"].height == out["wp"].height == 4
    assert out["xpass"]["id"].to_list() == [1, 2] and out["xpass"]["label"].to_list() == [1, 0]
    assert sorted(out["cp"]["id"].to_list()) == [1, 4] and out["cp"]["label"].sum() == 1
    # the naive WP set is a strict subset of the exported columns, so one frame serves both
    assert set(C.WP_NAIVE_FEATURES) <= set(out["wp"].columns) and "spread_time" in out["wp"].columns


def test_export_writes_one_parquet_per_model_and_a_manifest(tmp_path):
    src = tmp_path / "pbp.parquet"
    _frame().with_columns(pl.col("id").cast(pl.Utf8)).write_parquet(src)  # pbp_full ships String ids
    rows = export_analysis_frames(src, tmp_path / "analysis", ("wp", "xpass"))
    assert rows == {"wp": 4, "xpass": 2}
    assert sorted(p.name for p in (tmp_path / "analysis").glob("*.parquet")) == [
        "analysis_wp.parquet",
        "analysis_xpass.parquet",
    ]
    m = json.loads((tmp_path / "analysis" / "analysis_manifest.json").read_text())
    assert m["source_frame"] == str(src)
    assert m["models"]["wp"]["features"] == C.WP_SPREAD_FEATURES and m["models"]["wp"]["n_rows"] == 4
    xp = pl.read_parquet(tmp_path / "analysis" / "analysis_xpass.parquet")
    assert xp.columns[-1] == "label" and xp.schema["id"] == pl.Int64  # pinned at the boundary
