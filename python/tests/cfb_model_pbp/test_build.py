import polars as pl
from cfb_model_pbp.build import build_carry_frame


def test_build_carry_renames_and_keeps_keys(tmp_path):
    import json
    game = {"season": 2024, "plays": [{
        "game_id": 1, "id": 100, "sequenceNumber": 1, "game_play_number": 1, "drive.id": "d1",
        "week": 1, "period": 1, "EP_start": 2.0, "EP_end": 2.5, "EPA": 0.5,
        "wp_before": 0.5, "wp_after": 0.55, "wpa": 0.05, "type.text": "Rush", "completion": False,
    }]}
    (tmp_path / "1.json").write_text(json.dumps(game))
    df = build_carry_frame(tmp_path, seasons=[2024])
    assert {"game_id", "id", "epa", "wp_after", "ep_before"} <= set(df.columns)
    row = df.row(0, named=True)
    assert row["epa"] == 0.5 and row["ep_before"] == 2.0 and row["wp_after"] == 0.55
