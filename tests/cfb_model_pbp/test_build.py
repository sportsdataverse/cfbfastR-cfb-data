import polars as pl
from cfb_model_build.cfb_model_pbp.build import build_carry_frame


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


def test_build_carry_pins_athlete_ids_int64_and_materializes_missing(tmp_path):
    """Ids ride through as Int64; a game that never ships the key still yields the column.

    Mirrors the real chain: ESPN ids are ints in final.json (e.g. 4690158), null where
    no participant was tagged, and entirely absent from some older games' plays.
    """
    import json
    base = {"sequenceNumber": 1, "game_play_number": 1, "drive.id": "d1", "week": 1, "period": 1,
            "EP_start": 2.0, "EP_end": 2.5, "EPA": 0.5, "wp_before": 0.5, "wp_after": 0.55, "wpa": 0.05,
            "type.text": "Pass Reception", "completion": True}
    with_ids = {"season": 2025, "plays": [
        {**base, "game_id": 1, "id": 100, "passer_player_name": "Noah Kim", "passer_player_id": 4690158,
         "rusher_player_name": None, "rusher_player_id": None,
         "receiver_player_name": "WR One", "receiver_player_id": 5295532},
        {**base, "game_id": 1, "id": 101, "passer_player_name": None, "passer_player_id": None,
         "rusher_player_name": "RB One", "rusher_player_id": 5083848,
         "receiver_player_name": None, "receiver_player_id": None},
    ]}
    without_keys = {"season": 2004, "plays": [{**base, "game_id": 2, "id": 200, "passer_player_name": "Old QB"}]}
    (tmp_path / "1.json").write_text(json.dumps(with_ids))
    (tmp_path / "2.json").write_text(json.dumps(without_keys))

    df = build_carry_frame(tmp_path)
    for c in ("passer_player_id", "rusher_player_id", "receiver_player_id"):
        assert df.schema[c] == pl.Int64, (c, df.schema[c])
    for c in ("rusher_player_name", "receiver_player_name"):
        assert df.schema[c] == pl.Utf8
    by_id = {r["id"]: r for r in df.sort("id").iter_rows(named=True)}
    assert by_id[100]["passer_player_id"] == 4690158 and by_id[100]["receiver_player_id"] == 5295532
    assert by_id[101]["passer_player_id"] is None and by_id[101]["rusher_player_id"] == 5083848
    assert by_id[200]["passer_player_id"] is None and by_id[200]["passer_player_name"] == "Old QB"
