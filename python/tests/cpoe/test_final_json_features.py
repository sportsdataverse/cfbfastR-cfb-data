import polars as pl
from cpoe.features import extract_pass_features


def test_extract_pass_features_reads_final_json_play_types():
    plays = pl.DataFrame([
        {
            "gameId": "401628455",
            "type.text": "Pass Incompletion",
            "start.down": 3,
            "start.distance": 8,
            "start.yardsToEndzone": 30,
            "start.is_home": False,
            "pass_direction": "left",
            "qb_hurry": True,
            "air_yards": 15,
            "completion": 0,
        },
        {
            "gameId": "401628455",
            "type.text": "Rush",  # non-pass — should be filtered out
            "start.down": 1,
            "start.distance": 10,
            "start.yardsToEndzone": 50,
            "start.is_home": True,
            "pass_direction": "middle",
            "qb_hurry": True,
            "air_yards": 50,
            "completion": 0,
        },
    ], infer_schema_length=None)
    feats = extract_pass_features(plays)
    assert len(feats) == 1  # only the pass play survives
