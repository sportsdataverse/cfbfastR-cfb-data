import polars as pl
from cpoe.features import extract_pass_features


def test_extract_pass_features_reads_final_json_play_types():
    plays = pl.DataFrame([
        {"type.text": "Pass Completion", "completion": True, "start.down": 1, "start.distance": 10,
         "start.yardsToEndzone": 75, "pos_score_diff_start": 0, "start.TimeSecsRem": 1800,
         "start.is_home": True, "period": 1, "passing_down": False},
        {"type.text": "Rush", "completion": False, "start.down": 2, "start.distance": 8,
         "start.yardsToEndzone": 60, "pos_score_diff_start": 0, "start.TimeSecsRem": 1700,
         "start.is_home": True, "period": 1, "passing_down": False},
    ], infer_schema_length=None)
    feats = extract_pass_features(plays)
    assert len(feats) == 1  # only the pass play survives
