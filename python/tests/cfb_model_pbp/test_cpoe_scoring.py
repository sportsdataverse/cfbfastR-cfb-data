import polars as pl
from cfb_model_pbp.build import score_cpoe


def test_score_cpoe_appends_completion_prob_and_cpoe():
    carry = pl.DataFrame({"game_id": [1, 1], "id": [100, 101], "completion": [True, None], "pass": [True, False]})
    plays = pl.DataFrame({"game_id": [1, 1], "id": [100, 101], "completion": [True, None],
                          "type.text": ["Pass Completion", "Rush"], "start.down": [1, 2],
                          "start.distance": [10, 8], "start.yardsToEndzone": [75, 60],
                          "pos_score_diff_start": [0, 0], "start.TimeSecsRem": [1800, 1700],
                          "start.is_home": [True, True], "period": [1, 1], "passing_down": [False, False]},
                         infer_schema_length=None)
    out = score_cpoe(carry, plays, cp_model_path=None, _predict=lambda X: [0.6])  # 1 pass row -> cp 0.6
    pass_row = out.filter(pl.col("id") == 100).row(0, named=True)
    assert abs(pass_row["completion_prob"] - 0.6) < 1e-9
    assert abs(pass_row["cpoe"] - (1.0 - 0.6)) < 1e-9
    assert out.filter(pl.col("id") == 101).row(0, named=True)["completion_prob"] is None
