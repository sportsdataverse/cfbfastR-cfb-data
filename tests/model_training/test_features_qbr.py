import polars as pl
from cfb_model_build.model_training.ingest import add_winner


def test_add_winner_from_final_scores():
    df = pl.DataFrame({
        "game_id": [1, 1], "homeTeamName": ["A", "A"], "awayTeamName": ["B", "B"],
        "homeScore": [28, 28], "awayScore": [10, 10], "is_home": [1, 0],
        "pos_team": ["A", "B"],
    })
    out = add_winner(df)
    assert out["winner"].to_list() == ["A", "A"]
