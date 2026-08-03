import polars as pl
from cfb_data_ingest.schedule import season_game_ids


def test_season_game_ids_filters_by_season(tmp_path):
    p = tmp_path / "cfb_schedule_master.parquet"
    pl.DataFrame({"game_id": [1, 2, 3], "season": [2023, 2024, 2024]}).write_parquet(p)
    assert season_game_ids(p, [2024]) == [2, 3]
    assert season_game_ids(p, None) == [1, 2, 3]
