import polars as pl
from cfb_data_ingest.schedule import season_game_ids


def test_season_game_ids_filters_by_season(tmp_path):
    p = tmp_path / "cfb_schedule_master.parquet"
    pl.DataFrame({"game_id": [1, 2, 3], "season": [2023, 2024, 2024]}).write_parquet(p)
    assert season_game_ids(p, [2024]) == [2, 3]
    assert season_game_ids(p, None) == [1, 2, 3]


def _master(p, rows):
    pl.DataFrame(
        {"game_id": [r[0] for r in rows], "season": [r[1] for r in rows],
         "status_type_completed": [r[2] for r in rows]}
    ).write_parquet(p)
    return p


def test_season_completed_games_counts_only_finished(tmp_path):
    """The signal that separates "not started yet" from "everything failed"."""
    from cfb_data_ingest.schedule import season_completed_games

    p = _master(tmp_path / "m.parquet", [
        (1, 2025, True), (2, 2025, True), (3, 2025, False),   # season underway
        (4, 2026, False), (5, 2026, False),                    # scheduled, no kickoff
    ])
    assert season_completed_games(p, 2026) == 0
    assert season_completed_games(p, 2025) == 2


def test_season_completed_games_treats_null_as_not_completed(tmp_path):
    """A null flag must not count as complete.

    Counting nulls would make a preseason season look started, which would put the
    zero-row guard back to raising in August -- the bug this exists to prevent.
    """
    from cfb_data_ingest.schedule import season_completed_games

    p = _master(tmp_path / "m.parquet", [(1, 2026, None), (2, 2026, None)])
    assert season_completed_games(p, 2026) == 0


def test_season_completed_games_unknown_season_is_zero(tmp_path):
    from cfb_data_ingest.schedule import season_completed_games

    p = _master(tmp_path / "m.parquet", [(1, 2025, True)])
    assert season_completed_games(p, 1999) == 0
