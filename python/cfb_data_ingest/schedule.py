from __future__ import annotations

from pathlib import Path

import polars as pl

from . import RAW_BASE

SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule_master.parquet"


def season_game_ids(schedule_path_or_url: str | Path | None, seasons: list[int] | None) -> list[int]:
    """Return game_id values from the schedule master, optionally filtered by season."""
    src = str(schedule_path_or_url) if schedule_path_or_url is not None else SCHEDULE_URL
    lf = pl.scan_parquet(src).select("game_id", "season")
    if seasons is not None:
        lf = lf.filter(pl.col("season").is_in(seasons))
    return lf.collect().get_column("game_id").cast(pl.Int64).to_list()


def season_completed_games(schedule_path_or_url: str | Path | None, season: int) -> int:
    """How many games of ``season`` ESPN has marked complete.

    Lets a builder tell "the season has not started" apart from "every fetch
    failed" -- two states that both compile to zero rows but mean opposite things.
    A zero-row guard without this distinction fires every August on a correct
    result, and a build that is red all preseason is a build whose next real
    failure nobody notices.
    """
    src = str(schedule_path_or_url) if schedule_path_or_url is not None else SCHEDULE_URL
    done = (
        pl.scan_parquet(src)
        .select("season", "status_type_completed")
        .filter(pl.col("season") == season)
        .collect()
        .get_column("status_type_completed")
    )
    return int(done.fill_null(False).sum() or 0)
