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
