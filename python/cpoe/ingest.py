"""Ingest final.json play files for CPOE training.

Expected on-disk layout (cfbfastR-cfb-raw scraper output):

    <final_dir>/
        <game_id>.json    ← one per game, produced by CFBPlayProcess;
                            each file has a top-level ``plays`` list and a
                            ``season`` integer field.

``load_season_pass_plays`` globs every ``*.json`` file under ``final_dir``,
optionally filters by season, extracts the ``plays`` list, and concatenates
into a single DataFrame ready for training or LOSO cross-validation.
"""
from __future__ import annotations

import json
import pathlib

import polars as pl

from .features import extract_pass_features


def load_season_pass_plays(
    final_dir: pathlib.Path | str,
    seasons: list[int] | None = None,
) -> pl.DataFrame:
    """Load and filter pass plays from all final.json files under ``final_dir``.

    Args:
        final_dir: Directory containing per-game ``*.json`` files whose
            top-level structure is ``{"season": <int>, "plays": [...]}``.
        seasons: Optional list of season integers to include.  If ``None``
            all games are included.

    Returns:
        pandas DataFrame with FEATURE_COLS + TARGET_COL columns.
        Empty (zero rows) DataFrame if no plays files are found or no pass
        plays survive the filter.
    """
    import pandas as pd

    frames: list[pl.DataFrame] = []
    for f in sorted(pathlib.Path(final_dir).glob("*.json")):
        obj = json.loads(f.read_text())
        if seasons is not None and obj.get("season") not in seasons:
            continue
        season = obj.get("season")
        plays = obj.get("plays") or []
        if plays:
            # Stamp season onto every play so extract_pass_features can pass it
            # through for LOSO CV season-splitting.
            for p in plays:
                p.setdefault("season", season)
            frames.append(pl.DataFrame(plays, infer_schema_length=None))

    if not frames:
        return pd.DataFrame()

    return extract_pass_features(pl.concat(frames, how="diagonal_relaxed"))
