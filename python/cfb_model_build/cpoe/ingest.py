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

import pandas as pd
import polars as pl

from .features import extract_pass_features

# NOTE on cost: this reads the whole finals corpus (~64 GB across ~20k files,
# median 3.4 MB) even when training on two seasons, because the only way to
# learn a file's season is to parse it.
#
# A head-read pre-filter was tried and removed: the top-level key order is
# ``gameId, plays, season, ...``, so the file's own "season" sits after the
# entire plays array -- megabytes in, unreachable by any sane head window. The
# "season" that IS near the top of the file belongs to the first PLAY (every
# play carries its own), and using a nested value as proof of the file's season
# is only accidentally right. Getting it wrong would silently shrink the
# training set while still reporting success, which is worse than being slow.
#
# If retrain cost becomes a problem, the fix is a cached {file -> season} index
# built from real parses (keyed on mtime+size), not a cleverer scan of the head.


def load_season_pass_plays(
    final_dir: pathlib.Path | str,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
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
    frames: list[pl.DataFrame] = []
    wanted = set(seasons) if seasons is not None else None
    for f in sorted(pathlib.Path(final_dir).glob("*.json")):
        obj = json.loads(f.read_text(encoding="utf-8"))
        if wanted is not None and obj.get("season") not in wanted:
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
