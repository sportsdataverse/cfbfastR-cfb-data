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
import re

import pandas as pd
import polars as pl

from .features import extract_pass_features

# The finals corpus is ~64 GB across ~20k files (median 3.4 MB), and a
# season-filtered train would otherwise json-parse every byte of it to read one
# top-level integer. The top-level "season" key lands around byte 2,130 in every
# file checked, so an 8 KB head read answers "could this file be in scope?" for
# ~400x less I/O.
#
# This is a NEGATIVE filter only, and deliberately so: a nested "season": YYYY
# elsewhere in the payload would match this regex too, so a hit proves nothing
# and every surviving file is still fully parsed and checked against its real
# top-level ``obj["season"]``. A MISS is the only trustworthy signal -- if none
# of the requested seasons appears in the head at all, the file cannot be one we
# want. A file whose head has no "season" key at all is parsed rather than
# skipped, so an unexpected layout degrades to the old behaviour instead of
# silently dropping games.
_SEASON_HEAD_BYTES = 8192
_SEASON_RE = re.compile(rb'"season"\s*:\s*(\d{4})')


def _cannot_be_in_scope(path: pathlib.Path, seasons: set[int]) -> bool:
    """True only when the file's head proves it holds none of ``seasons``."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SEASON_HEAD_BYTES)
    except OSError:
        return False  # unreadable head -> let the full parse decide
    found = {int(m.group(1)) for m in _SEASON_RE.finditer(head)}
    return bool(found) and not (found & seasons)


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
        if wanted is not None and _cannot_be_in_scope(f, wanted):
            continue
        obj = json.loads(f.read_text(encoding="utf-8"))
        # Authoritative check: the head scan above only ruled files OUT.
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
