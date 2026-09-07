"""Feature extraction for the CFB CP models (game-state + air-yards variants).

Input: a pandas or polars DataFrame with columns produced by
       CFBPlayProcess.run_processing_pipeline() (or an equivalent ESPN PBP frame
       with `start.*` dot-notation columns, including final.json plays where the
       play-type column is named ``type.text``).

Output: a pandas DataFrame containing whichever of AIR_YARDS_FEATURE_COLS are
        available plus TARGET_COL, one row per pass play (non-pass plays are
        filtered out).  The three air-yards columns are absent for pre-2025
        data and nullable where present, so a caller can select rows the
        air-yards model can score with ``df['air_yards'].notna()``.
"""
from __future__ import annotations

import pandas as pd

from .constants import AIR_YARDS_FEATURE_COLS, PASS_PLAY_TYPES, TARGET_COL

# Mapping from ESPN dot-notation / cfbfastR column names to flat feature names.
_COL_MAP: dict[str, str] = {
    "start.down": "down",
    "start.distance": "distance",
    "start.yardsToEndzone": "yards_to_goal",
    "pos_score_diff_start": "score_diff",
    "start.TimeSecsRem": "seconds_remaining",
    "start.is_home": "is_home",
    "period": "period",
    "passing_down": "passing_down",
}


def _play_type_col(df: pd.DataFrame) -> str:
    """Return whichever play-type column is present."""
    for c in ("type.text", "playType", "play_type", "type"):
        if c in df.columns:
            return c
    return ""


def extract_pass_features(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to pass plays and return the 8-feature matrix + target column.

    Args:
        df: Raw or processed PBP DataFrame.  Must contain columns matching
            the ESPN dot-notation names in ``_COL_MAP`` plus a play-type
            column and (optionally) a ``completion`` target column.

    Returns:
        pandas DataFrame with columns ``FEATURE_COLS + [TARGET_COL]``,
        reset index, dtypes coerced to float/int.  Empty if no pass plays
        or if input is empty.
    """
    # Accept polars DataFrames — convert to pandas so the rest of the pipeline
    # (rename, isin, astype) stays unchanged.
    try:
        import polars as pl  # type: ignore[import]
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
    except ImportError:
        pass

    if df.empty:
        return pd.DataFrame()

    # --- filter to pass plays ---
    pt_col = _play_type_col(df)
    if not pt_col:
        return pd.DataFrame()

    mask = df[pt_col].isin(PASS_PLAY_TYPES)
    plays = df[mask].copy()
    if plays.empty:
        return pd.DataFrame()

    # --- rename to flat feature names ---
    # Real final.json data has both flat columns (e.g. `down`) and their ESPN
    # dot-notation equivalents (e.g. `start.down`).  When both exist, drop the
    # flat column first so the rename doesn't produce duplicates that break
    # xgboost.DMatrix column indexing.
    cols_to_drop = [
        target
        for src, target in _COL_MAP.items()
        if src in plays.columns and target in plays.columns and src != target
    ]
    if cols_to_drop:
        plays = plays.drop(columns=cols_to_drop)
    plays = plays.rename(columns=_COL_MAP)

    # --- build target column (1 = completion) ---
    if "completion" not in plays.columns:
        plays["completion"] = (
            plays.get(pt_col, pd.Series(dtype=str))
            .str.contains("Reception|Passing Touchdown", na=False)
            .astype(int)
        )
    else:
        plays["completion"] = plays["completion"].astype(int)

    # --- coerce boolean columns to int ---
    for col in ("is_home", "passing_down"):
        if col in plays.columns:
            plays[col] = plays[col].astype(int)

    # --- air-yards features (Approach B), present only where ESPN's play text
    # carried a "caught at"/"thrown to" spot.  These stay NULLABLE on purpose:
    # a null means "this play has no air-yards data", which is a different thing
    # from "the throw went 0 yards" or "the QB was not hurried".  xgboost handles
    # NaN natively as a missing value and learns a default direction for it;
    # filling 0 would instead teach it that every pre-2025 play was a hurried
    # screen pass.
    if "pass_direction" in plays.columns and "pass_is_middle" not in plays.columns:
        direction = plays["pass_direction"]
        plays["pass_is_middle"] = (
            direction.eq("middle").where(direction.notna()).astype(float)
        )
    for col in ("air_yards", "qb_hurry", "pass_is_middle"):
        if col in plays.columns:
            plays[col] = pd.to_numeric(plays[col], errors="coerce").astype(float)

    # Preserve join keys + ``season`` if present so callers can rejoin scored
    # output back to the carry frame without a separate index round-trip.
    extra_passthrough = [c for c in ("game_id", "id", "season") if c in plays.columns]
    # AIR_YARDS_FEATURE_COLS is a superset of FEATURE_COLS, so one list covers
    # both variants; the trainer picks which subset it fits on.
    keep = [
        c for c in AIR_YARDS_FEATURE_COLS + [TARGET_COL] if c in plays.columns
    ] + extra_passthrough
    return plays[keep].reset_index(drop=True)
