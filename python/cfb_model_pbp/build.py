from __future__ import annotations

from pathlib import Path

import polars as pl

from model_training.ingest import _read_final_plays
from .schema import CARRY_RENAME, DESCRIPTOR_COLS, IDENTITY_COLS

_REQUIRED_CARRY = list(CARRY_RENAME.keys())
_LAST = {"kept": 0, "dropped": 0}


def build_carry_frame(final_dir, seasons=None) -> pl.DataFrame:
    df = _read_final_plays(final_dir, seasons)
    if df.is_empty():
        return df
    # keep only rows that carry the EP/WP source columns (raw/pre-enrichment games lack them)
    present_required = [c for c in _REQUIRED_CARRY if c in df.columns]
    before = df.height
    if present_required:
        df = df.drop_nulls(subset=present_required)
    _LAST["kept"], _LAST["dropped"] = df.height, before - df.height
    df = df.rename({k: v for k, v in CARRY_RENAME.items() if k in df.columns})
    carry = [c for c in (IDENTITY_COLS + DESCRIPTOR_COLS + list(CARRY_RENAME.values())) if c in df.columns]
    return df.select(carry)


def last_completeness() -> dict:
    return dict(_LAST)
