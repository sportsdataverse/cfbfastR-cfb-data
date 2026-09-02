from __future__ import annotations

import polars as pl

from cfb_model_build.model_training.ingest import _read_final_plays
from .schema import ATHLETE_ID_COLS, ATHLETE_NAME_COLS, CARRY_RENAME, DESCRIPTOR_COLS, IDENTITY_COLS

_REQUIRED_CARRY = list(CARRY_RENAME.keys())
_LAST = {"kept": 0, "dropped": 0}


def _with_athlete_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Pin the athlete columns' dtypes and materialize the ones a season's finals lack.

    A season whose finals never carry a key would otherwise publish a parquet WITHOUT the
    column, and a cross-season concat would fail on schema; a game whose ids are all
    null infers ``Null`` and would concat to ``Null``/``String``. Cast is strict on purpose:
    a non-numeric id is a parser regression to surface, not to paper over.
    """
    return df.with_columns(
        [pl.col(c).cast(pl.Int64) if c in df.columns else pl.lit(None, dtype=pl.Int64).alias(c)
         for c in ATHLETE_ID_COLS]
        + [pl.col(c).cast(pl.Utf8) if c in df.columns else pl.lit(None, dtype=pl.Utf8).alias(c)
           for c in ATHLETE_NAME_COLS]
    )


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
    df = _with_athlete_cols(df.rename({k: v for k, v in CARRY_RENAME.items() if k in df.columns}))
    carry = [c for c in (IDENTITY_COLS + DESCRIPTOR_COLS + list(CARRY_RENAME.values())) if c in df.columns]
    return df.select(carry)


def last_completeness() -> dict:
    return dict(_LAST)


def score_cpoe(carry_df: pl.DataFrame, plays_df: pl.DataFrame, cp_model_path, _predict=None) -> pl.DataFrame:
    """Append completion_prob + cpoe (pass plays only) to carry_df, joined on (game_id, id)."""
    from cfb_model_build.cpoe.features import extract_pass_features
    feats = extract_pass_features(plays_df)  # pass rows only, with id retained
    if feats.empty:
        return carry_df.with_columns(completion_prob=pl.lit(None, dtype=pl.Float64),
                                     cpoe=pl.lit(None, dtype=pl.Float64))
    if _predict is None:
        import numpy as np
        import xgboost as xgb
        from cfb_model_build.cpoe.constants import FEATURE_COLS
        booster = xgb.Booster(); booster.load_model(str(cp_model_path))
        preds = booster.predict(xgb.DMatrix(feats[FEATURE_COLS]))
        preds = np.asarray(preds).tolist()
    else:
        preds = _predict(feats)
    feats_pl = pl.from_pandas(feats)
    scored = feats_pl.select("game_id", "id", "completion").with_columns(
        completion_prob=pl.Series("completion_prob", preds, dtype=pl.Float64),
    ).with_columns(cpoe=(pl.col("completion").cast(pl.Float64) - pl.col("completion_prob")))
    return carry_df.join(scored.select("game_id", "id", "completion_prob", "cpoe"),
                         on=["game_id", "id"], how="left")
