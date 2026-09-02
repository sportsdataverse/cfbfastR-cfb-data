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


# Athlete-id coverage gate (silent-no-op guard): _with_athlete_cols materializes NULL columns when
# a key is absent, so an upstream rename / re-scrape that dropped the ids would otherwise publish
# all-null id columns with no error. Ids ride along with names -- observed in pbp_full 2025:
# passer id on 42.2% of all plays vs name 43.1% -> id/name 0.979 (rusher 43.4/44.3 = 0.980,
# receiver 37.6/38.7 = 0.972); the 2-3% residue is regex-fallback names that carry no ESPN id.
# 2004 = 0.0 (ESPN shipped no passer ids before 2005), hence the season floor. Never lower to pass.
ATHLETE_ID_FLOOR = 0.9
ATHLETE_ID_GATE_FROM_SEASON = 2005


def athlete_id_coverage(df: pl.DataFrame) -> dict:
    """Newest season's non-null id / non-null name per role (a role with no names is unmeasurable)."""
    if df.is_empty() or "season" not in df.columns:
        return {}
    newest = int(df["season"].max())
    d = df.filter(pl.col("season") == newest)
    cov: dict = {"season": newest}
    for role in ("passer", "rusher", "receiver"):
        names = int(d[f"{role}_player_name"].is_not_null().sum())
        if names:
            cov[role] = round(int(d[f"{role}_player_id"].is_not_null().sum()) / names, 3)
    return cov


def check_athlete_ids(df: pl.DataFrame) -> dict:
    """Refuse the build when the newest season (>= 2005) carries names without ids."""
    cov = athlete_id_coverage(df)
    if cov and cov["season"] >= ATHLETE_ID_GATE_FROM_SEASON:
        low = {r: v for r, v in cov.items() if r != "season" and v < ATHLETE_ID_FLOOR}
        if low:
            raise ValueError(
                f"model_pbp REFUSED: athlete id coverage below {ATHLETE_ID_FLOOR} for season {cov['season']}: {low} "
                "-- ids are emitted upstream by sdv-py's participants module; an all-null id column means the key was dropped"
            )
    return cov


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
