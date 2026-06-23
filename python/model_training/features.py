"""Select/rename final.json plays into the exact shipped model input matrices.

Returns pandas DataFrames (xgboost.DMatrix-friendly) with columns in the EXACT shipped
order, plus the label and weight arrays. WP label is win_indicator =
(start.pos_team.name == winner), i.e. the posteam NAME compared to the game winner;
no sample weights for WP (per the cfbscrapR-wpa recipe). EP uses ScoreDiff_W weights.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from . import constants as C


def _select(df: pl.DataFrame, source: dict[str, str]):
    out = df.select([pl.col(src).alias(name) for name, src in source.items()])
    return out.to_pandas()


def _append_era_onehot(X: pd.DataFrame, df: pl.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Concat row-aligned era0..era3 dummies onto ``X`` and reorder to ``feats``+era.

    ``df`` must be the SAME (already-filtered) frame ``X`` was built from so the
    one-hot columns line up row-for-row.
    """
    ed = df.select(_era_onehot("season")).to_pandas()
    out = pd.concat([X.reset_index(drop=True), ed.reset_index(drop=True)], axis=1)
    return out[C.with_era_onehot(feats)]


def ep_matrix(df: pl.DataFrame, *, era_onehot: bool = False):
    X = _select(df, C.EP_SOURCE)[C.EP_FEATURES]
    if era_onehot:
        X = _append_era_onehot(X, df, C.EP_FEATURES)
    y = df["label"].to_numpy()
    w = df["ScoreDiff_W"].to_numpy()
    return X, y, w


def wp_matrix(df: pl.DataFrame, variant: str = "spread", *, era_onehot: bool = False):
    if variant == "spread":
        feats = C.WP_SPREAD_FEATURES
    elif variant == "naive":
        feats = C.WP_NAIVE_FEATURES
    else:
        raise ValueError(f"Unknown WP variant: {variant!r} (expected 'spread' or 'naive')")
    source = {k: v for k, v in C.WP_SOURCE.items() if k in feats}
    X = _select(df, source)[feats]
    if era_onehot:
        X = _append_era_onehot(X, df, feats)
    y = (df["start.pos_team.name"] == df["winner"]).cast(pl.Int32).to_numpy()
    return X, y, None


def _era(season_col: str = "season") -> pl.Expr:
    """Ordinal CFB rule-era factor derived from the play's season (0..3).

    Cuts (shared with the fourth_down model): <=2006 -> 0, <=2013 -> 1,
    <=2020 -> 2, else 3 (2004-2006 / 2007-2013 / 2014-2020 / 2021+).
    """
    lo, mid, hi = C.ERA_BOUNDS
    return (
        pl.when(pl.col(season_col) <= lo).then(0)
        .when(pl.col(season_col) <= mid).then(1)
        .when(pl.col(season_col) <= hi).then(2)
        .otherwise(3)
        .cast(pl.Int32)
    )


def _era_onehot(season_col: str = "season") -> list[pl.Expr]:
    """One-hot CFB rule-era dummies (era0..era3) derived from the play's season.

    Same ERA_BOUNDS cuts as ``_era`` but emitted as 0/1 indicator columns
    (nflfastR-style) — the encoding evaluated by ``era_experiment.py``. Returns one
    polars expression per bucket, aliased ``era0``..``era3``.
    """
    bounds = C.ERA_BOUNDS
    exprs: list[pl.Expr] = []
    prev: int | None = None
    for i, hi in enumerate((*bounds, None)):
        s = pl.col(season_col)
        if prev is None and hi is not None:
            cond = s <= hi
        elif hi is None:
            cond = s > prev
        else:
            cond = (s > prev) & (s <= hi)
        exprs.append(cond.cast(pl.Int32).alias(f"era{i}"))
        prev = hi
    return exprs


def _posteam_total() -> pl.Expr:
    """Possessing-team game total: (homeTeamSpread+overUnder)/2 if posteam is home,
    else (overUnder-homeTeamSpread)/2."""
    home_total = (pl.col("homeTeamSpread") + pl.col("overUnder")) / 2.0
    away_total = (pl.col("overUnder") - pl.col("homeTeamSpread")) / 2.0
    return (
        pl.when(pl.col("start.is_home").cast(pl.Boolean) == True)  # noqa: E712
        .then(home_total)
        .otherwise(away_total)
    )


def fg_matrix(df: pl.DataFrame, *, era_onehot: bool = False):
    """FG make-probability matrix: filter fg_attempt & yards_to_goal in [1,55].

    Returns (X[1 feat], y=fg_made int, None) — single-feature surface, no weights.
    With ``era_onehot`` the era0..era3 dummies are appended (FG has no era today).
    """
    f = df.filter(
        (pl.col("fg_attempt") == True)  # noqa: E712
        & pl.col("start.yardsToEndzone").is_between(1, 55)
    )
    X = f.select(pl.col("start.yardsToEndzone").alias("yards_to_goal"))[C.FG_FEATURES].to_pandas()
    if era_onehot:
        X = _append_era_onehot(X, f, C.FG_FEATURES)
    y = f["fg_made"].cast(pl.Int32).to_numpy()
    return X, y, None


def xpass_matrix(df: pl.DataFrame, *, era_onehot: bool = False):
    """xPass (pass-vs-rush) matrix: rush|pass plays with non-null down/distance/ytg.

    Returns (X[7 feats, ordered], y=pass int, None). With ``era_onehot`` the ordinal
    ``era`` factor is replaced by the era0..era3 dummies.
    """
    f = df.filter(
        ((pl.col("rush") == True) | (pl.col("pass") == True))  # noqa: E712
        & pl.col("start.down").is_not_null()
        & pl.col("start.distance").is_not_null()
        & pl.col("start.yardsToEndzone").is_not_null()
    ).with_columns(
        down=pl.col("start.down"),
        distance=pl.col("start.distance"),
        yards_to_goal=pl.col("start.yardsToEndzone"),
        pos_score_diff=pl.col("pos_score_diff_start"),
        TimeSecsRem=pl.col("start.TimeSecsRem"),
        era=_era("season"),
        period=pl.col("period"),
    )
    if era_onehot:
        f = f.with_columns(_era_onehot("season"))
        cols = C.with_era_onehot(C.XPASS_FEATURES)
    else:
        cols = C.XPASS_FEATURES
    X = f.select(cols).to_pandas()
    y = f["pass"].cast(pl.Int32).to_numpy()
    return X, y, None


def two_pt_matrix(df: pl.DataFrame, *, era_onehot: bool = False):
    """Two-point-conversion matrix: filter result in {success, failure}.

    Returns (X[4 feats, ordered], y=(result=='success') int, None). With
    ``era_onehot`` the ordinal ``era`` factor is replaced by the era0..era3 dummies.
    """
    f = df.filter(pl.col("two_point_conv_result").is_in(["success", "failure"])).with_columns(
        posteam_spread=pl.col("start.pos_team_spread"),
        posteam_total=_posteam_total(),
        pos_score_diff=pl.col("pos_score_diff_start"),
        era=_era("season"),
    )
    if era_onehot:
        f = f.with_columns(_era_onehot("season"))
        cols = C.with_era_onehot(C.TWO_PT_FEATURES)
    else:
        cols = C.TWO_PT_FEATURES
    X = f.select(cols).to_pandas()
    y = (f["two_point_conv_result"] == "success").cast(pl.Int32).to_numpy()
    return X, y, None


def qbr_matrix(df: pl.DataFrame, *, era_onehot: bool = False):
    """Per-(passer, game) weighted means of the 6 qbr_vars (mirrors CFBPlayProcess __process_qbr).

    `spread` is the posteam-perspective game spread. On final.json there is no flat `spread`
    column, but `start.pos_team_spread` IS the posteam spread -> alias it when `spread` is absent.
    Returns (X features, None, keys); the ESPN-QBR target is merged later in train_qbr.
    With ``era_onehot`` the era0..era3 dummies are appended (QBR has no era today).
    """
    if "spread" not in df.columns and "start.pos_team_spread" in df.columns:
        df = df.with_columns(spread=pl.col("start.pos_team_spread"))
    g = (
        df.filter(pl.col("passer_player_name").is_not_null())
        .group_by(["game_id", "season", "passer_player_name"])
        .agg(
            qbr_epa=(pl.col("qbr_epa") * pl.col("weight")).sum() / pl.col("weight").sum(),
            sack_epa=(pl.col("sack_epa") * pl.col("sack_weight")).sum() / pl.col("sack_weight").sum(),
            pass_epa=(pl.col("pass_epa") * pl.col("pass_weight")).sum() / pl.col("pass_weight").sum(),
            rush_epa=(pl.col("rush_epa") * pl.col("rush_weight")).sum() / pl.col("rush_weight").sum(),
            pen_epa=(pl.col("pen_epa") * pl.col("pen_weight")).sum() / pl.col("pen_weight").sum(),
            spread=pl.col("spread").first(),
        )
        .with_columns(pl.col(["sack_epa", "pass_epa", "rush_epa", "pen_epa"]).fill_null(0.0))
    )
    base = ["qbr_epa", "sack_epa", "pass_epa", "rush_epa", "pen_epa", "spread"]
    if era_onehot:
        g = g.with_columns(_era_onehot("season"))
        X = g.select(base + C.ERA_ONEHOT_COLS).to_pandas()
    else:
        X = g.select(base).to_pandas()
    keys = g.select(["game_id", "season", "passer_player_name"]).to_pandas()
    return X, None, keys
