"""Select/rename final.json plays into the exact shipped model input matrices.

Returns pandas DataFrames (xgboost.DMatrix-friendly) with columns in the EXACT shipped
order, plus the label and weight arrays. WP label is win_indicator =
(start.pos_team.name == winner), i.e. the posteam NAME compared to the game winner;
no sample weights for WP (per the cfbscrapR-wpa recipe). EP uses ScoreDiff_W weights.
"""
from __future__ import annotations

import polars as pl

from . import constants as C


def _select(df: pl.DataFrame, source: dict[str, str]):
    out = df.select([pl.col(src).alias(name) for name, src in source.items()])
    return out.to_pandas()


def ep_matrix(df: pl.DataFrame):
    X = _select(df, C.EP_SOURCE)[C.EP_FEATURES]
    y = df["label"].to_numpy()
    w = df["ScoreDiff_W"].to_numpy()
    return X, y, w


def wp_matrix(df: pl.DataFrame, variant: str = "spread"):
    if variant == "spread":
        feats = C.WP_SPREAD_FEATURES
    elif variant == "naive":
        feats = C.WP_NAIVE_FEATURES
    else:
        raise ValueError(f"Unknown WP variant: {variant!r} (expected 'spread' or 'naive')")
    source = {k: v for k, v in C.WP_SOURCE.items() if k in feats}
    X = _select(df, source)[feats]
    y = (df["start.pos_team.name"] == df["winner"]).cast(pl.Int32).to_numpy()
    return X, y, None


def _era(season_col: str = "season") -> pl.Expr:
    """Ordinal CFB rule-era factor derived from the play's season (0..3).

    Cuts (shared with the fourth_down model): <=2006 -> 0, <=2013 -> 1,
    <=2017 -> 2, else 3.
    """
    lo, mid, hi = C.ERA_BOUNDS
    return (
        pl.when(pl.col(season_col) <= lo).then(0)
        .when(pl.col(season_col) <= mid).then(1)
        .when(pl.col(season_col) <= hi).then(2)
        .otherwise(3)
        .cast(pl.Int32)
    )


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


def fg_matrix(df: pl.DataFrame):
    """FG make-probability matrix: filter fg_attempt & yards_to_goal in [1,55].

    Returns (X[1 feat], y=fg_made int, None) — single-feature surface, no weights.
    """
    f = df.filter(
        (pl.col("fg_attempt") == True)  # noqa: E712
        & pl.col("start.yardsToEndzone").is_between(1, 55)
    )
    X = f.select(pl.col("start.yardsToEndzone").alias("yards_to_goal"))[C.FG_FEATURES].to_pandas()
    y = f["fg_made"].cast(pl.Int32).to_numpy()
    return X, y, None


def xpass_matrix(df: pl.DataFrame):
    """xPass (pass-vs-rush) matrix: rush|pass plays with non-null down/distance/ytg.

    Returns (X[7 feats, ordered], y=pass int, None).
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
    X = f.select(C.XPASS_FEATURES).to_pandas()
    y = f["pass"].cast(pl.Int32).to_numpy()
    return X, y, None


def two_pt_matrix(df: pl.DataFrame):
    """Two-point-conversion matrix: filter result in {success, failure}.

    Returns (X[4 feats, ordered], y=(result=='success') int, None).
    """
    f = df.filter(pl.col("two_point_conv_result").is_in(["success", "failure"])).with_columns(
        posteam_spread=pl.col("start.pos_team_spread"),
        posteam_total=_posteam_total(),
        pos_score_diff=pl.col("pos_score_diff_start"),
        era=_era("season"),
    )
    X = f.select(C.TWO_PT_FEATURES).to_pandas()
    y = (f["two_point_conv_result"] == "success").cast(pl.Int32).to_numpy()
    return X, y, None


def qbr_matrix(df: pl.DataFrame):
    """Per-(passer, game) weighted means of the 6 qbr_vars (mirrors CFBPlayProcess __process_qbr).

    `spread` is the posteam-perspective game spread. On final.json there is no flat `spread`
    column, but `start.pos_team_spread` IS the posteam spread -> alias it when `spread` is absent.
    Returns (X features, None, keys); the ESPN-QBR target is merged later in train_qbr.
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
    X = g.select(["qbr_epa", "sack_epa", "pass_epa", "rush_epa", "pen_epa", "spread"]).to_pandas()
    keys = g.select(["game_id", "season", "passer_player_name"]).to_pandas()
    return X, None, keys
