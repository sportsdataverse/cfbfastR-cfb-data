"""The market as a ceiling, not a feature.

A closing spread is the best publicly available forecast of a game. It belongs
in the harness as the upper reference point -- "MAE 13.3" means nothing until
you know whether the attainable number is 10 or 13.

Deliberately NOT a model input. Feeding the market into a predictive head
produces a model that looks excellent and has learned nothing about football;
it also cannot be run before a line exists, which is when a preview surface
needs it. Kept separate so that boundary is structural, not a matter of
remembering.

Schema notes (``load_cfb_betting_lines``): long format, one row per
(game, market_type, team, book), everything typed as strings.
``market_type`` in {spread, total, money_line}; ``lines`` holds the number;
``abbr`` names the team the line belongs to -- which needs a crosswalk to
resolve against ``home_team_id``.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _pl(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _implied(odds: np.ndarray) -> np.ndarray:
    """American odds -> implied probability (still vigged)."""
    return np.where(odds > 0, 100.0 / (odds + 100.0), -odds / (-odds + 100.0))


def market_frame(seasons: list[int]) -> pl.DataFrame:
    """One row per game: median closing home spread + de-vigged home win prob.

    Returns columns ``game_id``, ``mkt_margin`` (positive = home favoured),
    ``mkt_wp``. Games without a usable line are absent, not null-filled.
    """
    from sportsdataverse.cfb import load_cfb_betting_lines, load_cfb_team_info

    bl = _pl(load_cfb_betting_lines(seasons))
    info = _pl(load_cfb_team_info(seasons))

    # abbr -> team_id. Abbreviations are not unique across all of history, so
    # keep it season-scoped where the info frame allows.
    abbr_col = next(
        (c for c in ("abbreviation", "team_abbreviation", "abbr") if c in info.columns),
        None,
    )
    id_col = next((c for c in ("team_id", "id") if c in info.columns), None)
    if not abbr_col or not id_col:
        raise ValueError(f"no abbr/id columns in team_info: {info.columns[:20]}")
    xwalk = (
        info.select(
            pl.col(abbr_col).cast(pl.Utf8).alias("abbr"),
            pl.col(id_col).cast(pl.Int64).alias("tid"),
        )
        .drop_nulls()
        .unique(subset=["abbr"], keep="first")
    )

    base = bl.select(
        pl.col("game_id").cast(pl.Int64),
        pl.col("market_type").cast(pl.Utf8),
        pl.col("abbr").cast(pl.Utf8),
        pl.col("lines").cast(pl.Float64, strict=False),
        pl.col("odds").cast(pl.Float64, strict=False),
        pl.col("home_team_id").cast(pl.Int64, strict=False),
        pl.col("away_team_id").cast(pl.Int64, strict=False),
    ).join(xwalk, on="abbr", how="inner")

    is_home = pl.col("tid") == pl.col("home_team_id")

    # Spread: ESPN convention is negative = that team favoured, so the home
    # team's line negated is the predicted home margin.
    spread = (
        base.filter(
            (pl.col("market_type") == "spread")
            & is_home
            & pl.col("lines").is_not_null()
        )
        .group_by("game_id")
        .agg((-pl.col("lines").median()).alias("mkt_margin"))
    )

    # Money line: de-vig by normalising the two implied probabilities.
    ml = base.filter(
        (pl.col("market_type") == "money_line") & pl.col("odds").is_not_null()
    )
    if ml.height:
        ml = ml.with_columns(
            pl.col("odds")
            .map_batches(
                lambda s: pl.Series(_implied(s.to_numpy())), return_dtype=pl.Float64
            )
            .alias("p"),
            is_home.alias("is_home"),
        )
        agg = ml.group_by(["game_id", "is_home"]).agg(pl.col("p").median())
        wide = agg.pivot(values="p", index="game_id", on="is_home")
        cols = [c for c in wide.columns if c != "game_id"]
        if len(cols) == 2:
            h = "true" if "true" in cols else cols[-1]
            a = [c for c in cols if c != h][0]
            wide = wide.select(
                "game_id",
                (pl.col(h) / (pl.col(h) + pl.col(a))).alias("mkt_wp"),
            ).drop_nulls()
            return spread.join(wide, on="game_id", how="full", coalesce=True)
    return spread.with_columns(pl.lit(None, pl.Float64).alias("mkt_wp"))


def ceiling(frame: pl.DataFrame, seasons: list[int]):
    """Score the market on the same games a head was scored on."""
    from .metrics import evaluate

    mk = market_frame(seasons)
    j = (
        frame.select("game_id", "margin", "home_won")
        .join(mk, on="game_id", how="inner")
        .filter(pl.col("mkt_margin").is_not_null())
    )
    if not j.height:
        return None, 0
    return (
        evaluate(
            "MARKET closing line (ceiling, not a head)",
            boundary="market",
            pred_margin=j["mkt_margin"].to_numpy(),
            actual_margin=j["margin"].to_numpy(),
            prob=j["mkt_wp"].to_numpy() if j["mkt_wp"].null_count() == 0 else None,
            won=j["home_won"].to_numpy(),
        ),
        j.height,
    )
