"""Backfill missing game spreads for the WP training frame from historical odds.

``wp_spread``'s single most important feature is ``spread_time`` (the pregame point
spread decayed over game time). For games where ESPN ships no closing line
(``gameSpreadAvailable == False`` or a null ``homeTeamSpread``), the CFBPlayProcess
pipeline falls back to a flat default (gameSpread 2.5 / OU 55.5), which injects a
constant, wrong signal into the WP model. The bulk of those games are 2006-2011.

This module reconstructs a consensus closing spread (and total) per game from
``cfbfastR-data/betting/parquet/cfb_line_odds.parquet`` (a 2006-2019 multi-book
line history) and rewrites ``homeTeamSpread`` / ``start.pos_team_spread`` /
``start.spread_time`` for the missing games only — leaving every game that already
carries a real ESPN line untouched.

Home/away resolution is crosswalk-free: the odds ``abbr`` codes don't match ESPN's,
so each abbr's canonical team name is inferred from its ``game_desc`` ("Away@Home")
co-occurrences (the team present in *all* of an abbr's games), then the home abbr is
the one whose name matches the post-``@`` token.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

# Match pbp_full EXACTLY (verified MAE 0.00000 against start.spread_time):
#   spread_time = pos_team_spread * exp(SPREAD_TIME_DECAY * elapsed_share)
# NOTE the sign: pbp_full uses +pos_team_spread (NOT the -1.0 * form that appears in
# sdv-py's fourth_down_decision._predict_wp, which is wrong-signed vs the trained data).
# Convention: ESPN pbp_full homeTeamSpread is POSITIVE when the home team is favored
# (verified: home-favored games mean homeTeamSpread = +9.39), which is the OPPOSITE of
# the standard betting-market sign — so the market consensus is negated to match.
SPREAD_TIME_DECAY: float = -4.0
SPREAD_CLIP: float = 60.0  # guard against contaminated multi-book outlier rows


def _abbr_to_name(spread: pl.DataFrame) -> pl.DataFrame:
    """Infer each odds ``abbr`` -> canonical team name from game_desc co-occurrences.

    The abbr's team appears in *every* game it plays, so across that abbr's rows the
    correct name is the modal token among {away_name, home_name}.
    """
    parts = spread.with_columns(
        away_name=pl.col("game_desc").str.split_exact("@", 1).struct.field("field_0").str.strip_chars(),
        home_name=pl.col("game_desc").str.split_exact("@", 1).struct.field("field_1").str.strip_chars(),
    ).select("abbr", "away_name", "home_name")
    stacked = pl.concat([
        parts.select("abbr", name=pl.col("away_name")),
        parts.select("abbr", name=pl.col("home_name")),
    ])
    # The abbr's own team appears in EVERY one of its games, so it is the modal name
    # across that abbr's rows (opponents vary). Needs multi-game support to resolve —
    # teams with a single game can tie; on the real 8k-game corpus this is unambiguous.
    # Tie-break by name for determinism.
    return (
        stacked.group_by("abbr", "name").agg(c=pl.len())
        .group_by("abbr")
        .agg(team_name=pl.col("name").sort_by(["c", "name"], descending=[True, False]).first())
    )


def load_consensus_spreads(odds_path: str | Path) -> pl.DataFrame:
    """Build per-game consensus home spread + total from the odds line history.

    Returns:
        polars DataFrame ``[game_id, home_spread, over_under]`` (one row per game
        with a resolvable home spread). ``home_spread`` is the negated median
        home-team line across books, i.e. ESPN pbp_full convention (POSITIVE ==
        home favored), clipped to +/-``SPREAD_CLIP``. ``over_under`` is the median
        total (null if the totals market is absent).
    """
    odds = pl.read_parquet(odds_path).filter(pl.col("game_id").is_not_null())
    spread = odds.filter((pl.col("market_type") == "spread") & pl.col("lines").is_not_null())
    a2n = _abbr_to_name(spread)
    home_name = pl.col("game_desc").str.split_exact("@", 1).struct.field("field_1").str.strip_chars()
    labelled = (
        spread.with_columns(home_name=home_name)
        .join(a2n, on="abbr", how="left")
        .with_columns(is_home=(pl.col("team_name") == pl.col("home_name")))
    )
    home_spread = (
        labelled.filter(pl.col("is_home"))
        .group_by("game_id")
        # Negate the market line into ESPN's pbp_full convention (positive == home favored).
        .agg(home_spread=-pl.col("lines").median())
        .with_columns(home_spread=pl.col("home_spread").clip(-SPREAD_CLIP, SPREAD_CLIP))
    )
    totals = (
        odds.filter((pl.col("market_type") == "total") & pl.col("lines").is_not_null())
        .group_by("game_id").agg(over_under=pl.col("lines").median())
    )
    return home_spread.join(totals, on="game_id", how="left").with_columns(
        pl.col("game_id").cast(pl.Int64)
    )


def _missing_mask() -> pl.Expr:
    """Games with no real ESPN spread: null homeTeamSpread, unavailable flag, or the
    injected (2.5, 55.5) default fingerprint."""
    return (
        pl.col("homeTeamSpread").is_null()
        | (pl.col("gameSpreadAvailable").cast(pl.Boolean) == False)  # noqa: E712
        | ((pl.col("gameSpread") == 2.5) & (pl.col("overUnder") == 55.5))
    )


def apply_spread_backfill(pbp: pl.DataFrame, spreads: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Rewrite spread fields + the WP ``spread_time`` feature for missing-spread games.

    Only rows whose game is BOTH missing a real spread AND present in ``spreads`` are
    changed; ``homeTeamSpread``, ``start.pos_team_spread`` and ``start.spread_time``
    are recomputed (``gameSpreadAvailable`` set True, ``overUnder`` filled when known).

    Returns:
        (backfilled_pbp, stats) where stats reports games/plays touched.
    """
    s = spreads.rename({"home_spread": "_bf_home_spread", "over_under": "_bf_ou"})
    out = pbp.join(s, on="game_id", how="left")
    fill = _missing_mask() & pl.col("_bf_home_spread").is_not_null()
    n_games = out.filter(fill).select(pl.col("game_id").n_unique()).item()
    n_plays = out.filter(fill).height

    hts = pl.when(fill).then(pl.col("_bf_home_spread")).otherwise(pl.col("homeTeamSpread"))
    pos_spread = pl.when(pl.col("start.is_home").cast(pl.Boolean) == True).then(hts).otherwise(-hts)  # noqa: E712
    spread_time = pos_spread * (SPREAD_TIME_DECAY * pl.col("start.elapsed_share")).exp()
    # ONE with_columns: every RHS expression reads the ORIGINAL frame, so `fill`,
    # `hts`, `pos_spread` and `spread_time` stay mutually consistent. Splitting these
    # across two with_columns calls would re-evaluate `fill` against the already-updated
    # homeTeamSpread/gameSpreadAvailable (flipping it False) and silently no-op the
    # feature columns the WP model actually consumes.
    out = out.with_columns(
        homeTeamSpread=hts,
        gameSpreadAvailable=pl.when(fill).then(True).otherwise(pl.col("gameSpreadAvailable")),
        overUnder=pl.when(fill & pl.col("_bf_ou").is_not_null()).then(pl.col("_bf_ou")).otherwise(pl.col("overUnder")),
        **{
            "start.pos_team_spread": pl.when(fill).then(pos_spread).otherwise(pl.col("start.pos_team_spread")),
            "start.spread_time": pl.when(fill).then(spread_time).otherwise(pl.col("start.spread_time")),
        },
    ).drop("_bf_home_spread", "_bf_ou")
    return out, {"games_backfilled": int(n_games), "plays_backfilled": int(n_plays)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="model_training.spread_backfill")
    ap.add_argument("--pbp", default="artifacts/pbp_full.parquet")
    ap.add_argument("--odds", default="../../cfbfastR-data/betting/parquet/cfb_line_odds.parquet")
    ap.add_argument("--out", default="artifacts/pbp_full_spreadfilled.parquet")
    ap.add_argument("--validate", action="store_true",
                    help="report consensus-vs-ESPN agreement on games that already have a spread")
    args = ap.parse_args(argv)

    spreads = load_consensus_spreads(args.odds)
    print(f"consensus spreads resolved for {spreads.height} games")

    if args.validate:
        pf = pl.read_parquet(args.pbp, columns=["game_id", "homeTeamSpread", "gameSpreadAvailable"]).unique("game_id")
        have = pf.filter((pl.col("gameSpreadAvailable").cast(pl.Boolean) == True) & pl.col("homeTeamSpread").is_not_null())  # noqa: E712
        j = have.join(spreads, on="game_id", how="inner")
        a, b = j["homeTeamSpread"].to_numpy(), j["home_spread"].to_numpy()
        print(f"  overlap with already-spread games: {j.height}")
        print(f"  corr(ESPN homeTeamSpread, consensus): {np.corrcoef(a, b)[0,1]:.4f}")
        print(f"  MAE: {np.mean(np.abs(a-b)):.2f} pts | sign agreement: {np.mean(np.sign(a)==np.sign(b)):.3f}")

    pbp = pl.read_parquet(args.pbp)
    filled, stats = apply_spread_backfill(pbp, spreads)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    filled.write_parquet(args.out)
    print(f"backfill: {stats} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
