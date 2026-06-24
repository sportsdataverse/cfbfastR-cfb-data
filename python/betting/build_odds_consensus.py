"""Precompute a per-game ``odds_override`` lookup from cfb_line_odds.

The CFB reprocess (``cfbfastR-cfb-raw/python/reprocess_cfb_json.py``) feeds a
pregame spread/total into ``CFBPlayProcess`` as EPA/WPA inputs. The multi-book
consensus in ``cfb_line_odds`` is a better source than ESPN's single pickcenter,
so this script reduces the line history to one ``odds_override`` per game and
writes it where the reprocess reads it (cfb-raw is a separate venv that can't
import this package, hence a precomputed parquet is the bridge).

The consensus math reuses the validated ``spread_backfill.load_consensus_spreads``
(``home_spread`` POSITIVE == home favored), then maps to the CFBPlayProcess
``odds_override`` contract: ``gameSpread`` is the magnitude, ``homeFavorite`` the
side, ``overUnder`` the total (default 55.5 when a game has spread but no total).

Usage::

    uv run python -m betting.build_odds_consensus
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from model_training.spread_backfill import load_consensus_spreads

_ROOT = Path(__file__).resolve().parents[4]  # .../sdv-dev
ODDS = _ROOT / "cfbfastR-dev/cfbfastR-data/betting/parquet/cfb_line_odds.parquet"
OUT = _ROOT / "cfbfastR-dev/cfbfastR-cfb-raw/cfb/odds_consensus.parquet"


def build() -> dict:
    cons = load_consensus_spreads(str(ODDS))  # [game_id, home_spread, over_under]
    out = (
        cons.filter(pl.col("home_spread").is_not_null())
        .with_columns(
            game_id=pl.col("game_id").cast(pl.Int64),
            gameSpread=pl.col("home_spread").abs(),
            homeFavorite=pl.col("home_spread") > 0,
            overUnder=pl.col("over_under").fill_null(55.5),
            gameSpreadAvailable=pl.lit(True),
        )
        .select("game_id", "gameSpread", "overUnder", "homeFavorite", "gameSpreadAvailable")
        .unique(subset="game_id")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    summary = {
        "games": len(out),
        "home_fav_share": round(out["homeFavorite"].mean(), 3),
        "spread_median": float(out["gameSpread"].median()),
        "out": str(OUT),
    }
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    build()
