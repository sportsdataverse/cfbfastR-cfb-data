"""Build the unified 2006-2025 ``cfb_line_odds`` archive with real game_id + team_id.

Inputs
------
* ``cfbfastR-data/betting/parquet/cfb_line_odds.parquet`` -- the historical
  multi-book line archive (2006-2019). ~25% of rows have a NULL ``game_id``
  (mostly FCS games with no ESPN schedule entry, plus a few FBS alias misses).
* ``betting/.cache/cfbd_lines/*.json`` -- CFBD ``/lines`` chunks (2020-2025),
  fetched by :mod:`betting.fetch_cfbd_lines`. CFBD's game ``id`` IS the ESPN
  game_id, so the modern era is natively keyed.
* ``cfbfastR-cfb-raw/cfb/cfb_schedule_master.parquet`` -- the ESPN schedule
  (2004-2025); the authority for ``game_id -> (home_id, away_id)`` team ids and
  for resolving the archive's NULL game_ids by ``(season, away, home)`` name.

Output (same dir as the input parquet)
------
``cfb_line_odds.parquet`` + ``cfb_line_odds.csv.gz`` -- the archive's 14 columns
plus ``home_team_id`` / ``away_team_id`` (the real ESPN team ids). The original
file is copied to ``cfb_line_odds.prev.parquet`` first.

Usage::

    uv run python -m betting.build_line_odds            # build + write
    uv run python -m betting.build_line_odds --dry-run  # validate only, no write
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]  # .../sdv-dev
ARCHIVE = _ROOT / "cfbfastR-dev/cfbfastR-data/betting/parquet/cfb_line_odds.parquet"
SCHED = _ROOT / "cfbfastR-dev/cfbfastR-cfb-raw/cfb/cfb_schedule_master.parquet"
CACHE = _HERE / ".cache" / "cfbd_lines"
OUT_PARQUET = ARCHIVE
OUT_CSV = ARCHIVE.parent.parent / "csv" / "cfb_line_odds.csv.gz"
# Pristine historical archive (2006-2019). build() ALWAYS reads from here, never from
# the output, so re-runs are idempotent (reading the merged output would re-append the
# CFBD era and double-count). Created on first run from the original input.
SOURCE = ARCHIVE.parent / "cfb_line_odds.source.parquet"

# archive name (lowercased/normalized) -> schedule_master *_location (normalized).
# Only the handful of forms that differ from ESPN's "location" string.
_ALIAS = {
    "texas christian": "tcu",
    "louisiana state": "lsu",
    "southern methodist": "smu",
    "central florida": "ucf",
    "miami florida": "miami",
    "miami (fl)": "miami",
    "miami fl": "miami",
    "miami ohio": "miami (oh)",
    "brigham young": "byu",
    "alabama birmingham": "uab",
    "nevada las vegas": "unlv",
    "texas el paso": "utep",
    "texas san antonio": "ut san antonio",
    "louisiana lafayette": "louisiana",
    "louisiana monroe": "ul monroe",
    "north carolina state": "nc state",
    "southern mississippi": "southern miss",
    "florida international": "fiu",
    "middle tennessee st": "middle tennessee",
    "pittsburgh": "pitt",
}

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _norm(name: str | None) -> str | None:
    """Normalize a team name for cross-source matching."""
    if name is None:
        return None
    s = name.strip().lower()
    s = s.replace("&", " and ").replace(".", "")
    s = s.replace(" st ", " state ")
    if s.endswith(" st"):
        s = s[:-3] + " state"
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return _ALIAS.get(s, s)


# ---------------------------------------------------------------------------
# schedule_master: the team-id authority + the (season, names) -> game_id map
# ---------------------------------------------------------------------------
def load_sched() -> pl.DataFrame:
    sch = pl.read_parquet(SCHED).select(
        "game_id", "season", "week", "season_type",
        "home_id", "away_id", "home_location", "away_location",
    ).with_columns(
        pl.col("game_id").cast(pl.Int64),
        pl.col("home_id").cast(pl.Int64, strict=False),
        pl.col("away_id").cast(pl.Int64, strict=False),
        pl.col("season").cast(pl.Int64),
    )
    return sch


def _team_id_map(sch: pl.DataFrame) -> pl.DataFrame:
    """game_id -> (home_team_id, away_team_id)."""
    return sch.select(
        "game_id",
        pl.col("home_id").alias("home_team_id"),
        pl.col("away_id").alias("away_team_id"),
    ).unique(subset="game_id")


def _name_to_gid(sch: pl.DataFrame) -> pl.DataFrame:
    """(season, norm_away, norm_home) -> game_id, for resolving NULL archive keys."""
    return (
        sch.with_columns(
            norm_home=pl.col("home_location").map_elements(_norm, return_dtype=pl.Utf8),
            norm_away=pl.col("away_location").map_elements(_norm, return_dtype=pl.Utf8),
        )
        .select("season", "norm_away", "norm_home", pl.col("game_id").alias("_match_gid"),
                pl.col("home_id").alias("_m_home_id"), pl.col("away_id").alias("_m_away_id"))
        .unique(subset=["season", "norm_away", "norm_home"])
    )


# ---------------------------------------------------------------------------
# CFBD cache -> archive long schema (native game_id)
# ---------------------------------------------------------------------------
def normalize_cfbd() -> pl.DataFrame:
    """Flatten cached CFBD /lines games into the archive's long row shape."""
    rows: list[dict] = []
    for f in sorted(CACHE.glob("*.json")):
        for g in json.loads(f.read_text() or "[]"):
            gid = g.get("id")
            home, away = g.get("homeTeam"), g.get("awayTeam")
            base = dict(
                id=float(gid) if gid is not None else None,
                game_id=gid,
                season=float(g.get("season")) if g.get("season") is not None else None,
                game_desc=f"{away}@{home}",
                date_time=g.get("startDate"),
                season_type=g.get("seasonType"),
                week=g.get("week"),
                home_team_id=g.get("homeTeamId"),
                away_team_id=g.get("awayTeamId"),
            )
            for ln in g.get("lines") or []:
                book = ln.get("provider")
                sp, spo = ln.get("spread"), ln.get("spreadOpen")
                ou, ouo = ln.get("overUnder"), ln.get("overUnderOpen")
                hml, aml = ln.get("homeMoneyline"), ln.get("awayMoneyline")
                # spread: home line (= CFBD spread) + away line (= -spread)
                if sp is not None:
                    rows.append({**base, "market_type": "spread", "abbr": home,
                                 "lines": float(sp), "odds": None,
                                 "opening_lines": float(spo) if spo is not None else None,
                                 "opening_odds": None, "book": book})
                    rows.append({**base, "market_type": "spread", "abbr": away,
                                 "lines": -float(sp), "odds": None,
                                 "opening_lines": -float(spo) if spo is not None else None,
                                 "opening_odds": None, "book": book})
                # total: over + under both carry the same number
                if ou is not None:
                    for side in ("over", "under"):
                        rows.append({**base, "market_type": "total", "abbr": side,
                                     "lines": float(ou), "odds": None,
                                     "opening_lines": float(ouo) if ouo is not None else None,
                                     "opening_odds": None, "book": book})
                # money_line: home + away prices
                if hml is not None:
                    rows.append({**base, "market_type": "money_line", "abbr": home,
                                 "lines": None, "odds": int(hml),
                                 "opening_lines": None, "opening_odds": None, "book": book})
                if aml is not None:
                    rows.append({**base, "market_type": "money_line", "abbr": away,
                                 "lines": None, "odds": int(aml),
                                 "opening_lines": None, "opening_odds": None, "book": book})
    # explicit schema: list-of-dicts inference trips over None-then-float columns
    # (e.g. money_line rows carry lines=None before a spread row's float appears).
    schema = {
        "id": pl.Float64, "game_id": pl.Int64, "season": pl.Float64,
        "game_desc": pl.Utf8, "date_time": pl.Utf8, "season_type": pl.Utf8,
        "week": pl.Int64, "home_team_id": pl.Int64, "away_team_id": pl.Int64,
        "market_type": pl.Utf8, "abbr": pl.Utf8, "lines": pl.Float64,
        "odds": pl.Int64, "opening_lines": pl.Float64, "opening_odds": pl.Int64,
        "book": pl.Utf8,
    }
    # empty cache (no chunks fetched): return the full schema with 0 rows so build()'s
    # downstream .drop()/join/_shape don't crash on a 0-column frame.
    if not rows:
        return pl.DataFrame(schema=schema)
    # the team ids CFBD ships are its own; override with ESPN ids via game_id below.
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# archive: add team ids for matched games + resolve NULL game_ids by name
# ---------------------------------------------------------------------------
def key_archive(arc: pl.DataFrame, sch: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    n0 = len(arc)
    null0 = arc["game_id"].null_count()
    arc = arc.with_columns(
        norm_away=pl.col("game_desc").str.split_exact("@", 1).struct.field("field_0")
            .str.strip_chars().map_elements(_norm, return_dtype=pl.Utf8),
        norm_home=pl.col("game_desc").str.split_exact("@", 1).struct.field("field_1")
            .str.strip_chars().map_elements(_norm, return_dtype=pl.Utf8),
        season_i=pl.col("season").cast(pl.Int64),
    )
    n2g = _name_to_gid(sch)
    arc = arc.join(n2g, left_on=["season_i", "norm_away", "norm_home"],
                   right_on=["season", "norm_away", "norm_home"], how="left")
    # fill only the NULL game_ids from the name match
    arc = arc.with_columns(
        game_id=pl.when(pl.col("game_id").is_null()).then(pl.col("_match_gid"))
                  .otherwise(pl.col("game_id")).cast(pl.Int64),
    )
    null1 = arc["game_id"].null_count()
    # team ids via the (now-filled) game_id
    tid = _team_id_map(sch)
    arc = arc.join(tid, on="game_id", how="left").drop(
        "_match_gid", "_m_home_id", "_m_away_id", "norm_away", "norm_home", "season_i",
    )
    stats = {"rows": n0, "null_game_id_before": null0, "null_game_id_after": null1,
             "filled": null0 - null1}
    return arc, stats


_FINAL_COLS = ["id", "game_id", "season", "game_desc", "date_time", "market_type",
               "abbr", "lines", "odds", "opening_lines", "opening_odds", "book",
               "season_type", "week", "home_team_id", "away_team_id"]


def build(dry_run: bool = False) -> dict:
    sch = load_sched()
    # Read the pristine 2006-2019 archive, never the (overwritten) output -- see SOURCE.
    arc = pl.read_parquet(SOURCE if SOURCE.exists() else ARCHIVE)
    arc, arc_stats = key_archive(arc, sch)

    cfbd = normalize_cfbd()
    # override CFBD's own team ids with ESPN ids via game_id (authoritative)
    tid = _team_id_map(sch)
    cfbd = cfbd.drop("home_team_id", "away_team_id").join(tid, on="game_id", how="left")

    # align dtypes + column order, then stack
    def _shape(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("game_id").cast(pl.Int64, strict=False),
            pl.col("season").cast(pl.Float64, strict=False),
            pl.col("week").cast(pl.Int64, strict=False),
            pl.col("home_team_id").cast(pl.Int64, strict=False),
            pl.col("away_team_id").cast(pl.Int64, strict=False),
            pl.col("lines").cast(pl.Float64, strict=False),
            pl.col("odds").cast(pl.Int64, strict=False),
            pl.col("opening_lines").cast(pl.Float64, strict=False),
            pl.col("opening_odds").cast(pl.Int64, strict=False),
            pl.col("id").cast(pl.Float64, strict=False),
        ).select(_FINAL_COLS)

    merged = pl.concat([_shape(arc), _shape(cfbd)], how="vertical_relaxed")

    summary = {
        "archive": arc_stats,
        "cfbd_rows": len(cfbd),
        "cfbd_games": cfbd["game_id"].n_unique(),
        "merged_rows": len(merged),
        "season_range": (int(merged["season"].min()), int(merged["season"].max())),
        "null_game_id_total": int(merged["game_id"].null_count()),
        "null_team_id_rows": int(merged["home_team_id"].null_count()),
        "rows_per_season": merged.group_by(pl.col("season").cast(pl.Int64))
            .len().sort("season").to_dicts(),
    }
    if not dry_run:
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the pristine 2006-2019 archive as the committed, immutable SOURCE the
        # next run reads from (only ever created once, from the original input).
        if ARCHIVE.exists() and not SOURCE.exists():
            shutil.copy2(ARCHIVE, SOURCE)
        merged.write_parquet(OUT_PARQUET)
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        # polars write_csv ignores the .gz extension -- compress through a gzip stream.
        import gzip
        with gzip.open(OUT_CSV, "wb") as fh:
            merged.write_csv(fh)
        summary["wrote"] = [str(OUT_PARQUET), str(OUT_CSV)]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="betting.build_line_odds")
    ap.add_argument("--dry-run", action="store_true", help="validate only; do not write")
    args = ap.parse_args()
    summary = build(dry_run=args.dry_run)
    import pprint
    pprint.pprint(summary, width=120, sort_dicts=False)


if __name__ == "__main__":
    main()
