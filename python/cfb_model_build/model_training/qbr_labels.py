"""ESPN game-level QBR -- the xQBR training target -- captured once and committed.

The labels used to be re-scraped inside every pipeline run and thrown away, so no
QBR fit could be reproduced: ESPN revises and drops values, and nothing recorded
what a model had actually been fitted against. They now live in
``models/qbr/espn_qbr_labels.parquet`` (provenance in ``models/qbr/README.md``).
Re-capturing is a deliberate, reviewed step -- the parquet diff shows what ESPN moved.

Source: ESPN core API ``seasons/{season}/types/{2|3}/weeks/{week}/qbr/10000``
through sdv-py ``espn_cfb_season_qbr_week``. One row per (game_id, athlete_id),
both Int64, parsed from the item's ``$ref`` links (never through a float). ``QBR``
is ESPN raw QBR (the target), ``TQBR`` is Total QBR (opponent-adjusted).
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

LABELS_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "qbr" / "espn_qbr_labels.parquet"
)
URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
    "seasons/{season}/types/{season_type}/weeks/{week}/qbr/10000?limit=1000"
)
# ESPN's CFB calendar: regular season weeks 1-16 (17 is always empty, kept as a
# guard for a longer calendar); every bowl and CFP game sits in postseason week 1.
WEEKS = [(2, w) for w in range(1, 18)] + [(3, 1)]
_ID = {k: re.compile(rf"/{k}s/(\d+)") for k in ("event", "athlete", "team")}


def parse_week(payload: dict, season: int, season_type: int, week: int) -> list[dict]:
    """Flatten one weekly QBR payload to label rows."""
    if (payload.get("pageCount") or 0) > 1:
        raise ValueError(
            f"{season}/{season_type}/{week}: {payload['pageCount']} pages at limit=1000"
        )
    rows = []
    for item in payload.get("items") or []:
        ids = {}
        for k, rx in _ID.items():
            m = rx.search((item.get(k) or {}).get("$ref", ""))
            ids[k] = int(m.group(1)) if m else None
        row = {
            "season": season,
            "season_type": season_type,
            "week": week,
            "game_id": ids["event"],
            "athlete_id": ids["athlete"],
            "team_id": ids["team"],
        }
        for cat in (item.get("splits") or {}).get("categories") or []:
            for st in cat.get("stats") or []:
                row[st["abbreviation"]] = st.get("value")
        rows.append(row)
    return rows


def capture(seasons, *, fetch=None, pause: float = 0.5) -> pl.DataFrame:
    """Scrape every regular- and postseason week of ``seasons``; sequential, paced."""
    if fetch is None:
        from sportsdataverse.cfb.cfb_espn_ext import espn_cfb_season_qbr_week

        def fetch(season, week, season_type):
            return espn_cfb_season_qbr_week(
                season,
                week,
                season_type,
                split=10000,
                return_parsed=False,
                params={"limit": 1000},
            )

    rows = []
    for season in seasons:
        for season_type, week in WEEKS:
            rows += parse_week(
                fetch(season, week, season_type), season, season_type, week
            )
            time.sleep(pause)
        print(
            f"qbr labels {season}: {sum(r['season'] == season for r in rows)} rows",
            flush=True,
        )
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col(
            "season", "season_type", "week", "game_id", "athlete_id", "team_id"
        ).cast(pl.Int64),
        captured_at=pl.lit(datetime.now(UTC).strftime("%Y-%m-%d")),
    )
    dup = df.filter(pl.struct("game_id", "athlete_id").is_duplicated())
    if dup.height:
        raise ValueError(f"{dup.height} duplicate (game_id, athlete_id) label rows")
    return df


def update_labels(seasons, out=LABELS_PATH, **kw) -> pl.DataFrame:
    """Re-capture ``seasons`` and replace just those seasons in the committed file."""
    new = capture(seasons, **kw)
    out = Path(out)
    if out.exists():
        old = pl.read_parquet(out).filter(~pl.col("season").is_in(list(seasons)))
        new = pl.concat([old, new], how="diagonal_relaxed")
    new = new.sort("season", "season_type", "week", "game_id", "athlete_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    new.write_parquet(out)
    return new


def load_labels(path=LABELS_PATH) -> pl.DataFrame:
    """The committed labels: ids, week, raw QBR ``QBR`` and Total QBR ``TQBR``."""
    return pl.read_parquet(path)
