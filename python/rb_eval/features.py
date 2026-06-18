"""Load rush plays from final.json and compute fo_success + is_rush_opp.

Pre-2014 ESPN games carry no structured per-play participants, so
``rusher_player_name`` is null even though the play ``text`` names the rusher
("Mike Kafka rush for 4 yards ...") and the game ``boxscore`` lists the canonical
rushing names. ``_fill_rusher_from_text`` recovers the join for those seasons by
matching the play text against the boxscore's rushing athletes (text identifies
*which* rusher; the boxscore supplies the canonical name), with a regex fallback.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

# Leading-name-before-"rush" fallback when no boxscore rusher prefixes the text.
_RUSH_NAME_RE = re.compile(r"^\s*([A-Z][\w'.\-]+(?:\s+[A-Z][\w'.\-]+){0,3})\s+rush", re.IGNORECASE)


def _boxscore_rushers(raw: dict) -> list[str]:
    """Canonical rushing athlete display names from a final.json boxscore."""
    names: set[str] = set()
    for team in (raw.get("boxscore") or {}).get("players") or []:
        for stat in team.get("statistics") or []:
            if stat.get("name") == "rushing":
                for a in stat.get("athletes") or []:
                    nm = (a.get("athlete") or {}).get("displayName")
                    if nm:
                        names.add(str(nm))
    # longest-first so "Mike Williams" wins over a hypothetical "Mike"
    return sorted(names, key=len, reverse=True)


def _match_rusher(text: str, box_rushers: list[str]) -> str | None:
    """Resolve the rusher named in a rush play's text to a canonical box name."""
    if not text:
        return None
    head = text[:60]
    for nm in box_rushers:  # box-canonical: the rusher prefixes the narrative
        if head.startswith(nm + " ") or head.startswith(nm + ","):
            return nm
    m = _RUSH_NAME_RE.match(text)  # fallback: leading name token(s) before "rush"
    return m.group(1).strip() if m else None


def _fill_rusher_from_text(plays: list[dict], raw: dict) -> None:
    """In-place: populate null ``rusher_player_name`` on rush plays from text + boxscore."""
    box = _boxscore_rushers(raw)
    for p in plays:
        if p.get("rusher_player_name"):
            continue
        if p.get("rush") not in (True, 1):
            continue
        nm = _match_rusher(p.get("text") or "", box)
        if nm:
            p["rusher_player_name"] = nm


def add_fo_success(df: pl.DataFrame) -> pl.DataFrame:
    """Annotate each rush with first-opportunity success per down tier."""
    return df.with_columns(
        fo_success=pl.when(pl.col("start.down") == 1)
        .then(pl.col("yds_rushed") >= 0.5 * pl.col("start.distance"))
        .when(pl.col("start.down") == 2)
        .then(pl.col("yds_rushed") >= 0.7 * pl.col("start.distance"))
        .otherwise(pl.col("yds_rushed") >= pl.col("start.distance"))
        .cast(pl.Boolean),
    )


def filter_rush_plays(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only individual rusher plays; add fo_success + is_rush_opp."""
    epa_col = "EPA" if "EPA" in df.columns else "epa"
    out = (
        df.filter(pl.col("rush") == True)  # noqa: E712
        .filter(pl.col("pos_team").is_not_null())
        .filter(pl.col(epa_col).is_not_null())
        .filter(pl.col("rusher_player_name").is_not_null())
        .filter(pl.col("rusher_player_name") != "TEAM")
    )
    if epa_col == "EPA":
        out = out.rename({"EPA": "epa"})
    out = add_fo_success(out)
    return out.with_columns(is_rush_opp=(pl.col("yds_rushed") >= 4).cast(pl.Boolean))


def load_rush_plays(final_dir: str | Path, seasons: list[int] | None = None) -> pl.DataFrame:
    """Load rush plays from per-game final.json files in *final_dir*."""
    frames = []
    for path in sorted(Path(final_dir).glob("*.json")):
        raw = json.loads(path.read_text())
        if seasons is not None and raw.get("season") not in seasons:
            continue
        plays = raw.get("plays") or []
        if not plays:
            continue
        _fill_rusher_from_text(plays, raw)  # backfill pre-2014 null rusher names
        frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    return filter_rush_plays(pl.concat(frames, how="diagonal_relaxed"))
