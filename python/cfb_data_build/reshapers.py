"""Bespoke per-game reshapers (polars ports of the non-generic R creators).

Each function mirrors one ``reshape_*`` in ``R/espn_cfb_*_creation.R`` and is
registered in :data:`RESHAPERS` keyed by dataset name. The generic
block-flatten datasets do NOT appear here -- they are driven by
:func:`cfb_data_build.reshape.flat_block_frame` via a ``block`` path in the
registry. ``"pbp"`` maps to the conform builder in
:mod:`cfb_data_build.pbp`.

R ``%||%`` (null-coalesce), ``as.integer/as.numeric/as.logical/as.character``,
and ``isTRUE`` are reproduced by the small ``_*`` coercion helpers; nested
``a$b$c`` access by :func:`_dig`. Output column ORDER matches each R reshape
exactly (verified against the captured R oracle parquets).
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from cfb_data_build.pbp import build_pbp_frame
from cfb_data_build.reshape import flat_block_frame, stamp_identity


def _dig(node: Any, *keys: str) -> Any:
    """Nested ``a$b$c`` access; ``None`` if any level is missing/non-dict."""
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
        if node is None:
            return None
    return node


def _int(v: Any) -> int | None:
    """``as.integer`` -- ``None`` on null/uncoercible (R NA)."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _num(v: Any) -> float | None:
    """``as.numeric`` -- ``None`` on null/uncoercible."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v: Any) -> bool | None:
    """``as.logical`` -- ``None`` on null; JSON bools pass through."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t"):
            return True
        if s in ("false", "f"):
            return False
        return None
    return bool(v)


def _istrue(v: Any) -> bool:
    """R ``isTRUE`` -- only literal ``True`` is true; everything else is False."""
    return v is True


def _chr(v: Any) -> str | None:
    """``as.character`` -- ``None`` on null, else ``str``."""
    return None if v is None else str(v)


# --- team_box (R/espn_cfb_02) --------------------------------------------------
def reshape_team_box(game: dict[str, Any]) -> pl.DataFrame:
    """Pivot ``g$boxscore$teams[].statistics`` (name -> displayValue) per team."""
    teams = _dig(game, "boxscore", "teams")
    if not teams:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for t in teams:
        row: dict[str, Any] = {}
        for s in t.get("statistics") or []:
            name = s.get("name") or s.get("label") or "stat"
            dv = s.get("displayValue")
            if dv is None:
                dv = s.get("value")
            row[name] = _chr(dv)
        row["team_id"] = _int(_dig(t, "team", "id"))
        row["team_abbreviation"] = _dig(t, "team", "abbreviation")
        row["team_name"] = _dig(t, "team", "displayName")
        row["home_away"] = t.get("homeAway")
        rows.append(row)
    return stamp_identity(
        pl.from_dicts(rows, infer_schema_length=None), game, week=False
    )


# --- player_box (R/espn_cfb_03) ------------------------------------------------
def reshape_player_box(game: dict[str, Any]) -> pl.DataFrame:
    """One row per athlete per stat category; category ``keys`` become columns."""
    groups = _dig(game, "boxscore", "players")
    if not groups:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for grp in groups:
        tid = _int(_dig(grp, "team", "id"))
        for cat in grp.get("statistics") or []:
            keys = cat.get("keys") or []
            cname = cat.get("name")
            for a in cat.get("athletes") or []:
                vals = [_chr(x) for x in (a.get("stats") or [])]
                if len(keys) == len(vals) and len(keys) > 0:
                    row: dict[str, Any] = dict(zip(keys, vals))
                elif len(vals) > 0:
                    row = {f"stat_{i + 1}": v for i, v in enumerate(vals)}
                else:
                    row = {}
                row["category"] = cname
                row["athlete_id"] = _int(_dig(a, "athlete", "id"))
                row["athlete_name"] = _dig(a, "athlete", "displayName")
                row["jersey"] = _dig(a, "athlete", "jersey")
                row["team_id"] = tid
                rows.append(row)
    if not rows:
        return pl.DataFrame()
    return stamp_identity(
        pl.from_dicts(rows, infer_schema_length=None), game, week=False
    )


# --- drives (R/espn_cfb_06) ----------------------------------------------------
def _drive_to_row(d: dict[str, Any]) -> dict[str, Any]:
    """One ESPN drive object -> 1-row dict (nested plays[] dropped)."""
    return {
        "drive_id": d.get("id"),
        "team_id": _int(_dig(d, "team", "id")),
        "result": d.get("result"),
        "display_result": d.get("displayResult"),
        "short_display_result": d.get("shortDisplayResult"),
        "description": d.get("description"),
        "yards": d.get("yards"),
        "offensive_plays": d.get("offensivePlays"),
        "is_score": d.get("isScore"),
        "start_period": _dig(d, "start", "period", "number"),
        "start_yard_line": _dig(d, "start", "yardLine"),
        "start_clock": _dig(d, "start", "clock", "displayValue"),
        "start_text": _dig(d, "start", "text"),
        "end_period": _dig(d, "end", "period", "number"),
        "end_yard_line": _dig(d, "end", "yardLine"),
        "end_clock": _dig(d, "end", "clock", "displayValue"),
        "time_elapsed": _dig(d, "timeElapsed", "displayValue"),
        "n_plays": len(d.get("plays") or []),
    }


def reshape_drives(game: dict[str, Any]) -> pl.DataFrame:
    """Unroll ``g$drives`` ({previous, current} or bare list) to one row per drive."""
    dv = game.get("drives")
    if not dv:
        return pl.DataFrame()
    if isinstance(dv, dict) and ("previous" in dv or "current" in dv):
        all_drives = (dv.get("previous") or []) + (dv.get("current") or [])
    elif isinstance(dv, dict):
        all_drives = []
        for v in dv.values():
            all_drives.extend(v if isinstance(v, list) else [v])
    else:
        all_drives = []
        for v in dv:
            all_drives.extend(v if isinstance(v, list) else [v])
    all_drives = [
        d for d in all_drives if isinstance(d, dict) and d.get("id") is not None
    ]
    if not all_drives:
        return pl.DataFrame()
    rows = [_drive_to_row(d) for d in all_drives]
    # game_id + season (no week), per R
    return stamp_identity(
        pl.from_dicts(rows, infer_schema_length=None), game, week=False
    )


# --- betting (R/espn_cfb_09) ---------------------------------------------------
def reshape_betting(game: dict[str, Any]) -> pl.DataFrame:
    """One game-level row of the resolved scalar odds fields of ``g$betting``."""
    b = game.get("betting")
    if b is None:
        return pl.DataFrame()
    row = {
        "game_id": int(game["id"]),
        "season": int(game["season"]),
        "week": int(game["week"]) if game.get("week") is not None else None,
        "game_spread": _num(b.get("game_spread")),
        "over_under": _num(b.get("over_under")),
        "home_favorite": _bool(b.get("home_favorite")),
        "home_team_spread": _num(b.get("home_team_spread")),
        "game_spread_available": _bool(b.get("game_spread_available")),
        "odds_source": _chr(b.get("odds_source")),
    }
    return pl.from_dicts([row], infer_schema_length=None)


# --- schedules (R/espn_cfb_10) -------------------------------------------------
def reshape_schedules(game: dict[str, Any]) -> pl.DataFrame:
    """One game-meta row from ``g$header$competitions[[1]]`` + ``g$gameInfo``."""
    comps_list = _dig(game, "header", "competitions")
    comp = comps_list[0] if isinstance(comps_list, list) and comps_list else None
    if comp is None:
        return pl.DataFrame()
    comps = comp.get("competitors") or []

    def side(ha: str) -> dict[str, Any] | None:
        m = [c for c in comps if c.get("homeAway") == ha]
        return m[0] if m else None

    home, away = side("home"), side("away")
    row = {
        "game_id": int(game["id"]),
        "season": int(game["season"]),
        "week": int(game["week"]) if game.get("week") is not None else None,
        "season_type": int(game["season_type"])
        if game.get("season_type") is not None
        else None,
        "game_date": comp.get("date"),
        "neutral_site": _istrue(comp.get("neutralSite")),
        "conference_competition": _istrue(comp.get("conferenceCompetition")),
        "home_id": _int(_dig(home, "team", "id")) if home else None,
        "away_id": _int(_dig(away, "team", "id")) if away else None,
        "home_team": _dig(home, "team", "displayName") if home else None,
        "away_team": _dig(away, "team", "displayName") if away else None,
        "home_abbreviation": _dig(home, "team", "abbreviation") if home else None,
        "away_abbreviation": _dig(away, "team", "abbreviation") if away else None,
        "home_score": _int(home.get("score")) if home else None,
        "away_score": _int(away.get("score")) if away else None,
        "home_winner": _istrue(home.get("winner")) if home else False,
        "away_winner": _istrue(away.get("winner")) if away else False,
        "venue": _dig(game, "gameInfo", "venue", "fullName"),
        "attendance": _int(_dig(game, "gameInfo", "attendance")),
        "status": _dig(comp, "status", "type", "name"),
    }
    return pl.from_dicts([row], infer_schema_length=None)


# --- linescores (R/espn_cfb_11) ------------------------------------------------
def reshape_linescores(game: dict[str, Any]) -> pl.DataFrame:
    """Long per-(team, period) scoring from ``g$team_box_extra[[tid]]$linescores``."""
    tbe = game.get("team_box_extra")
    if not tbe:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for tid, obj in tbe.items():
        ls = (obj or {}).get("linescores")
        if not ls:
            continue
        for i, item in enumerate(ls, start=1):
            v = item.get("displayValue")
            if v is None:
                v = item.get("value")
            rows.append({"team_id": _int(tid), "period": i, "value": _chr(v)})
    if not rows:
        return pl.DataFrame()
    return stamp_identity(
        pl.from_dicts(rows, infer_schema_length=None), game, week=False
    )


# --- power_index (R/espn_cfb_12) ----------------------------------------------
def reshape_power_index(game: dict[str, Any]) -> pl.DataFrame:
    """Flatten the FPI ``items`` (or the wrapper itself when no ``items`` key)."""
    pidx = game.get("power_index")
    if not pidx:
        return pl.DataFrame()
    items = (
        pidx.get("items")
        if isinstance(pidx, dict) and pidx.get("items") is not None
        else pidx
    )
    if not items:
        return pl.DataFrame()
    return flat_block_frame(items, game)


# --- rosters (R/espn_cfb_08, derive_rosters) -----------------------------------
# Per-game circumstance columns dropped on the season dedup.
GAME_ROSTER_GAME_COLS: list[str] = [
    "game_id",
    "week",
    "starter",
    "did_not_play",
    "order",
    "home_away",
    "winner",
]
_ROSTER_KEYS = ("season", "team_id", "athlete_id")


def derive_rosters(gr: pl.DataFrame) -> pl.DataFrame:
    """Dedup game_rosters to one row per (season, team_id, athlete_id), latest game.

    Port of ``derive_rosters`` (``R/espn_cfb_08``): order by ``game_id`` and keep
    the last (most-recent) row per key, then drop per-game circumstance columns.
    """
    if gr is None or gr.height == 0:
        return pl.DataFrame()
    df = gr
    if "game_id" in df.columns:
        df = df.sort("game_id")
    keys = [k for k in _ROSTER_KEYS if k in df.columns]
    if keys:
        df = df.group_by(keys, maintain_order=True).last()
    drop = [c for c in GAME_ROSTER_GAME_COLS if c in df.columns]
    if drop:
        df = df.drop(drop)
    return df


def reshape_rosters(game: dict[str, Any]) -> pl.DataFrame:
    """Season-roster derive over one game's game_rosters (parity-correct for 1 game)."""
    return derive_rosters(flat_block_frame(game.get("game_rosters"), game))


RESHAPERS: dict[str, Callable[[dict[str, Any]], pl.DataFrame]] = {
    "pbp": build_pbp_frame,
    "team_box": reshape_team_box,
    "player_box": reshape_player_box,
    "drives": reshape_drives,
    "betting": reshape_betting,
    "schedules": reshape_schedules,
    "linescores": reshape_linescores,
    "power_index": reshape_power_index,
    "rosters": reshape_rosters,
}
