"""Head coach per team-season for CFB: a vendored roster plus a CFBD refresh.

No per-game coach source exists for college football the way the nflverse
schedule provides one for the NFL, so attribution is per TEAM-SEASON:
``data/cfb_coach_seasons.csv`` (season, coach, school, conference, games,
wins, losses, clean_attribution) names the head coach of each school-season;
``games`` is games coached (wins + losses, + ties where CFBD reports them) and
:func:`load_coach_seasons` refuses a roster where it is smaller than the record.
``clean_attribution`` is True when that coach led at least 80% of the
school's games that season; a school-season split between two coaches is
left unattributed rather than credited to either.

``school`` is the CFBD school name, which equals the ESPN ``location`` the
schedule master carries as ``home_location`` / ``away_location`` (all 137
schools of the vendored roster match exactly), so the school -> ESPN team id
map is a join against the season's schedule.

Refresh (a new season, or a correction) reads the CFBD ``/coaches`` endpoint::

    uv run python -m cfb_data_build.coaches -s 2026 -e 2026

with ``CFBD_API_KEY`` from the environment or, failing that, ``~/.Renviron`` /
``~/Documents/.Renviron`` read at call time. The key is never logged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

COACH_SEASONS_CSV = (
    Path(__file__).resolve().parents[2] / "data" / "cfb_coach_seasons.csv"
)
CFBD_COACHES = "https://api.collegefootballdata.com/coaches"
#: a coach owns a school-season when they led at least this share of its games
CLEAN_ATTRIBUTION_SHARE = 0.8
COACH_SCHEMA: dict[str, type[pl.DataType]] = {
    "season": pl.Int64,
    "coach": pl.Utf8,
    "school": pl.Utf8,
    "conference": pl.Utf8,
    "games": pl.Int64,
    "wins": pl.Int64,
    "losses": pl.Int64,
    "clean_attribution": pl.Boolean,
}
TEAM_COACH_SCHEMA: dict[str, type[pl.DataType]] = {
    "season": pl.Int64,
    "team_id": pl.Int64,
    "coach": pl.Utf8,
}
_UA = {"User-Agent": "Mozilla/5.0 (compatible; sportsdataverse/cfb-data)"}
_RENVIRON = (Path.home() / ".Renviron", Path.home() / "Documents" / ".Renviron")


def load_coach_seasons(path: str | Path = COACH_SEASONS_CSV) -> pl.DataFrame:
    """The vendored roster with the documented schema (empty frame when absent)."""
    p = Path(path)
    if not p.exists():
        return pl.DataFrame(schema=COACH_SCHEMA)
    df = pl.read_csv(
        p,
        schema_overrides={
            k: v for k, v in COACH_SCHEMA.items() if k != "clean_attribution"
        },
    )
    if df.schema.get("clean_attribution") != pl.Boolean:
        df = df.with_columns(
            pl.col("clean_attribution")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(["true", "1"])
            .alias("clean_attribution")
        )
    df = df.select(list(COACH_SCHEMA)).cast(COACH_SCHEMA)  # type: ignore[arg-type]
    short = df.filter(pl.col("games") < pl.col("wins") + pl.col("losses"))
    if short.height:
        raise ValueError(
            f"{p}: {short.height} coach-season(s) declare fewer games than wins + losses "
            f"(e.g. {short.row(0, named=True)}); games must be games coached"
        )
    return df


def school_team_ids(schedule: str | Path, season: int) -> pl.DataFrame:
    """``(team_id, school)`` for one season from the schedule master's home and away sides."""
    lf = pl.scan_parquet(str(schedule)).filter(pl.col("season") == int(season))
    sides = [
        lf.select(
            pl.col(f"{side}_id").cast(pl.Int64, strict=False).alias("team_id"),
            pl.col(f"{side}_location").cast(pl.Utf8).alias("school"),
        )
        for side in ("home", "away")
    ]
    return (
        pl.concat(sides).drop_nulls().unique(subset=["team_id"], keep="first").collect()
    )


def team_coaches(
    season: int, schedule: str | Path, roster: pl.DataFrame | None = None
) -> pl.DataFrame:
    """``(season, team_id, coach)`` for every cleanly attributed school-season.

    A school with two coaches in the roster for the season (neither clean) is
    absent, and so are schools the schedule does not name; the tendencies
    build then simply has no coach row for that team.
    """
    roster = load_coach_seasons() if roster is None else roster
    rows = roster.filter(
        (pl.col("season") == int(season)) & (pl.col("clean_attribution") == True)
    )
    if rows.height == 0:
        return pl.DataFrame(schema=TEAM_COACH_SCHEMA)
    ids = school_team_ids(schedule, season)
    assert rows.schema["school"] == ids.schema["school"]
    out = (
        rows.join(ids, on="school", how="inner")
        .select("season", "team_id", "coach")
        .unique(subset=["season", "team_id"], keep="first")
    )
    return out.cast(TEAM_COACH_SCHEMA)  # type: ignore[arg-type]


def coach_rows_from_cfbd(payload: list[dict[str, Any]], season: int) -> pl.DataFrame:
    """Flatten a CFBD ``/coaches?year=`` payload into roster rows for ``season``.

    ``clean_attribution`` is derived from the share of the school's games the
    coach led (CFBD lists an interim coach as a second entry for the school).
    """
    recs: list[dict[str, Any]] = []
    for coach in payload or []:
        name = " ".join(
            str(coach.get(k) or "").strip() for k in ("first_name", "last_name")
        ).strip()
        for s in coach.get("seasons") or []:
            if int(s.get("year") or 0) != int(season) or not name:
                continue
            recs.append(
                {
                    "season": int(season),
                    "coach": name,
                    "school": str(s.get("school") or ""),
                    "conference": None,
                    "games": int(s.get("games") or 0),
                    "wins": int(s.get("wins") or 0),
                    "losses": int(s.get("losses") or 0),
                }
            )
    if not recs:
        return pl.DataFrame(schema=COACH_SCHEMA)
    df = pl.DataFrame(recs)
    total = pl.col("games").sum().over("school")
    return (
        df.with_columns(
            clean_attribution=(
                pl.col("games") / pl.when(total > 0).then(total).otherwise(1)
            )
            >= CLEAN_ATTRIBUTION_SHARE
        )
        .select(list(COACH_SCHEMA))
        .cast(COACH_SCHEMA)  # type: ignore[arg-type]
        .sort(["school", "coach"])
    )


def _renviron_key(name: str = "CFBD_API_KEY") -> str | None:
    pat = re.compile(rf"^\s*{re.escape(name)}\s*=\s*['\"]?([^'\"\s#]+)")
    for path in _RENVIRON:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                m = pat.match(line)
                if m:
                    return m.group(1)
        except OSError:
            continue
    return None


def cfbd_api_key(api_key: str | None = None) -> str:
    """The CFBD key: explicit arg, then env, then ``.Renviron`` (read now, never echoed)."""
    key = api_key or os.environ.get("CFBD_API_KEY") or _renviron_key()
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set (env or ~/.Renviron); the coach refresh reads CFBD /coaches"
        )
    return key


def fetch_cfbd_coaches(
    season: int, *, api_key: str | None = None, url: str = CFBD_COACHES
) -> list[dict[str, Any]]:
    key = cfbd_api_key(api_key)
    req = urllib.request.Request(
        f"{url}?year={int(season)}",
        headers={**_UA, "Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    if not isinstance(payload, list):
        # an error envelope with HTTP 200 must not read as "no coaches this season"
        raise TypeError(
            f"CFBD /coaches?year={season}: unexpected payload shape {type(payload).__name__}"
        )
    return payload


def refresh_coach_seasons(
    seasons: list[int],
    *,
    path: str | Path = COACH_SEASONS_CSV,
    api_key: str | None = None,
    fetch=fetch_cfbd_coaches,
) -> pl.DataFrame:
    """Replace the roster rows of ``seasons`` with CFBD's and rewrite the CSV.

    Other seasons are untouched (the vendored 2004-2025 rows carry a
    conference CFBD's coaches endpoint does not).
    """
    current = load_coach_seasons(path)
    fresh = [coach_rows_from_cfbd(fetch(s, api_key=api_key), s) for s in seasons]
    # a season CFBD returns nothing for keeps its vendored rows: an empty answer
    # (not yet published, transient gap) must never erase attribution
    replaced = [int(s) for s, f in zip(seasons, fresh) if f.height]
    for s, f in zip(seasons, fresh):
        if not f.height:
            print(f"coaches {s}: CFBD returned no coach-seasons; existing rows kept")
    kept = current.filter(~pl.col("season").is_in(replaced))
    out = pl.concat(
        [kept, *[f for f in fresh if f.height]], how="vertical_relaxed"
    ).sort(["season", "school", "coach"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cfb_data_build.coaches",
        description="refresh data/cfb_coach_seasons.csv from CFBD",
    )
    ap.add_argument("-s", "--start-year", type=int, required=True)
    ap.add_argument("-e", "--end-year", type=int, default=None)
    ap.add_argument("--path", default=str(COACH_SEASONS_CSV))
    args = ap.parse_args(argv)
    end = args.end_year if args.end_year is not None else args.start_year
    seasons = list(range(args.start_year, end + 1))
    out = refresh_coach_seasons(seasons, path=args.path)
    for s in seasons:
        n = out.filter(pl.col("season") == s)
        print(
            f"{s}: {n.height} coach-seasons, {n['clean_attribution'].sum()} cleanly attributed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
