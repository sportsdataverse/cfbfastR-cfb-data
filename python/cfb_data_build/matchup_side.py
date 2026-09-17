"""CFBD side inputs for the matchup line: talent, weather, team meta, head coaches.

The source pipeline attached these per-team / per-game groups to its line from
a mix of CFBD calls and proprietary inputs. This module ports the CFBD-derivable
part and leaves the rest null with its documented dtype:

* ``talent`` -- CFBD ``/talent`` for the season (the season's own composite; the
  endpoint publishes the current season late, so a preseason build carries
  nulls exactly as the source seeded them).
* ``head_coach`` / ``hc_tenure`` -- CFBD ``/coaches``: the PRESEASON incumbent
  (the source scraped its coaching list before the season, so a mid-season
  interim never replaces the incumbent even when the interim coached more
  games); ``hc_tenure`` = ``season`` minus the first season of the coach's
  consecutive run at that school, counting a season only when the coach
  handled the majority of the school's games (the "first season" a coaching
  list reports), 0 in the first season.
* the 25 team-meta columns -- CFBD ``/teams`` (identity block + the school's
  home venue inline), current-season values, home-team venue even at neutral
  sites, as the source did.
* the 10 weather columns -- CFBD ``/games/weather``, verbatim, first row per
  game, joined by ``game_id`` only.

That fills 68 of the line's 90 side / meta / weather columns (4 side inputs
x 2 sides, 25 meta x 2, 10 weather). The other 22 (11 per side) come from the
versioned tables in ``cfb_data_build.matchup_reference``: ``team_talent_weighted`` (a
rank-decayed roster sum from a recruiting site), ``off/def/ovr_rtprod`` (an
external returning-production table whose numbers are not CFBD's
``percentPPA``), ``oc_cont`` / ``dc_cont`` (coordinator continuity from a
wiki scrape), and the REALIZED-starter QB block (post-hoc: the season's
most-attempts passer, not a pregame projection) ``athlete_id`` / ``qb_name``
/ ``returning_qb`` / ``qb_starter_years`` / ``qb_games``.
"""

from __future__ import annotations

import os

import polars as pl

from cfb_data_build.matchup_line import (
    SIDE_INPUT_COLS,
    TEAM_META_COLS,
    WEATHER_COLS,
    _column_dtype,
)

CFBD = "https://api.collegefootballdata.com"
#: FBS schools absent from CFBD /talent every season; the source scored them 0
ACADEMIES_NO_TALENT = ("Air Force", "Navy")


def _cfbd(path: str, *, api_key: str | None = None) -> list[dict]:
    """One CFBD GET through the repo's pooled + retrying client; error bodies raise."""
    from sportsdataverse.dl_utils import download

    key = api_key or os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(
            f"CFBD_API_KEY is not set -- the matchup side inputs read CFBD {path}"
        )
    payload = download(
        f"{CFBD}{path}",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    ).json()
    if not isinstance(payload, list):
        raise RuntimeError(f"CFBD {path}: unexpected body {type(payload).__name__}")
    return payload


def fetch_cfbd_talent(season: int) -> list[dict]:
    return _cfbd(f"/talent?year={season}")


def fetch_cfbd_weather(season: int) -> list[dict]:
    return _cfbd(f"/games/weather?year={season}&seasonType=both")


def fetch_cfbd_teams(season: int) -> list[dict]:
    return _cfbd(f"/teams?year={season}")


def fetch_cfbd_coaches(season: int, *, min_year: int = 1990) -> list[dict]:
    """Every coach-season from ``min_year`` through ``season`` (tenure needs history)."""
    return _cfbd(f"/coaches?minYear={min_year}&maxYear={season}")


def _typed(frame: pl.DataFrame, cols: tuple[str, ...]) -> pl.DataFrame:
    """Cast the named columns to their documented line dtype; refuse a lossy int cast."""
    # numeric text (CFBD ships elevation as a string) becomes Float64 FIRST, so
    # the lossy-cast check below sees the parsed value; a non-numeric string
    # raises here rather than silently becoming null
    present = [c for c in cols if c in frame.columns]
    frame = frame.with_columns(
        [
            pl.col(c).cast(pl.Float64, strict=True)
            for c in present
            if frame.schema[c] == pl.Utf8 and _column_dtype(c) in (pl.Float64, pl.Int64)
        ]
    )
    exprs = []
    for c in cols:
        target = _column_dtype(c)
        if c not in frame.columns:
            exprs.append(pl.lit(None, dtype=target).alias(c))
            continue
        if target == pl.Int64 and frame.schema[c] in (pl.Float64, pl.Float32):
            bad = frame.filter(
                pl.col(c).is_not_null() & (pl.col(c) != pl.col(c).round(0))
            )
            if bad.height:
                raise ValueError(
                    f"{c}: non-integer values but documented Int64: "
                    f"{bad[c].head(3).to_list()}"
                )
        exprs.append(pl.col(c).cast(target).alias(c))
    return frame.with_columns(exprs)


def tidy_cfbd_talent(payload: list[dict]) -> pl.DataFrame:
    """``/talent`` objects -> (season, team, talent)."""
    rows = [
        {"season": x.get("year"), "team": x.get("team"), "talent": x.get("talent")}
        for x in payload
    ]
    return pl.DataFrame(
        rows, schema={"season": pl.Int64, "team": pl.Utf8, "talent": pl.Float64}
    )


_WEATHER_FIELDS = {
    "temperature": "temperature",
    "dewPoint": "dew_point",
    "humidity": "humidity",
    "precipitation": "precipitation",
    "snowfall": "snowfall",
    "windDirection": "wind_direction",
    "windSpeed": "wind_speed",
    "pressure": "pressure",
    "weatherConditionCode": "weather_condition_code",
    "weatherCondition": "weather_condition",
}


def tidy_cfbd_weather(payload: list[dict]) -> pl.DataFrame:
    """``/games/weather`` objects -> ``game_id`` + the 10 weather columns, first row per game."""
    rows = [
        {
            "game_id": x.get("id"),
            **{out: x.get(src) for src, out in _WEATHER_FIELDS.items()},
        }
        for x in payload
    ]
    if not rows:
        return pl.DataFrame(
            schema={"game_id": pl.Int64, **{c: _column_dtype(c) for c in WEATHER_COLS}}
        )
    frame = pl.DataFrame(rows, schema_overrides={"game_id": pl.Int64})
    return (
        _typed(frame, WEATHER_COLS)
        .unique(subset=["game_id"], keep="first", maintain_order=True)
        .select(["game_id", *WEATHER_COLS])
    )


_META_TOP = {
    "mascot": "mascot",
    "abbreviation": "abbreviation",
    "classification": "classification",
    "color": "color",
    "alternateColor": "alt_color",
    "twitter": "twitter",
}
_META_LOC = {
    "id": "venue_id",
    "name": "venue_name",
    "city": "city",
    "state": "state",
    "zip": "zip",
    "countryCode": "country_code",
    "timezone": "timezone",
    "latitude": "latitude",
    "longitude": "longitude",
    "elevation": "elevation",
    "capacity": "capacity",
    "constructionYear": "year_constructed",
    "grass": "grass",
    "dome": "dome",
}


def tidy_cfbd_teams(payload: list[dict]) -> pl.DataFrame:
    """``/teams`` objects -> one row per school: ``team``, ``join_name`` + the 25 meta columns."""
    rows = []
    for x in payload:
        alts = [*(x.get("alternateNames") or [])[:3], None, None, None]
        logos = [*(x.get("logos") or [])[:2], None, None]
        loc = x.get("location") or {}
        school = x.get("school")
        mascot = x.get("mascot") or ""
        rows.append(
            {
                "team": school,
                "join_name": f"{school} {mascot}".strip(),
                **{out: x.get(src) for src, out in _META_TOP.items()},
                "alt_name1": alts[0],
                "alt_name2": alts[1],
                "alt_name3": alts[2],
                "logo": logos[0],
                "logo_2": logos[1],
                **{out: loc.get(src) for src, out in _META_LOC.items()},
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "team": pl.Utf8,
                "join_name": pl.Utf8,
                **{c: _column_dtype(c) for c in TEAM_META_COLS},
            }
        )
    frame = pl.DataFrame(rows, schema_overrides={"team": pl.Utf8, "join_name": pl.Utf8})
    return _typed(frame, TEAM_META_COLS).select(["team", "join_name", *TEAM_META_COLS])


def tidy_cfbd_coaches(payload: list[dict]) -> pl.DataFrame:
    """``/coaches`` objects -> one row per (coach, team, season) with the games coached."""
    rows = [
        {
            "coach": f"{x.get('firstName') or ''} {x.get('lastName') or ''}".strip(),
            "team": s.get("school"),
            "season": s.get("year"),
            "games": s.get("games"),
        }
        for x in payload
        for s in x.get("seasons") or []
    ]
    return pl.DataFrame(
        rows,
        schema={
            "coach": pl.Utf8,
            "team": pl.Utf8,
            "season": pl.Int64,
            "games": pl.Int64,
        },
    ).unique(maintain_order=True)


def head_coaches(coaches: pl.DataFrame, season: int) -> pl.DataFrame:
    """Per school: the season's head coach and ``hc_tenure``.

    The source took the PRESEASON head coach (a scrape made before the
    season), so a mid-season change keeps the incumbent: among the school's
    coaches that season, the one with the longest consecutive run at the
    school wins, then the most games. ``hc_tenure`` is ``season`` minus the
    first season of that run, where a season belongs to the run only if the
    coach handled the majority of the school's games (an interim bowl or a
    suspension stint does not start a tenure -- the same "first season" a
    coaching list reports), and is 0 in the first season.
    """
    team_games = coaches.group_by("team", "season").agg(
        pl.col("games").sum().alias("team_games")
    )
    majority = (
        coaches.join(team_games, on=["team", "season"], how="left")
        .filter(pl.col("games") * 2 > pl.col("team_games"))
        .select("team", "coach", "season")
        .unique()
    )
    cur = coaches.filter(pl.col("season") == season).select("team", "coach", "games")
    runs = (
        cur.join(
            majority.filter(pl.col("season") < season), on=["team", "coach"], how="left"
        )
        .group_by("team", "coach", "games", maintain_order=True)
        .agg(pl.col("season").drop_nulls().unique().sort(descending=True))
    )
    rows = []
    for team, coach, games, seasons in runs.iter_rows():
        first = season
        for y in seasons:
            if y == first - 1:
                first = y
            else:
                break
        rows.append(
            {"team": team, "coach": coach, "games": games, "tenure": season - first}
        )
    frame = pl.DataFrame(
        rows,
        schema={
            "team": pl.Utf8,
            "coach": pl.Utf8,
            "games": pl.Int64,
            "tenure": pl.Int64,
        },
    )
    return (
        frame.sort(
            ["team", "tenure", "games", "coach"],
            descending=[False, True, True, False],
            nulls_last=True,
        )
        .unique(subset=["team"], keep="first", maintain_order=True)
        .select(
            "team",
            pl.col("coach").alias("head_coach"),
            pl.col("tenure").alias("hc_tenure"),
        )
    )


def team_side_inputs(
    season: int,
    *,
    talent: pl.DataFrame,
    teams: pl.DataFrame,
    coaches: pl.DataFrame,
    reference: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per CFBD school: the 15 side-input columns + the 25 meta columns.

    Four columns come straight from CFBD (``join_name``, ``talent``,
    ``head_coach``, ``hc_tenure``). The other eleven come from ``reference``
    (:func:`cfb_data_build.matchup_reference.season_reference`); without it
    they stay null with their documented dtype.

    Keyed by ``team`` in CFBD's spelling; the line builder maps it to the
    delivered spelling when it prefixes the columns per side.
    """
    out = (
        teams.join(
            talent.filter(pl.col("season") == season).select("team", "talent"),
            on="team",
            how="left",
        )
        .join(head_coaches(coaches, season), on="team", how="left")
        # the service academies never appear in /talent; the source pinned them
        # to 0 rather than leaving the only two FBS holes null
        .with_columns(
            talent=pl.when(
                pl.col("talent").is_null() & pl.col("team").is_in(ACADEMIES_NO_TALENT)
            )
            .then(0.0)
            .otherwise(pl.col("talent"))
        )
    )
    if reference is not None:
        extra = [c for c in SIDE_INPUT_COLS if c in reference.columns]
        assert reference.schema["team"] == out.schema["team"]
        out = out.join(reference.select(["team", *extra]), on="team", how="left")
    return _typed(out, SIDE_INPUT_COLS).select(
        ["team", *SIDE_INPUT_COLS, *TEAM_META_COLS]
    )


def side_input_frames(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(per-team side inputs + meta, per-game weather) for one season.

    CFBD supplies talent, coaches, team meta and weather; the versioned
    reference tables under ``data/`` supply the eleven columns CFBD does not
    publish.
    """
    from cfb_data_build.matchup_reference import season_reference

    teams = tidy_cfbd_teams(fetch_cfbd_teams(season))
    side = team_side_inputs(
        season,
        talent=tidy_cfbd_talent(fetch_cfbd_talent(season)),
        teams=teams,
        coaches=tidy_cfbd_coaches(fetch_cfbd_coaches(season)),
        reference=season_reference(
            season, teams=teams, schools=teams["team"].to_list()
        ),
    )
    return side, tidy_cfbd_weather(fetch_cfbd_weather(season))
