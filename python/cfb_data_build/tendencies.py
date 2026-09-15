"""Season tendencies per team and per head coach (``sportsdataverse.football.tendencies``).

* ``team_tendencies`` -- one row per (season, team): pace, run/pass splits by
  down and score state, situation-neutral rates, explosive / success / EPA,
  third downs over expected, red-zone and scoring-opportunity efficiency,
  scripted vs non-scripted drives, fourth-down decisions against the bundled
  model, plus the ``def_*`` twin (what the defense allowed).
* ``coach_tendencies`` -- the same row per (season, team, head coach), the
  coach from :mod:`cfb_data_build.coaches` (team-season attribution).
* ``coach_careers`` -- every written ``coach_tendencies`` season summed per
  coach with the rates recomputed (one season-less file).

Each is computed at BUILD time from the season's full-tier pbp frames, so the
installed sdv-py defines every metric. Coordinators are not attributed (no
source), so ``role`` is always ``"HC"`` and exists for a later OC / DC row.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

LEAGUE = "cfb"
#: ESPN season types that count: regular season (2) and postseason (3)
COUNTED_SEASON_TYPES = (2, 3)
TEAM_GROUP = ("season", "pos_team")
COACH_GROUP = ("season", "pos_team", "coach")
COACH_DEF_GROUP = ("season", "def_pos_team", "def_coach")


def counted_plays(plays: pl.DataFrame) -> pl.DataFrame:
    """Regular + postseason snaps only."""
    if plays.height and "seasonType" in plays.columns:
        return plays.filter(
            pl.col("seasonType")
            .cast(pl.Int64, strict=False)
            .is_in(COUNTED_SEASON_TYPES)
        )
    return plays


def attach_coaches(plays: pl.DataFrame, coaches: pl.DataFrame) -> pl.DataFrame:
    """``coach`` / ``def_coach`` on every play from ``(season, team_id, coach)`` rows.

    A play is kept when EITHER side is attributed: an attributed offense keeps
    every snap it ran (also against an unattributed FCS opponent), and an
    attributed defense keeps every snap it faced, so a coach row equals the
    team-season it was cut from. The other side's null key forms no row
    (:func:`coach_tendencies` drops it). Plays with neither side attributed go.
    """
    if plays.height == 0 or coaches.height == 0:
        return plays.head(0).with_columns(
            coach=pl.lit(None, dtype=pl.Utf8), def_coach=pl.lit(None, dtype=pl.Utf8)
        )
    lookup = coaches.select(
        pl.col("season").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64),
        pl.col("coach").cast(pl.Utf8),
    ).unique(subset=["season", "team_id"], keep="first")
    df = plays.with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("pos_team").cast(pl.Int64, strict=False),
        pl.col("def_pos_team").cast(pl.Int64, strict=False),
    )
    assert df.schema["pos_team"] == lookup.schema["team_id"]
    df = df.join(
        lookup.rename({"team_id": "pos_team"}), on=["season", "pos_team"], how="left"
    )
    df = df.join(
        lookup.rename({"team_id": "def_pos_team", "coach": "def_coach"}),
        on=["season", "def_pos_team"],
        how="left",
    )
    return df.filter(pl.col("coach").is_not_null() | pl.col("def_coach").is_not_null())


def require_coaches(coaches: pl.DataFrame, season: int) -> pl.DataFrame:
    """Refuse to cut a coach season from an empty roster instead of silently writing nothing."""
    if coaches.height == 0:
        raise RuntimeError(
            f"coach_tendencies {season}: no attributed coach-seasons in data/cfb_coach_seasons.csv; "
            f"refresh it with `python -m cfb_data_build.coaches -s {season} -e {season}` (needs CFBD_API_KEY)"
        )
    return coaches


def team_tendencies(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, team)."""
    plays = counted_plays(plays)
    if plays.height == 0:
        return pl.DataFrame()
    from sportsdataverse.football.tendencies import tendencies

    return tendencies(plays, league=LEAGUE, group_cols=TEAM_GROUP)


def coach_tendencies(plays: pl.DataFrame, coaches: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, team, head coach), ``role`` = ``"HC"``."""
    df = attach_coaches(counted_plays(plays), coaches)
    if df.height == 0:
        return pl.DataFrame()
    from sportsdataverse.football.tendencies import tendencies

    out = tendencies(
        df, league=LEAGUE, group_cols=COACH_GROUP, def_group_cols=COACH_DEF_GROUP
    )
    out = out.filter(
        pl.col("coach").is_not_null()
    )  # the unattributed side's snaps form no row
    return out.with_columns(role=pl.lit("HC")).select(
        "season",
        "pos_team",
        "coach",
        "role",
        pl.exclude("season", "pos_team", "coach", "role"),
    )


def coach_careers(season_files: list[Path]) -> pl.DataFrame:
    """Sum every written ``coach_tendencies`` season per coach; rates recomputed.

    ``teams`` lists the schools coached (first to last season). Careers cover
    only the seasons present on disk, so a partial output root yields partial
    careers -- the build prints which seasons went in.
    """
    frames = [f for f in (pl.read_parquet(p) for p in season_files) if f.height]
    if not frames:
        return pl.DataFrame()
    from sportsdataverse.football.tendencies import aggregate_tendencies

    careers = aggregate_tendencies(frames, keys=("coach",))
    teams = (
        pl.concat(frames, how="diagonal_relaxed")
        .sort(["season", "pos_team"])
        .group_by("coach", maintain_order=True)
        .agg(
            pl.col("pos_team")
            .cast(pl.Utf8)
            .drop_nulls()
            .unique(maintain_order=True)
            .str.join(", ")
            .alias("teams")
        )
    )
    careers = careers.join(teams, on="coach", how="left").with_columns(
        role=pl.lit("HC")
    )
    front = ["coach", "role", "teams", "seasons", "first_season", "last_season"]
    return careers.select(*front, pl.exclude(front)).sort("plays", descending=True)
