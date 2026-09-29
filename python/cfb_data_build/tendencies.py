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

The game-context splits (home / away / neutral site / vs ranked / after bye /
opener / one-score game, each with games and wins) come from
:func:`attach_context`, which joins the season's ``cfb_schedules`` onto the
plays as the ``ctx_*`` / ``def_ctx_*`` columns sdv-py reads.
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
#: rest before a regular-season game, in days (rounded), that makes it "after a bye"
BYE_REST_DAYS = 12
#: final margin within which a game is a one-score game
ONE_SCORE_MARGIN = 8
#: games that were scheduled but never played: never a team's previous game or opener
NOT_PLAYED = ("STATUS_CANCELED", "STATUS_POSTPONED")
_KEYS = ("game_id", "pos_team", "def_pos_team")


def game_context(schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (``game_id``, ``team_id``) of the team's Boolean context flags.

    ``schedule`` is one season of ``cfb_schedules``, every division, so a rest
    or an opener counts games against any opponent. The flags:

    * ``home`` / ``away`` / ``neutral_site`` -- exactly one per game;
    * ``vs_ranked`` -- the OPPONENT was an FBS top-25 team (rank 1-25) at kickoff;
      the IF-0 ranks exist only for FBS. ABSENT, not all-false, unless the
      schedule has ``home_rank`` / ``away_rank`` with at least one value: a
      pre-IF-0 release, or a season with no ranks, must not read as "never
      played a ranked team";
    * ``after_bye`` -- a REGULAR-season game at least :data:`BYE_REST_DAYS` days
      after the team's previous played game. Kickoffs are UTC instants, so the
      rest is rounded to the nearest day (a Saturday-night game to a Thursday
      one 12 days later can be 11.9 days). 401 of 402 CFB regular-season games
      at a rounded 12 days follow a skipped week (Saturday -> Thursday).
      Postseason games never count: most follow 2+ weeks off (1,608 of 1,762
      postseason team-games would), which would swamp the split. The season's
      first game is NEVER after a bye: it has no previous game in the season,
      and last season's schedule is not read. Known ceiling: a Saturday ->
      Tuesday / Wednesday bye (10-11 days) is excluded, because those rests
      overlap non-bye Tuesday -> Saturday gaps of the same length;
    * ``opener`` -- the team's first played regular-season game;
    * ``one_score_game`` -- final margin within :data:`ONE_SCORE_MARGIN`;
    * ``win`` -- points for > points against, so a tie is not a win.

    Unscored games carry null ``win`` / ``one_score_game`` (null is not true).
    """
    played = schedule.filter(~pl.col("status").is_in(NOT_PLAYED).fill_null(False))
    ranked = {"home_rank", "away_rank"} <= set(schedule.columns) and bool(
        schedule.select(
            pl.any_horizontal(pl.col("home_rank", "away_rank").is_not_null()).any()
        ).item()
    )
    neutral = pl.col("neutral_site").fill_null(False)
    sides = [
        played.select(
            pl.col("game_id").cast(pl.Int64),
            pl.col(f"{me}_id").cast(pl.Int64).alias("team_id"),
            # "…T19:30:00.000Z" and the ESPN-only "…T19:30Z" share the first 16 chars
            pl.col("start_date")
            .str.slice(0, 16)
            .str.to_datetime("%Y-%m-%dT%H:%M")
            .alias("kickoff"),
            (pl.col("season_type_id") == 2).alias("regular"),
            (pl.lit(me == "home") & ~neutral).alias("home"),
            (pl.lit(me == "away") & ~neutral).alias("away"),
            neutral.alias("neutral_site"),
            *(
                [pl.col(f"{opp}_rank").is_between(1, 25).alias("vs_ranked")]
                if ranked
                else []
            ),
            (
                (pl.col("home_points") - pl.col("away_points")).abs()
                <= ONE_SCORE_MARGIN
            ).alias("one_score_game"),
            (pl.col(f"{me}_points") > pl.col(f"{opp}_points")).alias("win"),
        )
        for me, opp in (("home", "away"), ("away", "home"))
    ]
    rest = (pl.col("kickoff") - pl.col("kickoff").shift(1)).dt.total_minutes() / 1440
    return (
        pl.concat(sides)
        .drop_nulls(["game_id", "team_id", "kickoff"])
        # game_id breaks kickoff ties so opener / after_bye never flip between runs
        # (CFBD carries phantom duplicate games, e.g. Rhodes v Rhode Island 2004-08)
        .sort("team_id", "kickoff", "game_id")
        .with_columns(
            (pl.col("regular") & (rest.round(0) >= BYE_REST_DAYS))
            .over("team_id")
            .fill_null(False)
            .alias("after_bye"),
            (
                pl.col("regular") & (pl.col("regular").cum_sum().over("team_id") == 1)
            ).alias("opener"),
        )
        .drop("kickoff", "regular")
    )


def attach_context(plays: pl.DataFrame, schedule: pl.DataFrame | None) -> pl.DataFrame:
    """``ctx_*`` (the offense's context) and ``def_ctx_*`` (the defense's) on every play.

    Joined from :func:`game_context` on (``game_id``, ``pos_team``) and on
    (``game_id``, ``def_pos_team``). ``game_id`` and the team ids are cast to
    Int64 on both sides here -- an integer parse that raises on a non-integer id
    rather than nulling a join key -- and asserted equal before the join. A
    team-game the schedule lacks (a missing game, a team-id mismatch) keeps
    null context and is counted in the log, never silently. A ``None`` schedule
    returns the plays unchanged: no context columns, so sdv-py emits no context
    splits.
    """
    if schedule is None:
        return plays
    ctx = game_context(schedule)
    flags = [c for c in ctx.columns if c not in ("game_id", "team_id")]
    df = plays.with_columns(pl.col(*_KEYS).cast(pl.Int64))
    unmatched = (
        df.select("game_id", team_id="pos_team")
        .drop_nulls()
        .unique()
        .join(ctx, on=["game_id", "team_id"], how="anti")
    )
    if unmatched.height:
        print(
            f"  context: {unmatched.height} team-game(s) in "
            f"{unmatched['game_id'].n_unique()} game(s) not in cfb_schedules; no game context"
        )
    for side, prefix in (("pos_team", "ctx_"), ("def_pos_team", "def_ctx_")):
        right = ctx.select(
            "game_id",
            pl.col("team_id").alias(side),
            *(pl.col(f).alias(prefix + f) for f in flags),
        )
        for k in ("game_id", side):
            assert df.schema[k] == right.schema[k] == pl.Int64, k
        df = df.join(right, on=["game_id", side], how="left", validate="m:1")
    return df


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
        # built from a schema, not with_columns(lit): a literal on a frame with
        # no columns broadcasts to ONE row, which would then be a "play"
        return pl.DataFrame(
            schema={**plays.schema, "coach": pl.Utf8, "def_coach": pl.Utf8}
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


def team_tendencies(
    plays: pl.DataFrame, schedule: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per (season, team); ``schedule`` adds the game-context splits."""
    plays = counted_plays(plays)
    if plays.height == 0:
        return pl.DataFrame()
    from sportsdataverse.football.tendencies import tendencies

    return tendencies(
        attach_context(plays, schedule), league=LEAGUE, group_cols=TEAM_GROUP
    )


def coach_tendencies(
    plays: pl.DataFrame, coaches: pl.DataFrame, schedule: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per (season, team, head coach), ``role`` = ``"HC"``; ``schedule`` as in
    :func:`team_tendencies`."""
    df = attach_coaches(counted_plays(plays), coaches)
    if df.height == 0:
        return pl.DataFrame()
    from sportsdataverse.football.tendencies import tendencies

    out = tendencies(
        attach_context(df, schedule),
        league=LEAGUE,
        group_cols=COACH_GROUP,
        def_group_cols=COACH_DEF_GROUP,
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
