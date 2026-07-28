"""Build the ``plays_input`` frame for :mod:`cfb_data_build.team_summaries`
from the PYTHON side (released ``espn_cfb_pbp`` + ``espn_cfb_schedule``),
replacing the R ``cfbfastR::load_cfb_pbp`` capture path.

Why: R's ``load_cfb_pbp`` floors at 2014, capping the released
``espn_cfb_team_summaries`` family there. The Python release covers 2004+,
so this prep unlocks the 2004-2013 backfill.

The contract this satisfies (``team_summaries.build_team_summaries`` input,
mirroring R build lines 519-553):

* schedule-joined (neutral_site, divisions, conferences, home/away ids)
* FBS-vs-FBS games only, pass|rush scrimmage plays only
* kneel-downs stripped (text regex + end-of-half anonymized-TEAM-run window)
* cfbfastR-schema column names (the released asset ships ESPN's dotted names
  for a handful; aliased here) plus the derived per-role receiver columns the
  R pbp carries natively (target/completion/incompletion/reception player,
  interception-thrown / sack-taken ids).

The bad-game exclusion list mirrors the R build (401635537: corrupt 2024 feed).
"""

from __future__ import annotations

import polars as pl

# Bad games excluded by the R build (espn_cfb_15_team_summaries_creation.R).
_BAD_GAME_IDS = ("401635537",)

# Kneel-down heuristics (gameonpaper team_agg.R / sdv-py cfb_ratings parity).
_KNEEL_TEXT = r"(?i)kneel|takes a knee"
_KNEEL_TEAM_RUN = r"(?i)^team run for a loss of (?:1 yard|2 yards)"

# released espn_cfb_pbp name -> cfbfastR-schema name the builder expects.
_ALIASES: dict[str, str] = {
    "statYardage": "yards_gained",
    "EPA_success": "epa_success",
    "drive.id": "drive_id",
    "start.yardsToEndzone": "yards_to_goal",
    "start.pos_team.id": "pos_team_id",
    "homeTeamId": "home_id",
    "awayTeamId": "away_id",
    "type.text": "play_type",
    "sack": "sack_vec",
    "drive.yards": "drive_yards",
    "rusher_player_id": "rush_player_id",
}


def prepare_plays_input(
    pbp: pl.DataFrame, schedule: pl.DataFrame, season: int
) -> pl.DataFrame:
    """Released pbp + schedule -> the ``build_team_summaries`` input frame."""
    df = pbp.with_columns(pl.col("game_id").cast(pl.Utf8))
    sched = schedule.with_columns(pl.col("game_id").cast(pl.Utf8))

    # --- aliases (only when the canonical name is absent) ---
    ren = {
        src: dst
        for src, dst in _ALIASES.items()
        if dst not in df.columns and src in df.columns
    }
    df = df.rename(ren)

    # --- old-era releases (2004+) ship down/distance ~96% null with the
    # start.* variants fully populated — coalesce so success/third-down
    # families survive ---
    if "start.down" in df.columns:
        df = df.with_columns(
            pl.coalesce(pl.col("down"), pl.col("start.down")).alias("down"),
            pl.coalesce(pl.col("distance"), pl.col("start.distance")).alias("distance"),
        )

    # --- pre-participants-era releases (2004-13) ship player NAMES but no
    # per-play athlete ids (and no pass_breakup_player_name). Null-fill so the
    # team tables build; the player tables filter to non-null ids and come out
    # empty for those seasons (write_dataset skips empty frames, matching R). ---
    _ATHLETE_FALLBACKS = (
        "receiver_player_id",
        "passer_player_id",
        "rush_player_id",
        "pass_breakup_player_name",
    )
    df = df.with_columns(
        [
            pl.lit(None, dtype=pl.Utf8).alias(c)
            for c in _ATHLETE_FALLBACKS
            if c not in df.columns
        ]
    )

    # --- schedule join: divisions / conferences / neutral flag ---
    sched_cols = ["game_id"]
    for c in (
        "neutral_site",
        "home_division",
        "away_division",
        "home_conference",
        "away_conference",
    ):
        if c in sched.columns and c not in df.columns:
            sched_cols.append(c)
    df = df.join(
        sched.select(sched_cols).unique(subset=["game_id"]), on="game_id", how="left"
    )
    if "home_division" in df.columns:
        df = df.rename(
            {
                "home_division": "home_team_division",
                "away_division": "away_team_division",
            }
        )
    if "home_conference" in df.columns:
        df = df.rename(
            {
                "home_conference": "home_team_conference",
                "away_conference": "away_team_conference",
            }
        )

    # --- cfbfastR home/away are the same namespace as pos_team (ids here) ---
    exprs: list[pl.Expr] = []
    if "home" not in df.columns:
        exprs.append(pl.col("home_id").cast(pl.Utf8).alias("home"))
    if "away" not in df.columns:
        exprs.append(pl.col("away_id").cast(pl.Utf8).alias("away"))
    df = df.with_columns(exprs) if exprs else df
    df = df.with_columns(
        pl.col("pos_team").cast(pl.Utf8),
        pl.col("pos_team_id").cast(pl.Utf8),
        pl.col("home_id").cast(pl.Utf8),
        pl.col("away_id").cast(pl.Utf8),
        pl.lit(season, dtype=pl.Int64).alias("season"),
    )
    # R pbp carries BOTH home_id and home_team_id; the release ships one.
    if "home_team_id" not in df.columns:
        df = df.with_columns(
            pl.col("home_id").alias("home_team_id"),
            pl.col("away_id").alias("away_team_id"),
        )

    # --- 0/1 flag dtypes: the release ships Booleans; the R-schema builder does
    # arithmetic on doubles (bool * numeric is unsupported in polars) ---
    _FLAGS = (
        "pass",
        "rush",
        "pass_attempt",
        "completion",
        "target",
        "int",
        "sack_vec",
        "fumble_vec",
        "pass_td",
        "rush_td",
        "epa_success",
    )
    df = df.with_columns(
        [
            pl.col(c).cast(pl.Float64)
            for c in _FLAGS
            if c in df.columns and df.schema[c] != pl.Float64
        ]
    )

    # --- scrimmage + FBS/FBS + bad games ---
    df = df.filter(
        ((pl.col("pass") == 1) | (pl.col("rush") == 1))
        & ~pl.col("game_id").is_in(list(_BAD_GAME_IDS))
        & (pl.col("home_team_division") == "fbs")
        & (pl.col("away_team_division") == "fbs")
    )

    # --- kneel-downs (regex on text + end-of-half TEAM-run window) ---
    text = pl.col("text").cast(pl.Utf8)
    clock = pl.col("start.adj_TimeSecsRem").cast(pl.Float64)
    half_end = ((clock <= 1860) & (clock >= 1800)) | ((clock <= 60) & (clock >= 0))
    kneel = text.str.contains(_KNEEL_TEXT).fill_null(False) | (
        half_end & text.str.contains(_KNEEL_TEAM_RUN)
    ).fill_null(False)
    df = df.filter((pl.col("pass") == 1) | (kneel == False))  # noqa: E712

    # --- rule-based success (cfbfastR `success`; absent from the release) ---
    if "success" not in df.columns:
        gain = pl.col("yards_gained").cast(pl.Float64)
        dist = pl.col("distance").cast(pl.Float64)
        df = df.with_columns(
            pl.when(pl.col("down") == 1)
            .then(gain >= 0.5 * dist)
            .when(pl.col("down") == 2)
            .then(gain >= 0.7 * dist)
            .when(pl.col("down").is_in([3, 4]))
            .then(gain >= dist)
            .otherwise(False)
            .cast(pl.Float64)
            .alias("success")
        )

    # --- per-side, per-type play EPA (R pbp natives home_EPA_pass etc.) ---
    is_home_off = pl.col("pos_team") == pl.col("home")
    epa = pl.col("EPA").cast(pl.Float64)
    df = df.with_columns(
        pl.when(is_home_off & (pl.col("pass") == 1)).then(epa).alias("home_EPA_pass"),
        pl.when(is_home_off & (pl.col("rush") == 1)).then(epa).alias("home_EPA_rush"),
        pl.when(~is_home_off & (pl.col("pass") == 1)).then(epa).alias("away_EPA_pass"),
        pl.when(~is_home_off & (pl.col("rush") == 1)).then(epa).alias("away_EPA_rush"),
    )

    # --- per-role receiver columns (R pbp natives; derived from flags here) ---
    recv, recv_id = pl.col("receiver_player_name"), pl.col("receiver_player_id")
    is_target = pl.col("target") == 1
    is_comp = pl.col("completion") == 1
    # CFBD stat-type semantics (cfbd_play.R pivot): completion/incompletion are
    # charged to the QB; reception/target to the receiver. INTs and sacks are
    # charged separately (interception_thrown / sack_taken), so incompletion
    # excludes them — the builder revives those via its name-map aggregation.
    psr, psr_id = pl.col("passer_player_name"), pl.col("passer_player_id")
    is_incomp = (
        (pl.col("pass_attempt") == 1)
        & ~is_comp
        & (pl.col("int") == 0)
        & (pl.col("sack_vec") == 0)
    )
    df = df.with_columns(
        pl.when(is_target).then(recv).alias("target_player"),
        pl.when(is_target).then(recv_id).alias("target_player_id"),
        pl.when(is_comp).then(psr).alias("completion_player"),
        pl.when(is_comp).then(psr_id).alias("completion_player_id"),
        pl.when(is_incomp).then(psr).alias("incompletion_player"),
        pl.when(is_incomp).then(psr_id).alias("incompletion_player_id"),
        pl.when(is_comp).then(recv).alias("reception_player"),
        pl.when(is_comp).then(recv_id).alias("reception_player_id"),
        pl.when(pl.col("int") == 1)
        .then(pl.col("passer_player_id"))
        .alias("interception_thrown_player_id"),
        pl.when(pl.col("sack_vec") == 1)
        .then(pl.col("passer_player_id"))
        .alias("sack_taken_player_id"),
    )

    # --- drive start position: possession-oriented yards-to-goal from the
    # drive header. `drive.start.yardLine` is the absolute coordinate measured
    # from the HOME goal line, so the home offense flips it. (The R oracle's
    # column is CFBD's native possession-oriented drive field; validated
    # empirically vs the released 2024 table — mean 71.6 vs R 71.8.) ---
    if "drive_start_yards_to_goal" not in df.columns:
        yard = pl.col("drive.start.yardLine").cast(pl.Float64)
        df = df.with_columns(
            drive_start_yards_to_goal=pl.when(
                pl.col("pos_team_id") == pl.col("home_id")
            )
            .then(100 - yard)
            .otherwise(yard)
        )

    return df.sort(["game_id", "game_play_number"])
