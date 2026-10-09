"""Season team + player summaries ("Binion Box Score") -- polars/sklearn port of
``R/espn_cfb_15_team_summaries_creation.R``.

Unlike the other datasets this is a **season-level aggregation off a full
cfbfastR-schema season pbp** (``cfbfastR::load_cfb_pbp`` in R; the analogous
``sportsdataverse.cfb.load_cfb_pbp`` in Python). It produces 6 released tables:
``percentiles``, ``team_summaries``, ``passing``, ``rushing``, ``receiving``,
``league_averages``.

This module IS the live producer. R's ``espn_cfb_15`` was retired once the P2
pbp rebuild made the Python season-pbp source current -- ``scripts/
daily_cfb_processor.sh`` runs ``run_py summaries --publish``. Several workflow
comments still credit the R script; they are stale.

Opponent-adjusted EPA is delegated to the shared
:func:`sportsdataverse.cfb.cfb_adjusted_epa` primitive (single owner since sdv-py
0.0.71; was a local copy here). Parity caveat: the deterministic aggregations
match R exactly, but that ridge regression (R uses ``glmnet`` at the grid's
largest lambda; sdv-py uses ``sklearn`` ridge on standardized opponent dummies)
does not byte-match, so the opponent-adjusted EPA columns (``adj_off_epa`` /
``adj_def_epa`` / ``net_adj_epa`` + strengths/ranks) are held to a
**correlation** bar, not exact equality (see the integration parity test).

Input contract: a ``plays`` frame already schedule-joined, FBS/FBS + pass/rush
filtered, kneel-down filtered, and ``clean_play_text``-ed (R build lines
523-553). :func:`build_team_summaries` applies the derived-metrics mutate and
everything downstream.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import polars as pl
import sportsdataverse
from sportsdataverse.cfb import cfb_adjusted_epa, load_cfb_team_group_seasons

from .checks import assert_adjustment_is_real, assert_passer_epa_includes_sacks
from .league_averages import build_league_averages
from .tendencies import ONE_SCORE_MARGIN

# Explosive-play EPA thresholds (R build lines 604-608).
_EXPLOSIVE_PASS_EPA = 2.4
_EXPLOSIVE_RUSH_EPA = 1.8
#: A scoring opportunity is a drive with a snap at or inside the opponent 40:
#: sdv-py's per-play ``scoring_opp`` (``start.yardsToEndzone <= 40``), the flag
#: the shared ``tendencies`` producer reads. ``yards_to_goal`` is that column here.
_SCORING_OPP_YARDS_TO_GOAL = 40
#: Offensive points by ESPN ``drive.result`` (7 per TD, as ``tendencies`` scores
#: it), for both points per scoring opportunity and points per drive. 2004-2013
#: spell the outcomes out -- "RUSHING TD", "PASSING TD", "FG GOOD", "MADE FG",
#: and 2004 has no plain "TD"/"FG" drive at all -- so a "TD"/"FG"-only match
#: scores those seasons at zero. Return TDs ("INT TD", "PUNT RETURN TD", ...)
#: are the other team's points: 0 here.
#:
#: Known ceiling: a drive whose result is a clock or data label rather than an
#: outcome scores 0 even when it ended in a field goal -- over 2004-2025 that is
#: 618 "END OF HALF", 444 "Not provided" and 127 "END OF GAME" drives. Scoring
#: them needs the drive's plays, not its label.
_DRIVE_POINTS: dict[str, float] = {
    "TD": 7.0,
    "RUSHING TD": 7.0,
    "PASSING TD": 7.0,
    "RUSHING TD TD": 7.0,
    "PASSING TD TD": 7.0,
    "FG": 3.0,
    "FG GOOD": 3.0,
    "MADE FG": 3.0,
}


#: Connelly's average point value of a turnover, the scale ``adv_turnover``'s
#: ``turnover_luck`` and GOP's glossary already use for turnover luck.
_POINTS_PER_TURNOVER = 5.0


@functools.cache
def _field_position_ep() -> pl.DataFrame:
    """Expected points of a drive start by own yardline (99 rows), bundled with sdv-py.

    The yardline-only table GOP's Paper Index reads, on purpose: the pbp's
    ``EP_start`` is sdv-py's EP model, whose inputs include score differential and
    clock, so a drive start priced by it carries the scoreboard. Measured on
    2022-2025, starting EP from ``EP_start`` correlated 0.95 with points margin
    against 0.58 from this table (ClaudeCowork notes/2026-09-29-gop-factors-epa-exploration).
    """
    path = (
        Path(sportsdataverse.__file__).parent
        / "cfb"
        / "models"
        / "cfb_field_position_ep.parquet"
    )
    return pl.read_parquet(path).select(
        pl.col("yardline_own").cast(pl.Int64), pl.col("ep").cast(pl.Float64)
    )


# NOTE: there is deliberately no _RIDGE_LAMBDA here. The penalty is owned by
# `sportsdataverse.cfb.cfb_adjusted_epa` (0.035 as of the 2026-08 audit) and
# this module calls it without an override. A local `_RIDGE_LAMBDA = 325.0`
# lived here until 2026-08-03 -- unused, but 325 is the glmnet-scale value that
# turned out to be a NO-OP under sklearn's `alpha = lambda * n` convention, so
# leaving it lying next to a ridge call was an invitation to wire it back in.
# If a local override is ever genuinely needed, pass it explicitly at the call
# site and say why.
# GEI normalization constant (R build line 664).
_GEI_NORM = 179.01777401608126

#: leaderboard qualifier gates, per team game (Pro-Football-Reference's three
#: minimums; nfl-data's grid uses the same). ONE definition feeds the rank /
#: percentile attach and the league baselines, so a baseline is never taken over a
#: different population than the ``_pct`` beside it. (gate, minimum per team game)
PLAYER_QUALIFIERS: dict[str, tuple[pl.Expr, float]] = {
    "passing": (pl.col("dropbacks") >= 14.0 * pl.col("team_games"), 14.0),
    "rushing": (pl.col("plays") >= 6.25 * pl.col("team_games"), 6.25),
    "receiving": (pl.col("plays") >= 1.875 * pl.col("team_games"), 1.875),
}

#: SDV conference group ids (``cfb_groups`` release) of the power conferences,
#: under the label each era used: P5 through 2023, P4 from 2024, after ten
#: members left the Pac-12. The Pac-10 (2004-2010) carries ``cfb:pac-12``.
_POWER_CONFERENCES: dict[str, list[str]] = {
    "P5": ["cfb:acc", "cfb:big-ten", "cfb:big-12", "cfb:pac-12", "cfb:sec"],
    "P4": ["cfb:acc", "cfb:big-ten", "cfb:big-12", "cfb:sec"],
}
#: The football Big East (``cfb:big-east``, 1991-2012; a separate lineage from
#: ``cfb:american``) was a BCS automatic qualifier, so it is P5 through its last
#: season, 2012. Owner decision on cfbfastR-cfb-data#104.
_BIG_EAST_P5_THROUGH = 2012
#: Notre Dame's ESPN id: an FBS independent classed with the power conferences.
_NOTRE_DAME_ID = "87"

#: league-baseline levels over the published ``fbs_class``: P5/G5 through 2023,
#: P4/G6 from 2024. A null class (a team ``cfb_groups`` puts outside FBS that
#: season) is in ``fbs`` only. No ``fcs`` level: this family is built from
#: FBS-vs-FBS games.
LEVELS: dict[str, pl.Expr | None] = {
    "fbs": None,
    "p4": pl.col("fbs_class").is_in(["P4", "P5"]),
    "g5": pl.col("fbs_class").is_in(["G5", "G6"]),
}

#: the per-(game, team) metrics the percentile ladder is cut on, in R ``reframe`` order
PERCENTILE_METRICS: list[str] = [
    "GEI",
    "EPAplay",
    "pass_success",
    "rush_success",
    "early_down_success",
    "early_down_EPA",
    "late_down_success",
    "success",
    "yardsplay",
    "dropbacks",
    "rushes",
    "EPAdropback",
    "EPArush",
    "yardsdropback",
    "pass_explosive",
    "rush_explosive",
    "explosive",
    "third_down_success",
    "red_zone_success",
    "play_stuffed",
    "nonExplosiveEpaPerPlay",
    "havoc",
    "yardsrush",
    "lineyards",
    "opportunity_run",
    "third_down_distance",
]


def _rank(col: str, *, descending: bool) -> pl.Expr:
    """R ``rank()`` with average ties (``rank(-x)`` -> descending=True), but NULL
    wherever a rank would say nothing.

    Deliberately diverges from the R oracle's ``na.last = TRUE`` (C4): R gave a
    null metric a TRAILING rank, so an unknown read as the worst value (Rice
    2026 ``red_zone_success_off_pass``: null, ranked 138th). A null metric now
    gets a null rank, and a column with fewer than two distinct values ranks
    nobody -- all-null (``line_yards`` on pass plays) ranked teams 1..n in
    team-id order, and constant (``passrate`` on pass plays is 1) tied every
    team at the middle rank.
    """
    c = pl.col(col)
    return pl.when(c.drop_nulls().n_unique() > 1).then(
        c.rank(method="average", descending=descending).cast(pl.Float64)
    )


def _pct(col: str) -> pl.Expr:
    """Percentile of ``{col}_rank`` AMONG QUALIFIERS THAT HAVE THE METRIC, 0-100.

    ``_rank`` already encodes direction (an ascending column ranks low-is-good),
    so the percentile needs no direction of its own.

    Two things the rank column alone does not give us:

    * a null metric (null rank) yields a null percentile, and null rows are
      excluded from ``n`` rather than depressing everyone else's placing.
    * The position is Weibull, ``(n + 1 - rank) / (n + 1)``, which is symmetric:
      the best qualifier lands at ``n/(n+1)`` and the worst at ``1/(n+1)``.
      Nobody is pinned to an exact 0 or 100, which reads better in a UI and
      leaves room for a future qualifier at either end.
    """
    src = pl.col(col)
    rank = pl.col(f"{col}_rank")
    n = src.is_not_null().sum()
    return (
        pl.when(src.is_null())
        .then(None)
        .otherwise(100 * (n + 1 - rank) / (n + 1))
        .cast(pl.Float64)
    )


def add_derived_metrics(plays: pl.DataFrame) -> pl.DataFrame:
    """Port of the build's possession/EPA/explosive derived-column mutate (lines 554-643)."""
    home = pl.col("pos_team") == pl.col("home")
    away = pl.col("pos_team") == pl.col("away")
    df = plays.with_columns(
        pos_team_id=pl.when(home)
        .then(pl.col("home_id"))
        .when(away)
        .then(pl.col("away_id"))
        .otherwise(None)
        .cast(pl.Utf8),
        def_pos_team_id=pl.when(home)
        .then(pl.col("away_id"))
        .when(away)
        .then(pl.col("home_id"))
        .otherwise(None)
        .cast(pl.Utf8),
        pos_EPA_pass=pl.when(home)
        .then(pl.col("home_EPA_pass"))
        .when(away)
        .then(pl.col("away_EPA_pass"))
        .otherwise(None),
        pos_EPA_rush=pl.when(home)
        .then(pl.col("home_EPA_rush"))
        .when(away)
        .then(pl.col("away_EPA_rush"))
        .otherwise(None),
        game_id=pl.col("game_id").cast(pl.Utf8),
        # The next two deliberately DIVERGE from the R oracle (as pass_success /
        # rush_success do in per_game_metrics), which is wrong on both:
        #
        # * an interception gains the offense nothing. statYardage carries the
        #   RETURN on an INT play (1,238 INTs in 2025, mean +12.4), and R counted
        #   it as offense; the sdv-py box zeroes it (cfb_pbp.py `yards_per_play`).
        # * a stuff is a RUN for no gain, over rushes only: sdv-py's `stuffed_run`
        #   (play type "Rush", yds_rushed <= 0), the box's `rushing_stuff_rate`
        #   that GOP ranks against this ladder. R's `yards_gained <= 0` over every
        #   play counted incompletions and sacks (2025 ladder p50 0.300 vs box
        #   0.161). Null on non-rush plays, so every mean is over rushes.
        yards_gained=pl.when(pl.col("int") == 1)
        .then(0)
        .otherwise(pl.col("yards_gained")),
        play_stuffed=pl.when(pl.col("rush") == 1).then(
            ((pl.col("play_type") == "Rush") & (pl.col("yds_rushed") <= 0)).fill_null(
                False
            )
        ),
        red_zone=pl.col("yards_to_goal") <= 20,
        epa_success=pl.col("epa_success").cast(pl.Float64),
    )
    df = df.with_columns(
        red_zone_success=pl.when(pl.col("red_zone"))
        .then(pl.col("epa_success"))
        .otherwise(None),
        third_down_success=pl.when(pl.col("down") == 3)
        .then(pl.col("epa_success"))
        .otherwise(None),
        late_down_success=pl.when(pl.col("down") >= 3)
        .then(pl.col("epa_success"))
        .otherwise(None),
        third_down_distance=pl.when(pl.col("down") == 3)
        .then(pl.col("distance"))
        .otherwise(None),
        early_down_EPA=pl.when(pl.col("down") <= 2).then(pl.col("EPA")).otherwise(None),
        early_down_success=pl.when(pl.col("down") <= 2)
        .then(pl.col("epa_success"))
        .otherwise(None),
        havoc=(
            (pl.col("sack_vec") == True)  # noqa: E712
            | (pl.col("int") == True)  # noqa: E712
            | (pl.col("fumble_vec") == True)  # noqa: E712
            | pl.col("pass_breakup_player_name").is_not_null()
            | (pl.col("yards_gained") < 0)
        ),
        explosive=pl.when(pl.col("pass") == 1)
        .then(pl.col("EPA") >= _EXPLOSIVE_PASS_EPA)
        .when(pl.col("rush") == 1)
        .then(pl.col("EPA") >= _EXPLOSIVE_RUSH_EPA)
        .otherwise(False),
        # over rushes, null elsewhere (R's False on passes made "Opportunity %"
        # rushrate x per-carry rate: 2025 Army #1 at 0.850 x 0.480)
        opportunity_run=pl.when(pl.col("rush") == 1).then(pl.col("yds_rushed") >= 4),
    )
    df = df.with_columns(
        adj_rush_yardage=pl.when((pl.col("rush") == 1) & (pl.col("yds_rushed") > 10))
        .then(pl.lit(10.0))
        .when((pl.col("rush") == 1) & (pl.col("yds_rushed") <= 10))
        .then(pl.col("yds_rushed"))
        .otherwise(None),
    )
    df = df.with_columns(
        line_yards=pl.when((pl.col("rush") == 1) & (pl.col("yds_rushed") < 0))
        .then(1.2 * pl.col("adj_rush_yardage"))
        .when(
            (pl.col("rush") == 1)
            & (pl.col("yds_rushed") >= 0)
            & (pl.col("yds_rushed") <= 4)
        )
        .then(pl.col("adj_rush_yardage"))
        .when(
            (pl.col("rush") == 1)
            & (pl.col("yds_rushed") >= 5)
            & (pl.col("yds_rushed") <= 10)
        )
        .then(0.5 * pl.col("adj_rush_yardage"))
        .when((pl.col("rush") == 1) & (pl.col("yds_rushed") >= 11))
        .then(pl.lit(0.0))
        .otherwise(None),
        second_level_yards=pl.when((pl.col("rush") == 1) & (pl.col("yds_rushed") >= 5))
        .then(0.5 * (pl.col("adj_rush_yardage") - 5))
        .when(pl.col("rush") == 1)
        .then(pl.lit(0.0))
        .otherwise(None),
        open_field_yards=pl.when((pl.col("rush") == 1) & (pl.col("yds_rushed") > 10))
        .then(pl.col("yds_rushed") - pl.col("adj_rush_yardage"))
        .when(pl.col("rush") == 1)
        .then(pl.lit(0.0))
        .otherwise(None),
    )
    df = df.with_columns(
        highlight_yards=pl.col("second_level_yards") + pl.col("open_field_yards")
    )
    df = df.with_columns(
        opp_highlight_yards=pl.when(pl.col("opportunity_run") == True)  # noqa: E712
        .then(pl.col("highlight_yards"))
        .when((pl.col("opportunity_run") == False) & (pl.col("rush") == 1))  # noqa: E712
        .then(pl.lit(0.0))
        .otherwise(None),
        nonExplosiveEpa=pl.when(
            pl.col("EPA").is_not_null() & (pl.col("explosive") == False)  # noqa: E712
        )
        .then(pl.col("EPA"))
        .otherwise(None),
    )
    return df.sort(["game_id", "game_play_number"])


#: every MEAN-aggregated team metric -> the play-level column it averages. Its
#: ``_n`` is that column's non-null count: the mean's actual denominator (a red-zone
#: success rate is over red-zone plays, not all plays).
_TEAM_MEAN_SOURCES: dict[str, str] = {
    "passrate": "pass",
    "rushrate": "rush",
    "havoc": "havoc",
    "explosive": "explosive",
    "EPAplay": "EPA",
    "yardsplay": "yards_gained",
    "play_stuffed": "play_stuffed",
    "success": "epa_success",
    "red_zone_success": "red_zone_success",
    "third_down_success": "third_down_success",
    "third_down_distance": "third_down_distance",
    "late_down_success": "late_down_success",
    "early_down_EPA": "early_down_EPA",
    "nonExplosiveEpaPerPlay": "nonExplosiveEpa",
    "line_yards": "line_yards",
    "opportunity_rate": "opportunity_run",
}
#: ratio metrics -> the count they divide by (read before n_games / n_drives are dropped)
_TEAM_RATIO_DENOMINATORS: dict[str, str] = {
    "playsgame": "n_games",
    "EPAgame": "n_games",
    "yardsgame": "n_games",
    "drivesgame": "n_games",
    "EPAdrive": "n_drives",
    "yardsdrive": "n_drives",
    "playsdrive": "n_drives",
    "turnovers": "n_games",
}


def _summarize_team(
    df: pl.DataFrame, group: str, *, ascending: bool, remove_cols: tuple[str, ...] = ()
) -> pl.DataFrame:
    """Port of ``summarize_team_df`` (group already chosen via ``group``)."""
    g = df.group_by(group).agg(
        *[
            pl.col(src).is_not_null().sum().cast(pl.Int64).alias(f"{m}_n")
            for m, src in _TEAM_MEAN_SOURCES.items()
        ],
        plays=pl.len(),
        n_games=pl.col("game_id").n_unique(),
        n_drives=pl.col("drive_id").n_unique(),
        passrate=pl.col("pass").mean(),
        rushrate=pl.col("rush").mean(),
        havoc=pl.col("havoc").mean(),
        explosive=pl.col("explosive").mean(),
        TEPA=pl.col("EPA").sum(),
        EPAplay=pl.col("EPA").mean(),
        yards=pl.col("yards_gained").sum(),
        yardsplay=pl.col("yards_gained").mean(),
        play_stuffed=pl.col("play_stuffed").mean(),
        success=pl.col("epa_success").mean(),
        red_zone_success=pl.col("red_zone_success").mean(),
        third_down_success=pl.col("third_down_success").mean(),
        third_down_distance=pl.col("third_down_distance").mean(),
        late_down_success=pl.col("late_down_success").mean(),
        early_down_EPA=pl.col("early_down_EPA").mean(),
        nonExplosiveEpaPerPlay=pl.col("nonExplosiveEpa").mean(),
        line_yards=pl.col("line_yards").mean(),
        opportunity_rate=pl.col("opportunity_run").mean(),
    )
    # Turnovers come per game, counted over every play including special teams
    # (summaries_input.game_giveaways), so take one value per (team, game): the
    # offense's giveaways, or -- grouped by the defense -- its opponent's, which
    # are its takeaways.
    tov = (
        df.unique(subset=[group, "game_id"])
        .group_by(group)
        .agg(n_turnovers=pl.col("pos_team_game_giveaways").sum())
    )
    g = g.join(tov, on=group, how="left")
    g = g.with_columns(
        turnovers=pl.col("n_turnovers") / pl.col("n_games"),
        playsgame=pl.col("plays") / pl.col("n_games"),
        EPAdrive=pl.col("TEPA") / pl.col("n_drives"),
        EPAgame=pl.col("TEPA") / pl.col("n_games"),
        yardsgame=pl.col("yards") / pl.col("n_games"),
        drives=pl.col("n_drives"),
        drivesgame=pl.col("n_drives") / pl.col("n_games"),
        yardsdrive=pl.col("yards") / pl.col("n_drives"),
        playsdrive=pl.col("plays") / pl.col("n_drives"),
        *[
            pl.col(d).cast(pl.Int64).alias(f"{m}_n")
            for m, d in _TEAM_RATIO_DENOMINATORS.items()
        ],
    ).drop("n_games", "n_drives", "n_turnovers")

    g = g.sort(group)  # dplyr group_by+summarize returns key-sorted
    d = ascending  # ascending=True -> rank(x); else rank(-x)
    g = g.with_columns(
        playsgame_rank=_rank("playsgame", descending=not d),
        TEPA_rank=_rank("TEPA", descending=not d),
        EPAgame_rank=_rank("EPAgame", descending=not d),
        EPAplay_rank=_rank("EPAplay", descending=not d),
        EPAdrive_rank=_rank("EPAdrive", descending=not d),
        early_down_EPA_rank=_rank("early_down_EPA", descending=not d),
        success_rank=_rank("success", descending=not d),
        yards_rank=_rank("yards", descending=not d),
        yardsplay_rank=_rank("yardsplay", descending=not d),
        yardsgame_rank=_rank("yardsgame", descending=not d),
        drivesgame_rank=_rank("drivesgame", descending=not d),
        yardsdrive_rank=_rank("yardsdrive", descending=not d),
        playsdrive_rank=_rank("playsdrive", descending=not d),
        # play_stuffed: asc -> rank(-x); off -> rank(x)
        play_stuffed_rank=_rank("play_stuffed", descending=d),
        red_zone_success_rank=_rank("red_zone_success", descending=not d),
        third_down_success_rank=_rank("third_down_success", descending=not d),
        late_down_success_rank=_rank("late_down_success", descending=not d),
        # third_down_distance: asc -> rank(-x); off -> rank(x)
        third_down_distance_rank=_rank("third_down_distance", descending=d),
        # havoc: asc -> rank(-x); off -> rank(x)
        havoc_rank=_rank("havoc", descending=d),
        # turnovers: fewer giveaways (off) / more takeaways (def) rank first
        turnovers_rank=_rank("turnovers", descending=d),
        explosive_rank=_rank("explosive", descending=not d),
        passrate_rank=_rank("passrate", descending=True),
        rushrate_rank=_rank("rushrate", descending=True),
        nonExplosiveEpaPerPlay_rank=_rank("nonExplosiveEpaPerPlay", descending=not d),
        line_yards_rank=_rank("line_yards", descending=not d),
        opportunity_rate_rank=_rank("opportunity_rate", descending=not d),
    )
    if remove_cols:
        g = g.drop([c for c in remove_cols if c in g.columns])
    return g


_MARGIN_BASES = [
    ("TEPA", "TEPA"),
    ("EPAplay", "EPAplay"),
    ("EPAdrive", "EPAdrive"),
    ("EPAgame", "EPAgame"),
    ("success", "success"),
    ("yardsplay", "yardsplay"),
]


def _mutate_summary_margins(df: pl.DataFrame) -> pl.DataFrame:
    """Port of ``mutate_summary_df`` -- off/def margins + their ranks."""
    out = df.with_columns(
        [
            (pl.col(f"{b}_off") - pl.col(f"{b}_def")).alias(f"{m}_margin")
            for b, m in _MARGIN_BASES
        ]
    )
    out = out.with_columns(
        [
            _rank(f"{m}_margin", descending=True).alias(f"{m}_margin_rank")
            for _, m in _MARGIN_BASES
        ]
    )
    # Five Factors margins: whole-team table only (the pass/rush splits drop
    # turnovers, so they skip this)
    if "turnovers_off" in df.columns:
        out = out.with_columns(
            explosive_margin=pl.col("explosive_off") - pl.col("explosive_def"),
            # takeaways minus giveaways, so positive is good
            turnover_margin=pl.col("turnovers_def") - pl.col("turnovers_off"),
            # havoc rate created minus allowed (percentage points of plays): havoc is
            # bad for an offense, so def - off, positive is good
            havoc_margin=pl.col("havoc_def") - pl.col("havoc_off"),
        ).with_columns(
            explosive_margin_rank=_rank("explosive_margin", descending=True),
            turnover_margin_rank=_rank("turnover_margin", descending=True),
            havoc_margin_rank=_rank("havoc_margin", descending=True),
        )
    return out


def _drive_owners(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per ``drive_id``: ``_drive_owner``, the team whose drive it is.

    An ESPN drive id can hold snaps from BOTH teams -- a two-point try after a
    pick-six filed under the passer's drive, say -- so a drive belongs to its
    team, not to every team with a snap in it. ESPN's own drive team
    (``drive.team.abbreviation`` against the home / away abbreviations) decides
    when it names a team that has a snap in the drive; a label contradicted by
    every snap (a shifted drive header) falls back to the team with the most
    snaps, the first snap breaking a tie.
    """
    snaps = plays.filter(pl.col("drive_id").is_not_null()).with_row_index("_i")
    espn = pl.lit(None, dtype=pl.Utf8)
    if {"drive.team.abbreviation", "homeTeamAbbrev", "awayTeamAbbrev"} <= set(
        plays.columns
    ):
        abbr, home, away = (
            pl.col("drive.team.abbreviation"),
            pl.col("homeTeamAbbrev"),
            pl.col("awayTeamAbbrev"),
        )
        espn = (
            pl.when(home == away)
            .then(None)
            .when(abbr == home)
            .then(pl.col("home_id"))
            .when(abbr == away)
            .then(pl.col("away_id"))
        )
    per_team = (
        snaps.with_columns(_espn=(pl.col("pos_team_id") == espn).fill_null(False))
        .group_by("drive_id", "pos_team_id")
        .agg(espn=pl.col("_espn").any(), n=pl.len(), first=pl.col("_i").min())
    )
    return per_team.group_by("drive_id").agg(
        _drive_owner=pl.col("pos_team_id")
        .sort_by(["espn", "n", "first"], descending=[True, True, False])
        .first()
    )


def _drives(plays: pl.DataFrame, group: str, *, ascending: bool) -> pl.DataFrame:
    """One side's drive totals (``group`` = the offense or the defense team id).

    ``plays`` carries ``_drive_owner`` (:func:`_drive_owners`). Every drive total
    counts only the owner's drives, so a stray snap in someone else's drive is
    nobody's opportunity and nobody's drive. (Until 2026-10 the yardage totals
    kept R's per-(team, drive) grouping, which handed a stray snap the OTHER
    team's drive yards against its own start: available-yards shares above 1.)
    A drive gains at most the yards it had available; ESPN's ``drive.yards``
    exceeds that on a few dozen drives a season.
    """
    own = pl.col("pos_team_id") == pl.col("_drive_owner")
    per_drive = (
        plays.filter(pl.col("drive_id").is_not_null())
        .group_by([group, "drive_id"])
        .agg(
            total_available_yards=pl.col("drive_start_yards_to_goal").first(),
            total_gained_yards=pl.col("drive_yards").last(),
            scoring_opp=(
                own & (pl.col("yards_to_goal") <= _SCORING_OPP_YARDS_TO_GOAL)
            ).any(),
            owned=own.any(),
            points=pl.col("drive.result")
            .first()
            .replace_strict(_DRIVE_POINTS, default=0.0, return_dtype=pl.Float64),
        )
        .with_columns(
            total_gained_yards=pl.when(
                pl.col("total_available_yards").is_not_null()
            ).then(pl.min_horizontal("total_gained_yards", "total_available_yards"))
        )
        .with_columns(
            _own_yardline=(100 - pl.col("total_available_yards"))
            .round()
            .cast(pl.Int64)
            .clip(1, 99)
        )
        .join(
            _field_position_ep(),
            left_on="_own_yardline",
            right_on="yardline_own",
            how="left",
        )
    )
    agg = per_drive.group_by(group).agg(
        total_available_yards=pl.col("total_available_yards")
        .filter(pl.col("owned") == True)  # noqa: E712
        .sum(),
        total_gained_yards=pl.col("total_gained_yards")
        .filter(pl.col("owned") == True)  # noqa: E712
        .sum(),
        opp_points=pl.col("points").filter(pl.col("scoring_opp") == True).sum(),  # noqa: E712
        pts_per_opp_n=pl.col("scoring_opp").sum().cast(pl.Int64),
        drive_points=pl.col("points").filter(pl.col("owned") == True).sum(),  # noqa: E712
        pts_per_drive_n=pl.col("owned").sum().cast(pl.Int64),
        # field position is averaged per DRIVE over the owner's drives, in yards and in
        # points from the same drives. (Until 2026-09-29 start_position was a mean over
        # PLAYS, so a long drive counted once per snap.)
        start_position=pl.col("total_available_yards")
        .filter(pl.col("owned") == True)
        .mean(),
        start_position_n=(
            (pl.col("owned") == True) & pl.col("total_available_yards").is_not_null()
        )
        .sum()
        .cast(pl.Int64),
        drive_start_ep=pl.col("ep").filter(pl.col("owned") == True).mean(),
    )
    agg = (
        agg.with_columns(
            # null, not Infinity, for a side with no available yards on record
            available_yards_pct=pl.when(pl.col("total_available_yards") > 0).then(
                pl.col("total_gained_yards") / pl.col("total_available_yards")
            ),
            # null, not 0/0 = NaN, for a side that never reached the 40
            pts_per_opp=pl.when(pl.col("pts_per_opp_n") > 0).then(
                pl.col("opp_points") / pl.col("pts_per_opp_n")
            ),
            # null for a side whose only snaps sit in the other team's drives
            pts_per_drive=pl.when(pl.col("pts_per_drive_n") > 0).then(
                pl.col("drive_points") / pl.col("pts_per_drive_n")
            ),
        )
        .drop("opp_points", "drive_points")
        .sort(group)
    )
    return agg.with_columns(
        available_yards_pct_rank=_rank("available_yards_pct", descending=not ascending),
        # more points per trip is better on offense, fewer allowed on defense
        pts_per_opp_rank=_rank("pts_per_opp", descending=not ascending),
        pts_per_drive_rank=_rank("pts_per_drive", descending=not ascending),
        # a better start is worth more on offense, less allowed on defense: fewer
        # yards to go ranks first on offense, more yards to go allowed on defense
        start_position_rank=_rank("start_position", descending=ascending),
        drive_start_ep_rank=_rank("drive_start_ep", descending=not ascending),
    )


def _summarize_drives(plays: pl.DataFrame) -> pl.DataFrame:
    """Offense and defense :func:`_drives` joined on the team, plus their margins."""
    plays = plays.join(_drive_owners(plays), on="drive_id", how="left")
    off_dr = _suffix_nonkey(
        _drives(plays, "pos_team_id", ascending=False), "pos_team_id", "_off"
    )
    def_dr = _suffix_nonkey(
        _drives(plays, "def_pos_team_id", ascending=True), "def_pos_team_id", "_def"
    )
    return (
        off_dr.join(
            def_dr, left_on="pos_team_id", right_on="def_pos_team_id", how="left"
        )
        .with_columns(
            total_available_yards_margin=pl.col("total_available_yards_off")
            - pl.col("total_available_yards_def"),
            total_gained_yards_margin=pl.col("total_gained_yards_off")
            - pl.col("total_gained_yards_def"),
            available_yards_pct_margin=pl.col("available_yards_pct_off")
            - pl.col("available_yards_pct_def"),
            pts_per_opp_margin=pl.col("pts_per_opp_off") - pl.col("pts_per_opp_def"),
            pts_per_drive_margin=pl.col("pts_per_drive_off")
            - pl.col("pts_per_drive_def"),
            drive_start_ep_margin=pl.col("drive_start_ep_off")
            - pl.col("drive_start_ep_def"),
            # (100 - off) - (100 - def): positive when the team starts closer to goal
            start_position_margin=pl.col("start_position_def")
            - pl.col("start_position_off"),
        )
        .with_columns(
            total_available_yards_margin_rank=_rank(
                "total_available_yards_margin", descending=True
            ),
            total_gained_yards_margin_rank=_rank(
                "total_gained_yards_margin", descending=True
            ),
            available_yards_pct_margin_rank=_rank(
                "available_yards_pct_margin", descending=True
            ),
            pts_per_opp_margin_rank=_rank("pts_per_opp_margin", descending=True),
            pts_per_drive_margin_rank=_rank("pts_per_drive_margin", descending=True),
            drive_start_ep_margin_rank=_rank("drive_start_ep_margin", descending=True),
            start_position_margin_rank=_rank("start_position_margin", descending=True),
        )
    )


def _havoc_and_expected_turnovers(team_off: pl.DataFrame) -> pl.DataFrame:
    """Whole-team havoc EPA per game and Connelly's expected turnover margin per game.

    * ``havoc_EPAgame_off``: EPA per game on the team's own havoc snaps (the cost,
      negative); ``_def``: the same on its opponents' snaps it made havoc on;
      ``_margin`` = off - def, positive when its havoc costs opponents more.
    * ``expected_turnover_margin``: half of every scrimmage fumble recovered by each
      side, and interceptions at the season's national share of passes defensed
      (INT + PBU). ``turnover_luck`` (= 5.0 points x (``turnover_margin`` minus this)) is
      set by the caller. The interception share is measured from the season, not
      ``adv_turnover``'s fixed 0.22: ESPN's text under-records pass breakups against
      official stats (2025: 29.8% of INT + PBU are INTs), so a fixed 22% would read
      every defense's expected interceptions low. Kick-play turnovers are in ``turnover_margin`` but not here, so a muff
      reads as luck -- recoveries of loose kicks are close to a coin flip anyway.

    Counts are cast to Float64 before any subtraction: polars sums booleans as
    UInt32, and takeaways - giveaways wraps to ~4.3e9 whenever it should go negative.
    """
    defended = (pl.col("int") == 1) | pl.col("pass_breakup_player_name").is_not_null()
    on_pass = pl.col("pass") == 1
    int_share = team_off.select(
        (pl.col("int") == 1).sum().cast(pl.Float64)
        / defended.filter(on_pass).sum().cast(pl.Float64)
    ).item()

    def side(group: str, suffix: str) -> pl.DataFrame:
        return (
            team_off.group_by(group)
            .agg(
                games=pl.col("game_id").n_unique().cast(pl.Float64),
                havoc_epa=pl.col("EPA").filter(pl.col("havoc") == True).sum(),
                fumbles=(pl.col("fumble_vec") == 1).sum().cast(pl.Float64),
                defended=defended.filter(on_pass).sum().cast(pl.Float64),
            )
            .rename({group: "pos_team_id"})
            .rename(lambda c: c if c == "pos_team_id" else f"{c}{suffix}")
        )

    off, de = side("pos_team_id", "_off"), side("def_pos_team_id", "_def")
    per_game = lambda col, side: pl.col(f"{col}_{side}") / pl.col(f"games_{side}")
    return (
        off.join(de, on="pos_team_id", how="left")
        .with_columns(
            havoc_EPAgame_off=per_game("havoc_epa", "off"),
            havoc_EPAgame_def=per_game("havoc_epa", "def"),
            # expected giveaways (off) and takeaways (def) per game: half of each
            # scrimmage fumble, INTs at the season's share of passes defensed
            expected_turnovers_off=0.5 * per_game("fumbles", "off")
            + int_share * per_game("defended", "off"),
            expected_turnovers_def=0.5 * per_game("fumbles", "def")
            + int_share * per_game("defended", "def"),
        )
        .with_columns(
            havoc_EPAgame_margin=pl.col("havoc_EPAgame_off")
            - pl.col("havoc_EPAgame_def"),
            expected_turnover_margin=pl.col("expected_turnovers_def")
            - pl.col("expected_turnovers_off"),
        )
        .select(
            "pos_team_id",
            "havoc_EPAgame_off",
            "havoc_EPAgame_def",
            "havoc_EPAgame_margin",
            "expected_turnovers_off",
            "expected_turnovers_def",
            "expected_turnover_margin",
        )
        .sort("pos_team_id")
        .with_columns(
            # less EPA lost to havoc is better on offense; more inflicted on defense
            havoc_EPAgame_off_rank=_rank("havoc_EPAgame_off", descending=True),
            havoc_EPAgame_def_rank=_rank("havoc_EPAgame_def", descending=False),
            havoc_EPAgame_margin_rank=_rank("havoc_EPAgame_margin", descending=True),
            # fewer expected giveaways rank first on offense, more takeaways on defense
            expected_turnovers_off_rank=_rank(
                "expected_turnovers_off", descending=False
            ),
            expected_turnovers_def_rank=_rank(
                "expected_turnovers_def", descending=True
            ),
            expected_turnover_margin_rank=_rank(
                "expected_turnover_margin", descending=True
            ),
        )
    )


def _add_turnover_luck(team_data: pl.DataFrame) -> pl.DataFrame:
    """Turnover luck in points per game, per side and overall, with ranks.

    The part of the turnover counts the team's fumbles and passes defensed did not
    earn, scaled by ``_POINTS_PER_TURNOVER`` (the scale ``adv_turnover``'s
    ``turnover_luck`` and GOP's glossary use). Positive = lucky on both sides: fewer
    giveaways than expected (``_off``), more takeaways than expected (``_def``); the
    two sum to ``turnover_luck`` = 5 x (``turnover_margin`` - expected margin).
    """
    return team_data.with_columns(
        turnover_luck_off=_POINTS_PER_TURNOVER
        * (pl.col("expected_turnovers_off") - pl.col("turnovers_off")),
        turnover_luck_def=_POINTS_PER_TURNOVER
        * (pl.col("turnovers_def") - pl.col("expected_turnovers_def")),
        turnover_luck=_POINTS_PER_TURNOVER
        * (pl.col("turnover_margin") - pl.col("expected_turnover_margin")),
    ).with_columns(
        turnover_luck_off_rank=_rank("turnover_luck_off", descending=True),
        turnover_luck_def_rank=_rank("turnover_luck_def", descending=True),
        turnover_luck_rank=_rank("turnover_luck", descending=True),
    )


def _strength_faced_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """Rank the adjusted-EPA strengths of schedule, 1 = the toughest slate.

    ``off_strength_faced`` is the opposing offenses the defense faced (higher is
    tougher); ``def_strength_faced`` is the EPA/play the opposing defenses allow
    (lower is tougher). Null (fewer than 2 valid games) stays unranked.
    """
    return df.with_columns(
        off_strength_faced_rank=_rank("off_strength_faced", descending=True),
        def_strength_faced_rank=_rank("def_strength_faced", descending=False),
    )


def _join_adjusted_epa(team_data: pl.DataFrame, adjusted: pl.DataFrame) -> pl.DataFrame:
    """Left-join :func:`cfb_adjusted_epa` onto ``team_data``, strength ranks included.

    One step, so the strengths of schedule can never ship without their ranks.
    """
    return _strength_faced_ranks(
        team_data.join(adjusted.drop("pos_team"), on="team_id", how="left")
    )


def _suffix_nonkey(df: pl.DataFrame, key: str, suffix: str) -> pl.DataFrame:
    """Rename every non-key column with ``suffix`` (mirrors dplyr join suffix on both sides)."""
    return df.rename({c: f"{c}{suffix}" for c in df.columns if c != key})


def summarize_passer(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Port of ``summarize_passer_df`` (attempt-based metrics only).

    The count-based columns ``sacked``, ``sack_yds``, ``pass_int``, and the
    five derived columns that depend on them (``detmer``, ``detmergame``,
    ``dropbacks``, ``sack_adj_yards``, ``yardsdropback``) are intentionally
    absent here.  The caller computes those separately from the FULL offensive
    frame (keyed by ``sack_taken_player_id`` / ``interception_thrown_player_id``)
    and joins them after this call.  This is necessary because sack and
    interception plays carry no ``passer_player_id`` and are therefore dropped
    by the passer filter before reaching this function.
    """
    g = df.group_by(by).agg(
        passer_player_name=pl.col("passer_player_name").drop_nulls().first(),
        plays=pl.len(),
        games=pl.col("game_id").n_unique(),
        team_games=pl.col("team_games").last(),
        TEPA=pl.col("EPA").sum(),
        EPAplay=pl.col("EPA").mean(),
        yards=pl.col("yds_receiving").sum(),
        success=pl.col("success").mean(),
        comp=pl.col("completion").sum(),
        att=pl.col("pass_attempt").sum(),
        comppct=pl.col("completion").mean(),
        passing_td=pl.col("pass_td").sum(),
    )
    return g.with_columns(
        playsgame=pl.col("plays") / pl.col("games"),
        EPAgame=pl.col("TEPA") / pl.col("games"),
        yardsplay=pl.col("yards") / pl.col("plays"),
        yardsgame=pl.col("yards") / pl.col("games"),
    )


def summarize_rusher(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Port of ``summarize_rusher_df``."""
    g = df.group_by(by).agg(
        rusher_player_name=pl.col("rusher_player_name").drop_nulls().first(),
        plays=pl.len(),
        games=pl.col("game_id").n_unique(),
        team_games=pl.col("team_games").last(),
        TEPA=pl.col("EPA").sum(),
        EPAplay=pl.col("EPA").mean(),
        yards=pl.col("yds_rushed").sum(),
        success=pl.col("epa_success").mean(),
        rushing_td=pl.col("rush_td").sum(),
        fumbles=pl.col("fumble_vec").sum(),
    )
    return g.with_columns(
        playsgame=pl.col("plays") / pl.col("games"),
        EPAgame=pl.col("TEPA") / pl.col("games"),
        yardsplay=pl.col("yards") / pl.col("plays"),
        yardsgame=pl.col("yards") / pl.col("games"),
    )


def summarize_receiver(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Port of ``summarize_receiver_df``."""
    g = df.group_by(by).agg(
        receiver_player_name=pl.col("receiver_player_name").drop_nulls().first(),
        plays=pl.len(),
        games=pl.col("game_id").n_unique(),
        team_games=pl.col("team_games").last(),
        TEPA=pl.col("EPA").sum(),
        EPAplay=pl.col("EPA").mean(),
        yards=pl.col("yds_receiving").sum(),
        success=pl.col("epa_success").mean(),
        comp=pl.col("reception_player_id").is_not_null().sum(),
        targets=(
            pl.col("target_player_id").is_not_null()
            | pl.col("reception_player_id").is_not_null()
        ).sum(),
        passing_td=pl.col("pass_td").sum(),
        fumbles=pl.col("fumble_vec").sum(),
    )
    return g.with_columns(
        playsgame=pl.col("plays") / pl.col("games"),
        EPAgame=pl.col("TEPA") / pl.col("games"),
        yardsplay=pl.col("yards") / pl.col("plays"),
        yardsgame=pl.col("yards") / pl.col("games"),
        catchpct=pl.col("comp") / pl.col("targets"),
    )


def per_game_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, pos_team): the team-game metrics both the percentile
    ladder and the ``team_game`` league baselines are cut over.
    """
    per_game = df.group_by(["game_id", "pos_team"]).agg(
        GEI=pl.col("GEI").drop_nulls().first(),
        EPAplay=pl.col("EPA").mean(),
        # RATES, not shares. The R oracle computes these as
        # `mean(epa_success * pass)` over EVERY play -- but `epa_success * pass`
        # is 1 only on a successful pass and 0 on every other play, including
        # every rush, so that mean is
        #     (# successful passes) / (# ALL plays)
        # i.e. the share of all plays that were successful passes, not the pass
        # success rate. The tell in the published data: the pass and rush values
        # SUM to the overall `success` (2024 medians 0.2153 + 0.2190 = 0.4343 vs
        # success 0.4426) instead of each sitting near it. A real pass success
        # rate is ~0.44, not ~0.22.
        #
        # This deliberately DIVERGES from the R oracle -- the oracle is wrong.
        # Note the same file already splits pass/rush correctly by filtering for
        # the summary tables (`team_off.filter(pl.col("pass") == 1)`) and divides
        # `EPAdropback` by `dropbacks`, so the multiply form was an inconsistency
        # inside this one function rather than a deliberate definition.
        # A game-team with no pass (or rush) attempts yields null, which the
        # quantile step skips -- correct for a rate, where 0 would be a lie.
        pass_success=pl.col("epa_success").filter(pl.col("pass") == 1).mean(),
        rush_success=pl.col("epa_success").filter(pl.col("rush") == 1).mean(),
        early_down_success=pl.col("early_down_success").mean(),
        early_down_EPA=pl.col("early_down_EPA").mean(),
        late_down_success=pl.col("late_down_success").mean(),
        success=pl.col("epa_success").mean(),
        yardsplay=pl.col("yards_gained").mean(),
        dropbacks=pl.col("pass").sum(),
        rushes=pl.col("rush").sum(),
        sum_pos_EPA_pass=pl.col("pos_EPA_pass").sum(),
        sum_pos_EPA_rush=pl.col("pos_EPA_rush").sum(),
        sum_yds_receiving=pl.col("yds_receiving").sum(),
        sum_yds_sacked=pl.col("yds_sacked").sum(),
        # Same defect, same fix as pass_success/rush_success above: these were
        # `mean(explosive * pass)` over all plays, so they summed to `explosive`
        # (2024 medians 0.0400 + 0.0286 = 0.0686 vs explosive 0.0725) instead of
        # each being an explosive RATE within pass / rush plays.
        pass_explosive=pl.col("explosive").filter(pl.col("pass") == 1).mean(),
        rush_explosive=pl.col("explosive").filter(pl.col("rush") == 1).mean(),
        explosive=pl.col("explosive").mean(),
        third_down_success=pl.col("third_down_success").mean(),
        red_zone_success=pl.col("red_zone_success").mean(),
        play_stuffed=pl.col("play_stuffed").mean(),
        nonExplosiveEpaPerPlay=pl.col("nonExplosiveEpa").mean(),
        havoc=pl.col("havoc").mean(),
        yardsrush=pl.col("yds_rushed").mean(),
        lineyards=pl.col("line_yards").mean(),
        opportunity_run=pl.col("opportunity_run").mean(),
        third_down_distance=pl.col("third_down_distance").mean(),
    )
    per_game = per_game.with_columns(
        EPAdropback=pl.when(pl.col("dropbacks") == 0)
        .then(pl.lit(0.0))
        .otherwise(pl.col("sum_pos_EPA_pass") / pl.col("dropbacks")),
        EPArush=pl.when(pl.col("rushes") == 0)
        .then(pl.lit(0.0))
        .otherwise(pl.col("sum_pos_EPA_rush") / pl.col("rushes")),
        yardsdropback=pl.when(pl.col("dropbacks") == 0)
        .then(pl.lit(0.0))
        .otherwise(
            (pl.col("sum_yds_receiving") + pl.col("sum_yds_sacked"))
            / pl.col("dropbacks")
        ),
    ).drop(
        "sum_pos_EPA_pass", "sum_pos_EPA_rush", "sum_yds_receiving", "sum_yds_sacked"
    )
    return per_game.select("game_id", "pos_team", *PERCENTILE_METRICS)


def _quantiles(per_game: pl.DataFrame) -> pl.DataFrame:
    """The 1..99 ladder over :func:`per_game_metrics` (R quantile type 7 == numpy 'linear')."""
    pctiles = [round(0.01 * i, 2) for i in range(1, 100)]
    rows = {"pctile": pctiles}
    for c in PERCENTILE_METRICS:
        vals = per_game[c].to_numpy().astype(float)
        rows[c] = [float(np.nanquantile(vals, p, method="linear")) for p in pctiles]
    return pl.DataFrame(rows)


#: below this many games with a play, every dispersion column is null: a spread, a
#: floor and a ceiling over one or two games describe those games, not the player
DISPERSION_MIN_GAMES = 3
_DISPERSION_COLS = (
    "EPAplay_sd",
    "EPAplay_p10",
    "EPAplay_p90",
    "boom_rate",
    "bust_rate",
)


def player_dispersion(rows: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Spread of a player's per-game EPA/play, one row per ``keys``.

    ``rows`` is one row per play the player's table credits him with (the table's
    own play filter), carrying ``keys``, ``game_id`` and ``EPA``. A game's EPA/play
    is ``sum(EPA) / plays``, the form of the published ``EPAplay``, taken over every
    game with at least one such play (``dispersion_games``). Over those games:

    * ``EPAplay_sd``: population sd (ddof=0);
    * ``EPAplay_p10`` / ``EPAplay_p90``: floor and ceiling, linear interpolation
      (R type 7, the method :func:`_quantiles` uses);
    * ``boom_rate`` / ``bust_rate``: the share of games MORE than one sd above /
      below the player's own per-game mean. That mean is the unweighted mean of
      the games, the centre the sd is taken around, not the play-weighted
      ``EPAplay``. Strict, so a player whose games are all equal (sd 0) is neither
      rather than both.

    All five are null when ``dispersion_games < DISPERSION_MIN_GAMES``.

    The games are SORTED before they are reduced: the per-game frame leaves its
    group_by in no fixed order and float sums are not associative, so an unsorted
    mean / sd moves the last bit between two identical builds, and the committed
    parquet with it.
    """
    x = pl.col("game_epa").sort()
    mean, sd = x.mean(), x.std(ddof=0)
    out = (
        rows.group_by([*keys, "game_id"])
        .agg(game_epa=pl.col("EPA").sum() / pl.len())
        .group_by(keys)
        .agg(
            dispersion_games=pl.len().cast(pl.Int64),
            EPAplay_sd=sd,
            EPAplay_p10=x.quantile(0.1, interpolation="linear"),
            EPAplay_p90=x.quantile(0.9, interpolation="linear"),
            boom_rate=(x > mean + sd).mean(),
            bust_rate=(x < mean - sd).mean(),
        )
    )
    enough = pl.col("dispersion_games") >= DISPERSION_MIN_GAMES
    return out.with_columns(
        [pl.when(enough).then(pl.col(c)).alias(c) for c in _DISPERSION_COLS]
    )


#: Football Outsiders' cut-points on a carry's own yards (``yds_rushed``, penalty
#: yardage excluded): the line takes 4 or fewer, losses included; the second level
#: 5-10; the open field 11 or more. The same cuts ``line_yards`` /
#: ``second_level_yards`` / ``open_field_yards`` split on in add_derived_metrics.
_LINE_MAX_YARDS = 4
_SECOND_LEVEL_MAX_YARDS = 10


def rusher_tiers(rows: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per-rusher yardage tiers, stuff rate and one-score split, one row per ``keys``.

    The three tier shares and ``stuff_rate`` (carries for 0 or fewer yards) are over
    carries with a ``yds_rushed`` (105 of 51,345 FBS carries lacked one in 2024; a
    missing value is not a 0), so the shares sum to 1. ``EPAplay_one_score`` is EPA
    per carry with the score within ``ONE_SCORE_MARGIN`` at the snap
    (``pos_score_diff_start``, the offense's margin before the play) and
    ``EPAplay_not_one_score`` the rest. Each has its ``_n`` carries; with none the
    rate is null.
    """
    y = pl.col("yds_rushed")
    close = pl.col("pos_score_diff_start").abs() <= ONE_SCORE_MARGIN
    epa = pl.col("EPA")
    return rows.group_by(keys).agg(
        line_yards_share=(y <= _LINE_MAX_YARDS).mean(),
        second_level_share=(
            (y > _LINE_MAX_YARDS) & (y <= _SECOND_LEVEL_MAX_YARDS)
        ).mean(),
        open_field_share=(y > _SECOND_LEVEL_MAX_YARDS).mean(),
        stuff_rate=(y <= 0).mean(),
        EPAplay_one_score=epa.filter(close).mean(),
        EPAplay_one_score_n=close.sum().cast(pl.Int64),
        EPAplay_not_one_score=epa.filter(~close).mean(),
        EPAplay_not_one_score_n=(~close).sum().cast(pl.Int64),
    )


def warn_implausible_epa_games(plays: pl.DataFrame, yr: int) -> pl.DataFrame:
    """Report games whose mean EPA/play is impossible, before they poison the tables.

    2016 week 2 published with ``end.yardsToEndzone`` pinned to 99 on essentially
    every play (12,423 of them across 72 of that week's 75 games). ``EP_end`` was
    therefore evaluated as if the offense were on its own 1-yard line after every
    snap, so mean EPA ran about -2.6/play where a healthy game sits near 0. It
    reached the published percentiles as a first-percentile early-down EPA of
    -2.97 -- roughly six times every neighbouring season -- and went unnoticed
    because the median and the upper tail were untouched.

    A healthy season's per-game mean EPA sits inside +/-0.5 in all 22 published
    seasons (2004-2025 scan: |season mean| <= 0.063). Anything outside that is a
    parse or feed defect, not a team playing badly, so it is surfaced loudly
    rather than averaged into a quantile.

    Returns the offending games (empty frame when clean) so a caller can fail the
    build; this function only warns, because a legitimately weird game should not
    silently block a rebuild.
    """
    if plays.height == 0 or "EPA" not in plays.columns:
        return pl.DataFrame()
    per_game = (
        plays.filter(pl.col("EPA").is_not_null())
        .group_by("game_id")
        .agg(mean_epa=pl.col("EPA").mean(), plays=pl.len())
    )
    bad = per_game.filter(pl.col("mean_epa").abs() > 0.5).sort("mean_epa")
    if bad.height:
        print(
            f"  !! {yr}: {bad.height} of {per_game.height} games have an implausible "
            f"mean EPA/play (|mean| > 0.5) -- suspect corrupt end-state yardlines; "
            f"worst {bad['mean_epa'][0]:+.2f} in game {bad['game_id'][0]}",
            flush=True,
        )
    return bad


def _build_schools(plays: pl.DataFrame, yr: int, groups: pl.DataFrame) -> pl.DataFrame:
    """Distinct team -> (pos_team, division, conference, fbs_class) lookup (R build lines 916-923).

    ``fbs_class`` comes from the team's SDV group ids that season in ``groups``
    (``cfb_team_group_seasons``), not from the schedule's conference NAME. The
    name lists this replaced had "Pac-12" but not "Pac-10", so every 2004-2010
    Pac-10 team published as G5, and they left out the Big East (2004-2012).
    """
    is_home = pl.col("home_team_id").cast(pl.Utf8) == pl.col("pos_team_id")
    schools = (
        plays.with_columns(
            pos_team_division=pl.when(is_home)
            .then(pl.col("home_team_division"))
            .otherwise(pl.col("away_team_division")),
            pos_team_conference=pl.when(is_home)
            .then(pl.col("home_team_conference"))
            .otherwise(pl.col("away_team_conference")),
        )
        .unique(subset=["pos_team_id"], keep="first")
        .select("pos_team_id", "pos_team", "pos_team_division", "pos_team_conference")
        .sort("pos_team_id")
    )
    # team_id is the ESPN id only where team_id_source == "espn"; a CFBD id
    # must never match an ESPN one.
    ids = groups.filter(
        (pl.col("season") == yr) & (pl.col("team_id_source") == "espn")
    ).select("team_id", "subdivision_id", "conference_id")
    assert schools.schema["pos_team_id"] == ids.schema["team_id"], (
        f"team id dtypes differ: {schools.schema['pos_team_id']} vs "
        f"{ids.schema['team_id']}"
    )
    missing = schools.join(ids, left_on="pos_team_id", right_on="team_id", how="anti")
    if missing.height:
        # a silent null here drops the team from the p4/g5 baselines
        raise ValueError(
            f"{yr}: {missing.height} team(s) missing from cfb_groups: "
            f"{missing.select('pos_team_id', 'pos_team').rows()}"
        )
    power, rest = ("P4", "G6") if yr >= 2024 else ("P5", "G5")
    power_ids = _POWER_CONFERENCES[power] + (
        ["cfb:big-east"] if yr <= _BIG_EAST_P5_THROUGH else []
    )
    return (
        schools.join(
            ids, left_on="pos_team_id", right_on="team_id", how="left", validate="1:1"
        )
        .with_columns(
            fbs_class=pl.when(
                (pl.col("pos_team_id") == _NOTRE_DAME_ID)
                | pl.col("conference_id").is_in(power_ids)
            )
            .then(pl.lit(power))
            .when(pl.col("subdivision_id") == "cfb:fbs")
            .then(pl.lit(rest))
        )
        .drop("subdivision_id", "conference_id")
    )


def _clean_rank_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Port of ``clean_columns`` -- relocate the ``_rank`` and ``_n`` suffixes to the column END.

    ``TEPA_rank_off`` -> ``TEPA_off_rank`` (after the join ``_off``/``_pass``
    suffixes land mid-name). No-op for plain ``X_rank`` leaderboard columns.
    """
    rank_renames = {
        c: c.replace("_rank", "", 1) + "_rank" for c in df.columns if "_rank" in c
    }
    # ...and ``EPAplay_n_off_pass`` -> ``EPAplay_off_pass_n`` for the sample sizes
    n_renames = {c: c.replace("_n_", "_", 1) + "_n" for c in df.columns if "_n_" in c}
    assert not (rank_renames.keys() & n_renames.keys()), (
        "a column needs both a _rank and a _n_ mid-name relocation -- "
        "the two rename dicts must stay disjoint"
    )
    renames = rank_renames | n_renames
    renames = {k: v for k, v in renames.items() if k != v}
    return df.rename(renames) if renames else df


def _prepare_for_write(
    df: pl.DataFrame, yr: int, schools: pl.DataFrame
) -> pl.DataFrame:
    """Port of ``prepare_for_write`` -- clean rank cols, join schools (with fbs_class), identity-first."""
    df = _clean_rank_columns(df)
    out = (
        # season is a JOIN KEY across the ecosystem and must be an integer. This
        # emitted Float64 (mirroring R's double numerics), publishing season as
        # 2023.0 in team_summaries AND the passing/rushing/receiving player tables,
        # which all route through here. Every other producer site (io.py,
        # reshape.py, reshapers.py, summaries_input.py) already uses Int64.
        df.with_columns(season=pl.lit(int(yr), dtype=pl.Int64))
        .join(schools, on="pos_team_id", how="left")
        .rename(
            {
                "pos_team_id": "team_id",
                "pos_team_division": "division",
                "pos_team_conference": "conference",
            }
        )
    )
    lead = ["team_id", "pos_team", "division", "conference", "season"]
    rest = [c for c in out.columns if c not in lead]
    return out.select(lead + rest)


def build_team_summaries(
    plays_input: pl.DataFrame,
    yr: int,
    *,
    through_week: int | None = None,
    groups: pl.DataFrame | None = None,
    rosters: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """Build the 6 season tables from a cleaned cfbfastR pbp frame (R build lines 554-958).

    ``through_week`` marks a through-week snapshot (``None`` = full season). It
    filters nothing -- the caller has -- it only exempts the snapshot from the
    no-op gate. ``groups`` is the season's ``cfb_team_group_seasons`` frame
    that ``fbs_class`` is read from (loaded from the release when ``None``).
    ``rosters`` is the season's ``cfb_rosters`` frame the player tables'
    ``position_group`` comes from (``None`` leaves it and ``_pos_pct`` null).
    """
    plays = add_derived_metrics(plays_input)
    warn_implausible_epa_games(plays, yr)
    team_off = plays.filter(
        pl.col("EPA").is_not_null()
        & pl.col("success").is_not_null()
        & pl.col("epa_success").is_not_null()
    )

    # percentiles
    pctls = team_off.with_columns(
        GEI=(pl.col("wpa").abs().sum().over("game_id"))
        * (_GEI_NORM / pl.len().over("game_id"))
    )
    per_game = per_game_metrics(pctls)
    percentiles = _quantiles(per_game)

    # team off/def overall + pass + rush + drives
    off = _suffix_nonkey(
        _summarize_team(team_off, "pos_team_id", ascending=False), "pos_team_id", "_off"
    )
    def_ = _suffix_nonkey(
        _summarize_team(team_off, "def_pos_team_id", ascending=True),
        "def_pos_team_id",
        "_def",
    )
    overall = _mutate_summary_margins(
        off.join(def_, left_on="pos_team_id", right_on="def_pos_team_id", how="left")
    )

    # turnovers are a whole-team figure: no pass/rush split (field position is one
    # too, and comes from _drives per drive, so the splits never build it)
    rc = (
        "turnovers",
        "turnovers_rank",
        "turnovers_n",
    )
    off_pass = _suffix_nonkey(
        _summarize_team(
            team_off.filter(pl.col("pass") == 1),
            "pos_team_id",
            ascending=False,
            remove_cols=rc,
        ),
        "pos_team_id",
        "_off",
    )
    def_pass = _suffix_nonkey(
        _summarize_team(
            team_off.filter(pl.col("pass") == 1),
            "def_pos_team_id",
            ascending=True,
            remove_cols=rc,
        ),
        "def_pos_team_id",
        "_def",
    )
    pass_data = _mutate_summary_margins(
        off_pass.join(
            def_pass, left_on="pos_team_id", right_on="def_pos_team_id", how="left"
        )
    )
    off_rush = _suffix_nonkey(
        _summarize_team(
            team_off.filter(pl.col("rush") == 1),
            "pos_team_id",
            ascending=False,
            remove_cols=rc,
        ),
        "pos_team_id",
        "_off",
    )
    def_rush = _suffix_nonkey(
        _summarize_team(
            team_off.filter(pl.col("rush") == 1),
            "def_pos_team_id",
            ascending=True,
            remove_cols=rc,
        ),
        "def_pos_team_id",
        "_def",
    )
    rush_data = _mutate_summary_margins(
        off_rush.join(
            def_rush, left_on="pos_team_id", right_on="def_pos_team_id", how="left"
        )
    )

    drives_data = _summarize_drives(plays)

    team_data = (
        overall.join(drives_data, on="pos_team_id", how="left", suffix="_drive")
        .join(pass_data, on="pos_team_id", how="left", suffix="_pass")
        .join(rush_data, on="pos_team_id", how="left", suffix="_rush")
        .join(_havoc_and_expected_turnovers(team_off), on="pos_team_id", how="left")
        .pipe(_add_turnover_luck)
    )

    # leaderboards
    qb_base = team_off.with_columns(
        passer_player_id=pl.when(pl.col("completion_player_id").is_not_null())
        .then(pl.col("completion_player_id"))
        .otherwise(pl.col("incompletion_player_id")),
        passer_player_name=pl.when(pl.col("completion_player_id").is_not_null())
        .then(pl.col("completion_player"))
        .otherwise(pl.col("incompletion_player")),
    )
    qb_keys = ["pos_team_id", "passer_player_id"]
    qb_attempts = qb_base.filter(
        (pl.col("pass") == 1) & pl.col("passer_player_id").is_not_null()
    ).pipe(_add_team_games)
    qb_data = summarize_passer(qb_attempts, by=qb_keys)
    # sack/INT plays carry no derived passer_player_id (filtered out of qb_data above)
    # -> aggregate them separately, attributed to the sacked QB / the QB who threw
    # the pick.
    #
    # ESPN populates the sack_taken_player_id / interception_thrown_player_id sidecar
    # for only ~29% of these plays (377/1300 INT, 3130/3185 sack in 2023), so a bare
    # id-keyed aggregation drops most of the INT count. The source-frame
    # ``passer_player_name`` column, however, is populated on ~92% of them (1199/1300
    # INT, 3161/3185 sack) -- ESPN charts the QB's *name* on the play even when it
    # omits the numeric id. We recover the missing attribution with a within-dataset
    # (pos_team_id, passer_player_name) -> passer_player_id map built from the attempts
    # leaderboard (each QB's name+id co-occur on their own completions/incompletions).
    #
    # Only UNAMBIGUOUS (team, name) pairs are kept: any name mapping to >1 id in the
    # season is dropped (anti-join) so a future collision can never mis-resolve -- those
    # plays then fall back to clean-id only. The resolved QB id is
    # coalesce(clean sidecar id, name-mapped id); rows where BOTH are null (the QB name
    # itself is absent, e.g. TEAM plays) are dropped and remain unattributed.
    #
    # ``plays`` already carries the source ``passer_player_name`` column, so it is
    # used directly as the name key (no rename -- that would DuplicateError).
    #
    # R PARITY: the map and both aggregations below read the FULL ``plays`` frame, not
    # the EPA-filtered ``team_off``. R applies the EPA/success filter only to the
    # attempts leaderboard (inline: `filter(pass == 1 & !is.na(EPA) & ...)`) and builds
    # `qb_id_map`/`qb_sacks`/`qb_ints` from `plays`. Sourcing these from `team_off`
    # would drop every sack/INT play whose EPA/success/epa_success is null, and lose the
    # (team, name) -> id mappings contributed by attempts the EPA filter removed --
    # undercounting the very columns this revives.
    qb_id_base = plays.with_columns(
        passer_player_id=pl.when(pl.col("completion_player_id").is_not_null())
        .then(pl.col("completion_player_id"))
        .otherwise(pl.col("incompletion_player_id")),
        passer_player_name=pl.when(pl.col("completion_player_id").is_not_null())
        .then(pl.col("completion_player"))
        .otherwise(pl.col("incompletion_player")),
    )
    id_map = (
        qb_id_base.select(["pos_team_id", "passer_player_name", "passer_player_id"])
        .drop_nulls()
        .unique()
    )
    _ambig = (
        id_map.group_by(["pos_team_id", "passer_player_name"])
        .len()
        .filter(pl.col("len") > 1)
    )
    id_map = id_map.join(
        _ambig.select(["pos_team_id", "passer_player_name"]),
        on=["pos_team_id", "passer_player_name"],
        how="anti",
    )
    # Bound to names before the group_by so `_neg_meta` below can reuse the
    # resolved rows -- a passer seeded only from these needs a name and a real
    # game count, and taking them from the union is exact where a per-side
    # maximum would undercount a QB sacked in one game and picked in another.
    _sack_rows = (
        plays.filter((pl.col("pass") == 1) & (pl.col("sack_vec") == 1))
        .select(
            [
                "pos_team_id",
                "passer_player_name",
                "sack_taken_player_id",
                "yds_sacked",
                "EPA",
                "success",
                "game_id",
            ]
        )
        .join(id_map, on=["pos_team_id", "passer_player_name"], how="left")
        .with_columns(
            qb_id=pl.coalesce(
                pl.col("sack_taken_player_id"), pl.col("passer_player_id")
            )
        )
        .filter(pl.col("qb_id").is_not_null())
    )
    sack_counts = (
        _sack_rows.group_by(["pos_team_id", "qb_id"])
        .agg(
            sacked=pl.len(),
            sack_yds=pl.col("yds_sacked").sum(),
            # EPA of the sacks, so it can be returned to the passer's total.
            # Without this the QB's TEPA omits its largest negative component.
            sack_epa=pl.col("EPA").sum(),
            _sack_success=pl.col("success").fill_null(0).sum(),
        )
        .rename({"qb_id": "passer_player_id"})
    )
    _int_rows = (
        plays.filter((pl.col("pass") == 1) & (pl.col("int") == 1))
        .select(
            [
                "pos_team_id",
                "passer_player_name",
                "interception_thrown_player_id",
                "EPA",
                "game_id",
            ]
        )
        .join(id_map, on=["pos_team_id", "passer_player_name"], how="left")
        .with_columns(
            qb_id=pl.coalesce(
                pl.col("interception_thrown_player_id"), pl.col("passer_player_id")
            )
        )
        .filter(pl.col("qb_id").is_not_null())
    )
    int_counts = (
        _int_rows.group_by(["pos_team_id", "qb_id"])
        .agg(pass_int=pl.len(), int_epa=pl.col("EPA").sum())
        .rename({"qb_id": "passer_player_id"})
    )
    _key = ["pos_team_id", "qb_id"]
    _neg_meta = (
        pl.concat(
            [
                _sack_rows.select([*_key, "game_id", "passer_player_name"]),
                _int_rows.select([*_key, "game_id", "passer_player_name"]),
            ]
        )
        .group_by(_key)
        .agg(
            neg_games=pl.col("game_id").n_unique(),
            neg_name=pl.col("passer_player_name").drop_nulls().first(),
        )
        .rename({"qb_id": "passer_player_id"})
    )
    # SEED FROM THE UNION, NOT FROM ATTEMPTS ALONE (cfbfastR-cfb-data#33).
    #
    # These were LEFT joins onto a `qb_data` built only from
    # completion/incompletion-derived ids, so a passer whose entire season is
    # sacks and/or interceptions had no seed row and vanished from the table --
    # 13 sack-only and 20 int-only keys in 2025, 7 and 6 in 2019, 9 and 0 in
    # 2007. Same family as #30: the negative-only outcomes were the ones lost.
    #
    # `full` + `coalesce` unions the keys. Negative ids are deliberately NOT
    # filtered -- `TEAM` rows already ship in this table (6 in 2025, with real
    # `att` and `TEPA`), so dropping them here would invent an inconsistency
    # rather than remove one.
    _team_games = (
        _add_team_games(plays).group_by("pos_team_id").agg(pl.col("team_games").first())
    )
    qb_data = (
        qb_data.join(
            sack_counts,
            on=["pos_team_id", "passer_player_id"],
            how="full",
            coalesce=True,
        )
        .join(
            int_counts,
            on=["pos_team_id", "passer_player_id"],
            how="full",
            coalesce=True,
        )
        .join(
            _neg_meta,
            on=["pos_team_id", "passer_player_id"],
            how="left",
        )
        .with_columns(
            sacked=pl.col("sacked").fill_null(0),
            sack_yds=pl.col("sack_yds").fill_null(0),
            pass_int=pl.col("pass_int").fill_null(0),
            sack_epa=pl.col("sack_epa").fill_null(0.0),
            int_epa=pl.col("int_epa").fill_null(0.0),
            _sack_success=pl.col("_sack_success").fill_null(0.0),
        )
        # Seed the attempt-side columns for rows the union just added. They have
        # no attempts by construction, so every count is a true 0; `games` and
        # the name come from the sack/INT plays themselves.
        .join(_team_games, on="pos_team_id", how="left", suffix="_tm")
        .with_columns(
            passer_player_name=pl.coalesce("passer_player_name", "neg_name"),
            games=pl.coalesce("games", "neg_games"),
            team_games=pl.coalesce("team_games", "team_games_tm"),
            plays=pl.col("plays").fill_null(0),
            TEPA=pl.col("TEPA").fill_null(0.0),
            yards=pl.col("yards").fill_null(0.0),
            comp=pl.col("comp").fill_null(0.0),
            att=pl.col("att").fill_null(0.0),
            passing_td=pl.col("passing_td").fill_null(0.0),
            # per-game rates over zero attempts are a legitimate 0; `yardsplay`
            # is left null because it divides by `plays`, which is also 0
            playsgame=pl.col("playsgame").fill_null(0.0),
            yardsgame=pl.col("yardsgame").fill_null(0.0),
        )
        .drop("neg_name", "neg_games", "team_games_tm")
        # RETURN SACK + INTERCEPTION EPA TO THE PASSER (cfbfastR-cfb-data#30).
        #
        # `qb_data` above aggregates only plays whose derived passer id survives
        # (completion_player_id ?? incompletion_player_id), which is neither on a
        # sack nor on a pick -- so TEPA carried the passer's completions and
        # incompletions and none of his worst outcomes. The counts were already
        # revived here; the EPA was not.
        #
        # Measured on 2025: sacks -6,277.9 EPA over 3,703 plays and interceptions
        # -5,628.7 over 1,421 were discarded league-wide, and the resulting
        # `EPAplay` ran ~2.8x a play-by-play reconstruction (Dylan Raiola: 0.4801
        # published vs 0.173 over his actual dropbacks).
        #
        # `EPAplay` is now per DROPBACK, matching the numerator. Dividing a
        # dropback-complete TEPA by attempts would swap one mismatch for another.
        .with_columns(
            TEPA=pl.col("TEPA") + pl.col("sack_epa") + pl.col("int_epa"),
        )
        .with_columns(
            detmer=(pl.col("yards") / (400 * pl.col("games")))
            * (
                (pl.col("passing_td") + pl.col("pass_int"))
                / (1 + (pl.col("passing_td") - pl.col("pass_int")).abs())
            ),
            detmergame=(pl.col("yardsgame") / 400)
            * (
                (
                    (pl.col("passing_td") / pl.col("games"))
                    + (pl.col("pass_int") / pl.col("games"))
                )
                / (
                    1
                    + (
                        (pl.col("passing_td") / pl.col("games"))
                        - (pl.col("pass_int") / pl.col("games"))
                    ).abs()
                )
            ),
            # `att` counts the `pass_attempt` flag over the frame `summarize_passer`
            # received, which excludes sacks AND interceptions -- so a dropback
            # count of `att + sacked` omits every pick (Raiola 2025 shipped 246
            # for a QB with 251). Both denominators here add `pass_int` back.
            dropbacks=pl.col("att") + pl.col("sacked") + pl.col("pass_int"),
            # null, not 0/0, for a #33 passer who never attempted a pass --
            # completion percentage is undefined without a denominator, and a
            # NaN here would poison comppct_rank.
            comppct=pl.when((pl.col("att") + pl.col("pass_int")) > 0)
            .then(pl.col("comp") / (pl.col("att") + pl.col("pass_int")))
            .otherwise(None),
            sack_adj_yards=pl.col("yards") - pl.col("sack_yds").abs(),
        )
        .with_columns(
            yardsdropback=pl.col("sack_adj_yards") / pl.col("dropbacks"),
            # re-derived from the corrected TEPA; `summarize_passer` computed
            # these from the attempts-only sum before sacks/INTs were folded in
            EPAplay=pl.col("TEPA") / pl.col("dropbacks"),
            EPAgame=pl.col("TEPA") / pl.col("games"),
            # success over DROPBACKS too, like EPAplay (D4): it was a mean over
            # attempts, so a passer's sacks and picks never counted against it
            # (Jack Layne 2025: .480 published vs .428 over his dropbacks). A sack
            # keeps its own `success`; an interception is never a success -- the
            # rule-based `success` reads statYardage, which on a pick carries the
            # RETURN, so 2 of Layne's 9 picks scored as successes.
            success=(
                pl.col("success").fill_null(0.0) * pl.col("plays")
                + pl.col("_sack_success")
            )
            / pl.col("dropbacks"),
        )
        .drop("_sack_success")
    )
    # per-game dispersion over the same plays TEPA / dropbacks sum: attempts, sacks
    # and interceptions, so a game of sacks or picks is a game here too
    qb_rows = pl.concat(
        [
            qb_attempts.select(*qb_keys, "game_id", "EPA"),
            *[
                r.select("pos_team_id", "qb_id", "game_id", "EPA").rename(
                    {"qb_id": "passer_player_id"}
                )
                for r in (_sack_rows, _int_rows)
            ],
        ]
    )
    qb_data = qb_data.join(
        player_dispersion(qb_rows, qb_keys), on=qb_keys, how="left", validate="1:1"
    )
    qb_data = _attach_leader_ranks(
        qb_data,
        keys=qb_keys,
        min_expr=PLAYER_QUALIFIERS["passing"][0],
        rank_cols=[
            "TEPA",
            "EPAgame",
            "EPAplay",
            "success",
            "comppct",
            "yards",
            "yardsplay",
            "yardsgame",
            "sack_adj_yards",
            "yardsdropback",
            "detmer",
            "detmergame",
            "passing_td",
            "pass_int",
            "sacked",
            "boom_rate",
        ],
        asc_cols=["pass_int", "sacked"],
    )
    qb_data = _attach_sample_sizes(qb_data, PLAYER_SAMPLE_SIZES["passing"])

    rb_keys = ["pos_team_id", "rush_player_id"]
    rb_rows = team_off.filter(
        (pl.col("rush") == 1) & pl.col("rush_player_id").is_not_null()
    ).pipe(_add_team_games)
    rb_data = (
        summarize_rusher(rb_rows, by=rb_keys)
        .join(
            player_dispersion(rb_rows, rb_keys), on=rb_keys, how="left", validate="1:1"
        )
        .join(rusher_tiers(rb_rows, rb_keys), on=rb_keys, how="left", validate="1:1")
    )
    rb_data = _attach_leader_ranks(
        rb_data,
        keys=rb_keys,
        min_expr=PLAYER_QUALIFIERS["rushing"][0],
        rank_cols=[
            "TEPA",
            "EPAgame",
            "EPAplay",
            "success",
            "plays",
            "yards",
            "rushing_td",
            "fumbles",
            "yardsplay",
            "yardsgame",
            "boom_rate",
            "stuff_rate",
        ],
        asc_cols=["fumbles", "stuff_rate"],
    )
    rb_data = _attach_sample_sizes(rb_data, PLAYER_SAMPLE_SIZES["rushing"])

    wr_keys = ["pos_team_id", "receiver_player_id"]
    wr_rows = (
        team_off.with_columns(
            receiver_player_id=pl.when(pl.col("reception_player_id").is_not_null())
            .then(pl.col("reception_player_id"))
            .when(pl.col("target_player_id").is_not_null())
            .then(pl.col("target_player_id"))
            .otherwise(None),
            receiver_player_name=pl.when(pl.col("reception_player_id").is_not_null())
            .then(pl.col("reception_player"))
            .when(pl.col("target_player_id").is_not_null())
            .then(pl.col("target_player"))
            .otherwise(None),
        )
        .filter((pl.col("pass") == 1) & pl.col("receiver_player_id").is_not_null())
        .pipe(_add_team_games)
    )
    wr_data = summarize_receiver(wr_rows, by=wr_keys).join(
        player_dispersion(wr_rows, wr_keys), on=wr_keys, how="left", validate="1:1"
    )
    wr_data = _attach_leader_ranks(
        wr_data,
        keys=wr_keys,
        min_expr=PLAYER_QUALIFIERS["receiving"][0],
        rank_cols=[
            "TEPA",
            "EPAgame",
            "EPAplay",
            "success",
            "comp",
            "targets",
            "catchpct",
            "yards",
            "passing_td",
            "fumbles",
            "yardsplay",
            "yardsgame",
            "boom_rate",
        ],
        asc_cols=["fumbles"],
    )
    wr_data = _attach_sample_sizes(wr_data, PLAYER_SAMPLE_SIZES["receiving"])

    if groups is None:
        groups = load_cfb_team_group_seasons(seasons=yr)
    schools = _build_schools(plays, yr, groups)
    # Opponent-adjusted EPA via the shared sdv-py primitive (was a local copy;
    # sportsdataverse.cfb.cfb_adjusted_epa is the single owner as of sdv-py 0.0.71).
    team_data = _join_adjusted_epa(
        _prepare_for_write(team_data, yr, schools), cfb_adjusted_epa(plays)
    )
    # Assert the adjustment ACTUALLY ADJUSTED. On 2026-08-01 this shipped with
    # adj_off_epa at corr 0.9928 against its own raw EPAplay_off -- the ridge
    # penalty was on the glmnet scale (325) which, under sklearn's
    # alpha = lambda * n, shrinks every team effect to zero. Nothing errored;
    # the columns were present and plausible; it sat live for two days. Check
    # the output, not the config.
    assert_adjustment_is_real(
        team_data, season=yr, through_week=through_week, label=f"team_summaries {yr}"
    )
    team_data = _attach_conference_percentiles(team_data)
    # A passer's TEPA must carry his sacks and interceptions -- see #30, where
    # it did not and every column still looked individually plausible.
    assert_passer_epa_includes_sacks(qb_data, label=f"passing {yr}")
    qb_out = _prepare_for_write(qb_data, yr, schools).rename(
        {"passer_player_id": "player_id"}
    )
    rb_out = _prepare_for_write(rb_data, yr, schools).rename(
        {"rush_player_id": "player_id"}
    )
    wr_out = _prepare_for_write(wr_data, yr, schools).rename(
        {"receiver_player_id": "player_id"}
    )
    qb_out, rb_out, wr_out = (
        _attach_position_cohorts(t, rosters) for t in (qb_out, rb_out, wr_out)
    )

    tables = {
        "percentiles": percentiles,
        "team_summaries": team_data,
        "passing": qb_out,
        "rushing": rb_out,
        "receiving": wr_out,
    }
    tables["league_averages"] = build_league_averages(
        {**tables, "team_game": per_game},
        yr,
        levels=LEVELS,
        qualifiers=PLAYER_QUALIFIERS,
    )
    return tables


_PER_GAME_N = {m: pl.col("games") for m in ("EPAgame", "yardsgame", "playsgame")}
#: player rate -> the count it is a rate over, so ``{rate}_n`` is the real sample.
#: Transcribed from summarize_passer/rusher/receiver and the QB pipeline in
#: build_team_summaries: a rate whose denominator changes there changes here.
PLAYER_SAMPLE_SIZES: dict[str, dict[str, pl.Expr]] = {
    "passing": {
        "EPAplay": pl.col(
            "dropbacks"
        ),  # TEPA / dropbacks (sacks + INTs folded in, #30)
        "yardsdropback": pl.col("dropbacks"),
        "comppct": pl.col("att") + pl.col("pass_int"),
        "success": pl.col("dropbacks"),  # attempts + sacks + INTs (D4)
        "yardsplay": pl.col("plays"),
        "detmer": pl.col("games"),
        "detmergame": pl.col("games"),
        **_PER_GAME_N,
    },
    "rushing": {
        "EPAplay": pl.col("plays"),
        "success": pl.col("plays"),
        "yardsplay": pl.col("plays"),
        **_PER_GAME_N,
    },
    "receiving": {
        "EPAplay": pl.col("plays"),
        "success": pl.col("plays"),
        "yardsplay": pl.col("plays"),
        "catchpct": pl.col("targets"),
        **_PER_GAME_N,
    },
}


def _attach_sample_sizes(
    df: pl.DataFrame, denominators: dict[str, pl.Expr]
) -> pl.DataFrame:
    """Add ``{rate}_n`` beside every rate ``df`` carries: the count it is a rate over (0 when null)."""
    present = {m: e for m, e in denominators.items() if m in df.columns}
    return df.with_columns(
        *[e.fill_null(0).cast(pl.Int64).alias(f"{m}_n") for m, e in present.items()]
    )


def _add_team_games(df: pl.DataFrame) -> pl.DataFrame:
    """team_games = distinct game count per pos_team_id, carried onto each row."""
    return df.with_columns(team_games=pl.col("game_id").n_unique().over("pos_team_id"))


def _attach_leader_ranks(
    data: pl.DataFrame,
    *,
    keys: list[str],
    min_expr: pl.Expr,
    rank_cols: list[str],
    asc_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Compute leaderboard ranks + percentiles among qualifiers, left-joined back.

    ``min_expr`` is the qualifier gate (R rank blocks), so both the rank and the
    percentile are AMONG QUALIFIERS -- a receiver at the 50th percentile here sits
    well above the median receiver. Anything the caller lists in ``asc_cols`` is
    ranked low-is-good (interceptions, sacks taken, fumbles).

    The percentile is computed on the qualifier frame, not on the rank frame,
    because it needs the source metric to tell a null apart from a bad value.
    """
    asc = set(asc_cols or [])

    # An asc_cols name that is not in rank_cols is a silent inversion, not a
    # no-op: direction is applied as `c not in asc` while iterating rank_cols,
    # so a misspelled or stale entry leaves that metric ranked HIGH-is-good.
    # For interceptions, sacks taken or fumbles that renders the worst
    # qualifier at the top of the leaderboard and near the 99th percentile in
    # the UI, with nothing anywhere reporting a problem. Fail instead.
    stray = sorted(asc - set(rank_cols))
    if stray:
        raise ValueError(
            f"asc_cols entries missing from rank_cols: {stray}. "
            "They would be ignored and their metrics ranked high-is-good."
        )

    qual = data.filter(min_expr).with_columns(
        *[_rank(c, descending=(c not in asc)).alias(f"{c}_rank") for c in rank_cols]
    )
    qual = qual.with_columns(*[_pct(c).alias(f"{c}_pct") for c in rank_cols])
    out = qual.select(
        *keys,
        *[f"{c}_rank" for c in rank_cols],
        *[f"{c}_pct" for c in rank_cols],
    )
    return data.join(out, on=keys, how="left")


#: Cohort floors (owner decision D-F5, 2026-09-26): a conference or position group
#: with fewer rows that have the metric gets a null cohort percentile, not a
#: percentile of two.
MIN_COHORT_TEAMS = 5
MIN_COHORT_PLAYERS = 10
#: a ``conference`` label that is not a conference: no ``_conf_pct`` cohort, any season
_INDEPENDENTS = "FBS Independents"

#: roster ``position_abbreviation`` -> ``position_group``. Any other listed position
#: is ``other``; ESPN's ``-`` placeholder is an unknown position and maps to null.
_POSITION_GROUPS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}


def _attach_cohort_percentiles(
    df: pl.DataFrame,
    *,
    cohort: str,
    rank_cols: list[str],
    suffix: str,
    min_cohort: int,
) -> pl.DataFrame:
    """Add ``<m>{suffix}``, the Weibull percentile of each ``<m>_rank`` within ``cohort``.

    The cohort rank ``r`` is the rank of ``<m>_rank`` ascending within the cohort,
    over the rows whose ``<m>`` and ``<m>_rank`` are both non-null (a null
    ``_rank`` is a non-qualifier, a null metric, or a column that ranks nobody
    -- see :func:`_rank`). ``_rank`` already encodes direction, so
    ascending is best-first. Ties take ``_rank``'s own ``average`` method. The
    value is ``100 * (n + 1 - r) / (n + 1)`` with ``n`` those rows in the cohort,
    and null when ``<m>`` or the cohort key is null or ``n < min_cohort``.
    """
    exprs = []
    for rc in rank_cols:
        m = rc.removesuffix("_rank")
        has = (
            pl.col(m).is_not_null()
            & pl.col(rc).is_not_null()
            & pl.col(cohort).is_not_null()
        )
        n = has.sum().over(cohort)
        r = pl.when(has).then(pl.col(rc)).rank(method="average").over(cohort)
        exprs.append(
            pl.when(has & (n >= min_cohort))
            .then(100 * (n + 1 - r) / (n + 1))
            .cast(pl.Float64)
            .alias(f"{m}{suffix}")
        )
    return df.with_columns(exprs)


def _attach_conference_percentiles(team_data: pl.DataFrame) -> pl.DataFrame:
    """``<m>_conf_pct`` beside every team ``<m>_rank``, cohort ``conference``.

    The independents get a null cohort key in a temporary column, so they get
    null ``_conf_pct`` and count toward no cohort's ``n``, while the published
    ``conference`` column is left as it is.
    """
    conf = pl.col("conference")
    return _attach_cohort_percentiles(
        team_data.with_columns(_conf_cohort=pl.when(conf != _INDEPENDENTS).then(conf)),
        cohort="_conf_cohort",
        rank_cols=[c for c in team_data.columns if c.endswith("_rank")],
        suffix="_conf_pct",
        min_cohort=MIN_COHORT_TEAMS,
    ).drop("_conf_cohort")


def _attach_position_cohorts(
    df: pl.DataFrame, rosters: pl.DataFrame | None
) -> pl.DataFrame:
    """Join the roster ``position_group`` onto a player table, then its ``_pos_pct``.

    ``rosters`` is the season's ``cfb_rosters`` (``athlete_id``,
    ``position_abbreviation``); None or empty leaves ``position_group`` null. An
    athlete listed twice under one group (a transfer) counts once; one listed
    under two different groups is ambiguous and stays null. ``athlete_id`` is
    pinned to the leaderboard ``player_id`` dtype by an integer or string parse;
    a float on either side raises rather than joining through ``"123.0"``.
    """
    key = df.schema["player_id"]
    if rosters is None or rosters.is_empty():
        df = df.with_columns(position_group=pl.lit(None, dtype=pl.Utf8))
    else:
        src = rosters.schema["athlete_id"]
        if not all(d.is_integer() or d == pl.Utf8 for d in (src, key)):
            raise TypeError(
                f"roster athlete_id is {src}, leaderboard player_id is {key}: "
                "ids join through an integer or string parse, never a float"
            )
        abbr = pl.col("position_abbreviation")
        pos = (
            rosters.select(
                player_id=pl.col("athlete_id").cast(key),
                position_group=pl.when(abbr.is_not_null() & (abbr != "-")).then(
                    abbr.replace_strict(_POSITION_GROUPS, default="other")
                ),
            )
            .drop_nulls()
            .unique()
            .filter(pl.col("player_id").is_unique())
        )
        assert df.schema["player_id"] == pos.schema["player_id"], (
            f"player_id {df.schema['player_id']} != roster {pos.schema['player_id']}"
        )
        df = df.join(pos, on="player_id", how="left", validate="m:1")
    return _attach_cohort_percentiles(
        df,
        cohort="position_group",
        rank_cols=[c for c in df.columns if c.endswith("_rank")],
        suffix="_pos_pct",
        min_cohort=MIN_COHORT_PLAYERS,
    )
