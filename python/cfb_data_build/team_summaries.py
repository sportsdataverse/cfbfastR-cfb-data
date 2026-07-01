"""Season team + player summaries ("Binion Box Score") -- polars/sklearn port of
``R/espn_cfb_15_team_summaries_creation.R``.

Unlike the other datasets this is a **season-level aggregation off a full
cfbfastR-schema season pbp** (``cfbfastR::load_cfb_pbp`` in R; the analogous
``sportsdataverse.cfb.load_cfb_pbp`` in Python -- currently stale for recent
seasons, which is why this dataset's CI step stays on R for now). It produces 5
released tables: ``percentiles``, ``team_summaries``, ``passing``, ``rushing``,
``receiving``.

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

import numpy as np
import polars as pl
from sportsdataverse.cfb import cfb_adjusted_epa

# Explosive-play EPA thresholds (R build lines 604-608).
_EXPLOSIVE_PASS_EPA = 2.4
_EXPLOSIVE_RUSH_EPA = 1.8
# Ridge penalty: R uses glmnet at the grid's largest lambda (cv$lambda[[1]]).
_RIDGE_LAMBDA = 325.0
# GEI normalization constant (R build line 664).
_GEI_NORM = 179.01777401608126


def _rank(col: str, *, descending: bool) -> pl.Expr:
    """R ``rank()`` -- average ties, ``na.last=TRUE``. ``rank(-x)`` -> descending=True.

    R keeps NA rows and assigns them the *trailing* ranks in row order; polars
    ranks nulls to null, so reproduce na.last explicitly. (Callers sort by the
    group key first, matching dplyr's sorted-group output, so the trailing ranks
    land deterministically.)
    """
    c = pl.col(col)
    base = c.rank(method="average", descending=descending)
    n_nonnull = c.is_not_null().sum()
    null_trail = (n_nonnull + c.is_null().cum_sum()).cast(pl.Float64)
    return pl.when(c.is_null()).then(null_trail).otherwise(base)


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
        play_stuffed=pl.col("yards_gained") <= 0,
        red_zone=pl.col("yards_to_goal") <= 20,
        epa_success=pl.col("epa_success").cast(pl.Float64),
    )
    df = df.with_columns(
        red_zone_success=pl.when(pl.col("red_zone")).then(pl.col("epa_success")).otherwise(None),
        third_down_success=pl.when(pl.col("down") == 3).then(pl.col("epa_success")).otherwise(None),
        late_down_success=pl.when(pl.col("down") >= 3).then(pl.col("epa_success")).otherwise(None),
        third_down_distance=pl.when(pl.col("down") == 3).then(pl.col("distance")).otherwise(None),
        early_down_EPA=pl.when(pl.col("down") <= 2).then(pl.col("EPA")).otherwise(None),
        early_down_success=pl.when(pl.col("down") <= 2).then(pl.col("epa_success")).otherwise(None),
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
        opportunity_run=(pl.col("rush") == 1) & (pl.col("yds_rushed") >= 4),
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
        .when((pl.col("rush") == 1) & (pl.col("yds_rushed") >= 0) & (pl.col("yds_rushed") <= 4))
        .then(pl.col("adj_rush_yardage"))
        .when((pl.col("rush") == 1) & (pl.col("yds_rushed") >= 5) & (pl.col("yds_rushed") <= 10))
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
    df = df.with_columns(highlight_yards=pl.col("second_level_yards") + pl.col("open_field_yards"))
    df = df.with_columns(
        opp_highlight_yards=pl.when(pl.col("opportunity_run") == True)  # noqa: E712
        .then(pl.col("highlight_yards"))
        .when((pl.col("opportunity_run") == False) & (pl.col("rush") == 1))  # noqa: E712
        .then(pl.lit(0.0))
        .otherwise(None),
        nonExplosiveEpa=pl.when(pl.col("EPA").is_not_null() & (pl.col("explosive") == False))  # noqa: E712
        .then(pl.col("EPA"))
        .otherwise(None),
    )
    return df.sort(["game_id", "game_play_number"])


def _summarize_team(
    df: pl.DataFrame, group: str, *, ascending: bool, remove_cols: tuple[str, ...] = ()
) -> pl.DataFrame:
    """Port of ``summarize_team_df`` (group already chosen via ``group``)."""
    g = df.group_by(group).agg(
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
        start_position=pl.col("drive_start_yards_to_goal").mean(),
        nonExplosiveEpaPerPlay=pl.col("nonExplosiveEpa").mean(),
        line_yards=pl.col("line_yards").mean(),
        opportunity_rate=pl.col("opportunity_run").mean(),
    )
    g = g.with_columns(
        playsgame=pl.col("plays") / pl.col("n_games"),
        EPAdrive=pl.col("TEPA") / pl.col("n_drives"),
        EPAgame=pl.col("TEPA") / pl.col("n_games"),
        yardsgame=pl.col("yards") / pl.col("n_games"),
        drives=pl.col("n_drives"),
        drivesgame=pl.col("n_drives") / pl.col("n_games"),
        yardsdrive=pl.col("yards") / pl.col("n_drives"),
        playsdrive=pl.col("plays") / pl.col("n_drives"),
    ).drop("n_games", "n_drives")

    g = g.sort(group)  # dplyr group_by+summarize returns key-sorted; needed for na.last ranks
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
        # third_down_distance / start_position: asc -> rank(-x); off -> rank(x)
        third_down_distance_rank=_rank("third_down_distance", descending=d),
        start_position_rank=_rank("start_position", descending=d),
        # havoc: asc -> rank(-x); off -> rank(x)
        havoc_rank=_rank("havoc", descending=d),
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
    out = df.with_columns([(pl.col(f"{b}_off") - pl.col(f"{b}_def")).alias(f"{m}_margin") for b, m in _MARGIN_BASES])
    out = out.with_columns([_rank(f"{m}_margin", descending=True).alias(f"{m}_margin_rank") for _, m in _MARGIN_BASES])
    if "start_position_off" in df.columns:
        out = out.with_columns(
            start_position_margin=(100 - pl.col("start_position_off")) - (100 - pl.col("start_position_def"))
        )
        out = out.with_columns(start_position_margin_rank=_rank("start_position_margin", descending=True))
    return out


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
        targets=(pl.col("target_player_id").is_not_null() | pl.col("reception_player_id").is_not_null()).sum(),
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


def prepare_percentiles(df: pl.DataFrame) -> pl.DataFrame:
    """Port of ``prepare_percentiles`` -- per-(game,team) metrics then 1..99 quantiles."""
    per_game = df.group_by(["game_id", "pos_team"]).agg(
        GEI=pl.col("GEI").drop_nulls().first(),
        EPAplay=pl.col("EPA").mean(),
        pass_success=(pl.col("epa_success") * pl.col("pass")).mean(),
        rush_success=(pl.col("epa_success") * pl.col("rush")).mean(),
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
        pass_explosive=(pl.col("explosive") * pl.col("pass")).mean(),
        rush_explosive=(pl.col("explosive") * pl.col("rush")).mean(),
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
        .otherwise((pl.col("sum_yds_receiving") + pl.col("sum_yds_sacked")) / pl.col("dropbacks")),
    ).drop("sum_pos_EPA_pass", "sum_pos_EPA_rush", "sum_yds_receiving", "sum_yds_sacked")

    # metric columns in the R `reframe` order (everything except game_id/pos_team)
    metric_cols = [
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
    pctiles = [round(0.01 * i, 2) for i in range(1, 100)]
    rows = {"pctile": pctiles}
    for c in metric_cols:
        vals = per_game[c].to_numpy().astype(float)
        # R quantile type 7 (default) == numpy 'linear'
        rows[c] = [float(np.nanquantile(vals, p, method="linear")) for p in pctiles]
    return pl.DataFrame(rows)


def _build_schools(plays: pl.DataFrame) -> pl.DataFrame:
    """Distinct team -> (pos_team, division, conference) lookup (R build lines 916-923)."""
    is_home = pl.col("home_team_id").cast(pl.Utf8) == pl.col("pos_team_id")
    return (
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


def _clean_rank_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Port of ``clean_columns`` -- relocate the ``_rank`` suffix to the column END.

    ``TEPA_rank_off`` -> ``TEPA_off_rank`` (after the join ``_off``/``_pass``
    suffixes land mid-name). No-op for plain ``X_rank`` leaderboard columns.
    """
    renames = {c: c.replace("_rank", "", 1) + "_rank" for c in df.columns if "_rank" in c}
    renames = {k: v for k, v in renames.items() if k != v}
    return df.rename(renames) if renames else df


def _prepare_for_write(df: pl.DataFrame, yr: int, schools: pl.DataFrame) -> pl.DataFrame:
    """Port of ``prepare_for_write`` -- clean rank cols, join schools, identity-first, fbs_class."""
    df = _clean_rank_columns(df)
    # R computes fbs_class AFTER the select-rename, so reference the renamed cols.
    out = (
        df.with_columns(season=pl.lit(float(yr)))
        .join(schools, on="pos_team_id", how="left")
        .rename(
            {
                "pos_team_id": "team_id",
                "pos_team_division": "division",
                "pos_team_conference": "conference",
            }
        )
    )
    p4 = ["SEC", "Big 12", "ACC", "Big Ten"]
    p5 = ["SEC", "Big 12", "ACC", "Big Ten", "Pac-12"]
    conf = pl.col("conference")
    tid = pl.col("team_id")
    fbs_class = (
        pl.when(
            (pl.col("season") >= 2024) & conf.is_not_null() & (conf.is_in(p4) | (pl.col("pos_team") == "Notre Dame"))
        )
        .then(pl.lit("P4"))
        .when((pl.col("season") >= 2024) & (conf.is_not_null() | tid.is_in(["41", "113"])))
        .then(pl.lit("G6"))
        .when((pl.col("season") <= 2023) & conf.is_not_null() & (conf.is_in(p5) | (pl.col("pos_team") == "Notre Dame")))
        .then(pl.lit("P5"))
        .when((pl.col("season") <= 2023) & (conf.is_not_null() | tid.is_in(["349", "41", "113"])))
        .then(pl.lit("G5"))
        .otherwise(None)
    )
    out = out.with_columns(fbs_class=fbs_class)
    lead = ["team_id", "pos_team", "division", "conference", "season"]
    rest = [c for c in out.columns if c not in lead]
    return out.select(lead + rest)


def build_team_summaries(plays_input: pl.DataFrame, yr: int) -> dict[str, pl.DataFrame]:
    """Build the 5 season tables from a cleaned cfbfastR pbp frame (R build lines 554-958)."""
    plays = add_derived_metrics(plays_input)
    team_off = plays.filter(
        pl.col("EPA").is_not_null() & pl.col("success").is_not_null() & pl.col("epa_success").is_not_null()
    )

    # percentiles
    pctls = team_off.with_columns(
        GEI=(pl.col("wpa").abs().sum().over("game_id")) * (_GEI_NORM / pl.len().over("game_id"))
    )
    percentiles = prepare_percentiles(pctls)

    # team off/def overall + pass + rush + drives
    off = _suffix_nonkey(_summarize_team(team_off, "pos_team_id", ascending=False), "pos_team_id", "_off")
    def_ = _suffix_nonkey(
        _summarize_team(team_off, "def_pos_team_id", ascending=True),
        "def_pos_team_id",
        "_def",
    )
    overall = _mutate_summary_margins(off.join(def_, left_on="pos_team_id", right_on="def_pos_team_id", how="left"))

    rc = ("start_position", "start_position_rank")
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
        off_pass.join(def_pass, left_on="pos_team_id", right_on="def_pos_team_id", how="left")
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
        off_rush.join(def_rush, left_on="pos_team_id", right_on="def_pos_team_id", how="left")
    )

    def _drives(group: str, ascending: bool) -> pl.DataFrame:
        per_drive = (
            plays.filter(pl.col("drive_id").is_not_null())
            .group_by([group, "drive_id"])
            .agg(
                total_available_yards=pl.col("drive_start_yards_to_goal").first(),
                total_gained_yards=pl.col("drive_yards").last(),
            )
        )
        agg = per_drive.group_by(group).agg(
            total_available_yards=pl.col("total_available_yards").sum(),
            total_gained_yards=pl.col("total_gained_yards").sum(),
        )
        agg = agg.with_columns(available_yards_pct=pl.col("total_gained_yards") / pl.col("total_available_yards")).sort(
            group
        )
        return agg.with_columns(available_yards_pct_rank=_rank("available_yards_pct", descending=not ascending))

    off_dr = _suffix_nonkey(_drives("pos_team_id", False), "pos_team_id", "_off")
    def_dr = _suffix_nonkey(_drives("def_pos_team_id", True), "def_pos_team_id", "_def")
    drives_data = (
        off_dr.join(def_dr, left_on="pos_team_id", right_on="def_pos_team_id", how="left")
        .with_columns(
            total_available_yards_margin=pl.col("total_available_yards_off") - pl.col("total_available_yards_def"),
            total_gained_yards_margin=pl.col("total_gained_yards_off") - pl.col("total_gained_yards_def"),
            available_yards_pct_margin=pl.col("available_yards_pct_off") - pl.col("available_yards_pct_def"),
        )
        .with_columns(
            total_available_yards_margin_rank=_rank("total_available_yards_margin", descending=True),
            total_gained_yards_margin_rank=_rank("total_gained_yards_margin", descending=True),
            available_yards_pct_margin_rank=_rank("available_yards_pct_margin", descending=True),
        )
    )

    team_data = (
        overall.join(drives_data, on="pos_team_id", how="left", suffix="_drive")
        .join(pass_data, on="pos_team_id", how="left", suffix="_pass")
        .join(rush_data, on="pos_team_id", how="left", suffix="_rush")
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
    qb_data = summarize_passer(
        qb_base.filter((pl.col("pass") == 1) & pl.col("passer_player_id").is_not_null()).pipe(_add_team_games),
        by=["pos_team_id", "passer_player_id"],
    )
    # sack/INT plays carry no passer_player_id (filtered out above) -> aggregate them
    # separately, keyed by the sacked QB / the QB who threw the pick.
    # Both sack_taken_player_id and interception_thrown_player_id are Float64, matching
    # passer_player_id (derived from completion_player_id / incompletion_player_id, also Float64).
    sack_counts = (
        team_off.filter(
            (pl.col("pass") == 1) & (pl.col("sack_vec") == 1) & pl.col("sack_taken_player_id").is_not_null()
        )
        .group_by(["pos_team_id", "sack_taken_player_id"])
        .agg(sacked=pl.len(), sack_yds=pl.col("yds_sacked").sum())
        .rename({"sack_taken_player_id": "passer_player_id"})
    )
    int_counts = (
        team_off.filter(
            (pl.col("pass") == 1) & (pl.col("int") == 1) & pl.col("interception_thrown_player_id").is_not_null()
        )
        .group_by(["pos_team_id", "interception_thrown_player_id"])
        .agg(pass_int=pl.len())
        .rename({"interception_thrown_player_id": "passer_player_id"})
    )
    qb_data = (
        qb_data.join(sack_counts, on=["pos_team_id", "passer_player_id"], how="left")
        .join(int_counts, on=["pos_team_id", "passer_player_id"], how="left")
        .with_columns(
            sacked=pl.col("sacked").fill_null(0),
            sack_yds=pl.col("sack_yds").fill_null(0),
            pass_int=pl.col("pass_int").fill_null(0),
        )
        .with_columns(
            detmer=(pl.col("yards") / (400 * pl.col("games")))
            * ((pl.col("passing_td") + pl.col("pass_int")) / (1 + (pl.col("passing_td") - pl.col("pass_int")).abs())),
            detmergame=(pl.col("yardsgame") / 400)
            * (
                ((pl.col("passing_td") / pl.col("games")) + (pl.col("pass_int") / pl.col("games")))
                / (1 + ((pl.col("passing_td") / pl.col("games")) - (pl.col("pass_int") / pl.col("games"))).abs())
            ),
            dropbacks=pl.col("att") + pl.col("sacked"),
            sack_adj_yards=pl.col("yards") - pl.col("sack_yds").abs(),
        )
        .with_columns(yardsdropback=pl.col("sack_adj_yards") / pl.col("dropbacks"))
    )
    qb_data = _attach_leader_ranks(
        qb_data,
        keys=["pos_team_id", "passer_player_id"],
        min_expr=pl.col("dropbacks") >= (14 * pl.col("team_games")),
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
        ],
    )

    rb_data = summarize_rusher(
        team_off.filter((pl.col("rush") == 1) & pl.col("rush_player_id").is_not_null()).pipe(_add_team_games),
        by=["pos_team_id", "rush_player_id"],
    )
    rb_data = _attach_leader_ranks(
        rb_data,
        keys=["pos_team_id", "rush_player_id"],
        min_expr=pl.col("plays") >= (6.25 * pl.col("team_games")),
        rank_cols=[
            "TEPA",
            "EPAgame",
            "EPAplay",
            "success",
            "yards",
            "yardsplay",
            "yardsgame",
        ],
    )

    wr_data = summarize_receiver(
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
        .pipe(_add_team_games),
        by=["pos_team_id", "receiver_player_id"],
    )
    wr_data = _attach_leader_ranks(
        wr_data,
        keys=["pos_team_id", "receiver_player_id"],
        min_expr=pl.col("plays") >= (1.875 * pl.col("team_games")),
        rank_cols=[
            "TEPA",
            "EPAgame",
            "EPAplay",
            "success",
            "catchpct",
            "yards",
            "yardsplay",
            "yardsgame",
        ],
    )

    schools = _build_schools(plays)
    # Opponent-adjusted EPA via the shared sdv-py primitive (was a local copy;
    # sportsdataverse.cfb.cfb_adjusted_epa is the single owner as of sdv-py 0.0.71).
    team_data = _prepare_for_write(team_data, yr, schools).join(
        cfb_adjusted_epa(plays).drop("pos_team"), on="team_id", how="left"
    )
    qb_out = _prepare_for_write(qb_data, yr, schools).rename({"passer_player_id": "player_id"})
    rb_out = _prepare_for_write(rb_data, yr, schools).rename({"rush_player_id": "player_id"})
    wr_out = _prepare_for_write(wr_data, yr, schools).rename({"receiver_player_id": "player_id"})

    return {
        "percentiles": percentiles,
        "team_summaries": team_data,
        "passing": qb_out,
        "rushing": rb_out,
        "receiving": wr_out,
    }


def _add_team_games(df: pl.DataFrame) -> pl.DataFrame:
    """team_games = distinct game count per pos_team_id, carried onto each row."""
    return df.with_columns(team_games=pl.col("game_id").n_unique().over("pos_team_id"))


def _attach_leader_ranks(
    data: pl.DataFrame, *, keys: list[str], min_expr: pl.Expr, rank_cols: list[str]
) -> pl.DataFrame:
    """Compute leaderboard ranks among qualifiers, left-joined back (R rank blocks)."""
    qual = data.filter(min_expr)
    ranks = qual.select(*keys, *[_rank(c, descending=True).alias(f"{c}_rank") for c in rank_cols])
    return data.join(ranks, on=keys, how="left")
