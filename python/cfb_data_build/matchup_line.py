"""Unit J of the matchup pipeline: assemble the per-game matchup line.

Port of ``2025_cfb_update.R`` 1550-1812 (season assembly), 2095-2183 (pace
coalesce + week-1 pace carry-forward) and 2424-2460 (bowl exclusion). One row
per FBS-vs-FBS game, bowls dropped (CFP kept), 268 columns in the delivered
order (:data:`LINE_COLUMNS`).

How each column group is filled -- the rules the delivered 2025 rows follow:

* ``home_/away_<feature>`` (the 34 F features per side): the team's AS-OF
  values for that game, else -- a team's opener, week 1 or later -- its FULL
  prior-season values (no current-season plays exist yet);
* ``prev_home_/prev_away_<feature>``: the full prior-season values, every row;
* ``*_sec_per_play_*`` (pace): the as-of-date ``pace_history`` value; week 1
  carries the prior season's full-season pace;
* ELO, opponent-ELO rolls, consensus lines: :mod:`cfb_data_build.matchup_elo`;
* ``home_mov = home_points - away_points``; ``prev_season = season - 1``;
* ``game_type``: ``regular`` / ``playoff`` (CFP by ``notes``) / ``bowl`` -- bowls
  are dropped;
* every input's team names are mapped to the delivered display names FIRST
  (``TEAM_NAME_MAPPING`` in ``matchup_features``), so all joins share one spelling.

Side inputs (talent, returning production, coaches, QB, weather, venue / team
metadata) are separate joins; a caller that has none gets those columns null.
"""

from __future__ import annotations

import polars as pl

from cfb_data_build.matchup_elo import consensus_lines, is_playoff, season_elo
from cfb_data_build.matchup_features import (
    TEAM_FEATURE_COLUMNS,
    _SIDE_FEATURES,
    to_display_names,
)

FEATURE_COLS = tuple(f"{s}_{f}" for s in ("off", "def") for f in _SIDE_FEATURES)  # 34
PACE_COLS = tuple(c for c in TEAM_FEATURE_COLUMNS if "sec_per_play" in c)  # 4
SIDE_INPUT_COLS = (
    "join_name",
    "team_talent_weighted",
    "talent",
    "off_rtprod",
    "def_rtprod",
    "ovr_rtprod",
    "head_coach",
    "hc_tenure",
    "oc_cont",
    "dc_cont",
    "athlete_id",
    "qb_name",
    "returning_qb",
    "qb_starter_years",
    "qb_games",
)
WEATHER_COLS = (
    "temperature",
    "dew_point",
    "humidity",
    "precipitation",
    "snowfall",
    "wind_direction",
    "wind_speed",
    "pressure",
    "weather_condition_code",
    "weather_condition",
)
TEAM_META_COLS = (
    "mascot",
    "abbreviation",
    "alt_name1",
    "alt_name2",
    "alt_name3",
    "classification",
    "color",
    "alt_color",
    "logo",
    "logo_2",
    "twitter",
    "venue_id",
    "venue_name",
    "city",
    "state",
    "zip",
    "country_code",
    "timezone",
    "latitude",
    "longitude",
    "elevation",
    "capacity",
    "year_constructed",
    "grass",
    "dome",
)
META_COLS = (
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "venue_id",
    "neutral_site",
    "conference_game",
    "notes",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_conference",
    "away_conference",
    "home_division",
    "away_division",
    "home_pregame_elo",
    "away_pregame_elo",
    "home_points",
    "away_points",
    "home_mov",
)
LINE_COLUMNS: tuple[str, ...] = (
    *META_COLS,
    *(f"home_{c}" for c in SIDE_INPUT_COLS),
    *(f"away_{c}" for c in SIDE_INPUT_COLS),
    *(f"home_{c}" for c in FEATURE_COLS),
    *(f"away_{c}" for c in FEATURE_COLS),
    *(f"prev_home_{c}" for c in FEATURE_COLS),
    *(f"prev_away_{c}" for c in FEATURE_COLS),
    *(f"home_{c}" for c in PACE_COLS),
    *(f"away_{c}" for c in PACE_COLS),
    "spread_open",
    "spread",
    "over_under",
    "over_under_open",
    *WEATHER_COLS,
    "prev_season",
    *(f"home_{c}" for c in TEAM_META_COLS),
    *(f"away_{c}" for c in TEAM_META_COLS),
    "home_opp_elo_roll_avg",
    "home_opp_elo_roll_median",
    "home_opp_elo_roll_sum",
    "away_opp_elo_roll_avg",
    "away_opp_elo_roll_median",
    "away_opp_elo_roll_sum",
    "game_type",
)


def _prefixed(
    frame: pl.DataFrame, cols: tuple[str, ...], prefix: str, key: str
) -> pl.DataFrame:
    """``team`` -> ``key`` and every feature column prefixed, for a side join."""
    return (
        to_display_names(frame, "team")
        .select(["team", *cols])
        .rename({"team": key, **{c: f"{prefix}{c}" for c in cols}})
    )


_UTF8 = {
    "join_name", "head_coach", "qb_name", "weather_condition", "mascot", "abbreviation",
    "alt_name1", "alt_name2", "alt_name3", "classification", "color", "alt_color", "logo",
    "logo_2", "twitter", "venue_name", "city", "state", "zip", "country_code", "timezone",
}  # fmt: skip
#: `humidity` is NOT here: CFBD reports it whole in some seasons and fractional in
#: others (2020: 450 of 560 weather rows carry values like 85.9, while 2016 and 2024
#: are whole), so an Int64 contract fails `_typed` for those seasons. It is a
#: measurement, so Float64 is the honest type.
_INT = {
    "hc_tenure", "oc_cont", "dc_cont", "athlete_id", "returning_qb", "qb_starter_years",
    "qb_games", "snowfall", "wind_direction", "weather_condition_code",
    "venue_id", "capacity", "year_constructed",
}  # fmt: skip
_BOOL = {"grass", "dome"}


def _column_dtype(col: str) -> pl.DataType:
    """Documented dtype of a side-input / weather / team-meta column (the delivered file's)."""
    base = col.removeprefix("home_").removeprefix("away_")
    if base in _UTF8:
        return pl.Utf8
    if base in _INT:
        return pl.Int64
    if base in _BOOL:
        return pl.Boolean
    return pl.Float64


def game_type() -> pl.Expr:
    """``regular`` / ``playoff`` (CFP by notes) / ``bowl`` over a CFBD games frame."""
    return (
        pl.when(
            pl.col("season_type").is_null() | (pl.col("season_type") != "postseason")
        )
        .then(pl.lit("regular"))
        .when(is_playoff(pl.col("notes")))
        .then(pl.lit("playoff"))
        .otherwise(pl.lit("bowl"))
    )


def build_matchup_line(
    games: pl.DataFrame,
    prev_games: pl.DataFrame,
    team_features: pl.DataFrame,
    prev_features: pl.DataFrame,
    pace_hist: pl.DataFrame,
    prev_pace_full: pl.DataFrame,
    lines: pl.DataFrame,
    *,
    side_inputs: pl.DataFrame | None = None,
    weather: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The 268-column matchup line for one season.

    ``games`` / ``prev_games``: CFBD ``/games`` rows (``cfbd_games_elo_*``
    shape) for the season and the prior one. ``team_features``: the as-of
    team-game frame (``game_id, team, <42 features>``). ``prev_features`` /
    ``prev_pace_full``: prior-season full-season features / pace per ``team``.
    ``pace_hist``: :func:`~cfb_data_build.matchup_features.pace_history` for
    the season. ``lines``: CFBD ``/lines`` rows. Optional ``side_inputs``
    (per ``team``: the 15 :data:`SIDE_INPUT_COLS` and 25 :data:`TEAM_META_COLS`)
    and ``weather`` (per ``game_id``) fill the remaining groups; absent, those
    columns are null.
    """
    filled, rolls = season_elo(games, prev_games)
    # season_elo reduces to the ELO columns; put the filled values back on the
    # full CFBD row (points, ids, venue, notes, ...)
    games = to_display_names(games, "home_team", "away_team")
    team_features = to_display_names(team_features, "team")
    pace_hist = to_display_names(pace_hist, "team")
    rolls = to_display_names(rolls, "team")
    base = games.drop("home_pregame_elo", "away_pregame_elo").join(
        filled.select("game_id", "home_pregame_elo", "away_pregame_elo"),
        on="game_id",
        how="left",
    )
    base = base.with_columns(
        start_date=pl.col("start_date").str.slice(0, 10).str.to_date(),
        home_mov=pl.col("home_points") - pl.col("away_points"),
        prev_season=pl.col("season") - 1,
        game_type=game_type(),
        home_team_id=pl.col("home_id"),
        away_team_id=pl.col("away_id"),
    ).filter(
        (pl.col("home_division") == "fbs")
        & (pl.col("away_division") == "fbs")
        & (pl.col("game_type") != "bowl")
    )
    week1 = (pl.col("week") == 1) & (pl.col("season_type") == "regular")
    out = base
    for side in ("home", "away"):
        key = f"{side}_team"
        # as-of features by (game_id, team); prior-season full by team
        asof = team_features.select(["game_id", "team", *FEATURE_COLS]).rename(
            {"team": key, **{c: f"_asof_{c}" for c in FEATURE_COLS}}
        )
        prev = _prefixed(prev_features, FEATURE_COLS, f"prev_{side}_", key)
        out = out.join(asof, on=["game_id", key], how="left").join(
            prev, on=key, how="left"
        )
        # regular-season week 1 takes the prior season's full-season values
        # outright (the source swaps the whole block there, even where its
        # same-day rule had produced a value); any later opener has no prior
        # plays either and coalesces to the same fill instead of a null the
        # source never produced
        # an opener is a row with NO as-of block at all (no prior plays): the
        # plays-per-game median is non-null the moment a single prior play
        # exists, so it is the block's presence flag; a single null column in a
        # present block (no scoring opportunity yet, say) stays null
        opener = week1 | pl.col("_asof_off_plays_per_game").is_null()
        out = out.with_columns(
            [
                pl.when(opener)
                .then(pl.col(f"prev_{side}_{c}"))
                .otherwise(pl.col(f"_asof_{c}"))
                .alias(f"{side}_{c}")
                for c in FEATURE_COLS
            ]
        ).drop([f"_asof_{c}" for c in FEATURE_COLS])
        # pace: the as-of history, else the prior season's full-season pace. The
        # source coalesces on EVERY row, not just week 1: a team whose first
        # game falls in week 2 has no prior plays either and takes the same fill.
        ph = pace_hist.select(["game_id", "team", *PACE_COLS]).rename(
            {"team": key, **{c: f"_ph_{c}" for c in PACE_COLS}}
        )
        pf = _prefixed(prev_pace_full, PACE_COLS, "_pf_", key)
        out = out.join(ph, on=["game_id", key], how="left").join(pf, on=key, how="left")
        out = out.with_columns(
            [
                pl.coalesce(f"_ph_{c}", f"_pf_{c}").alias(f"{side}_{c}")
                for c in PACE_COLS
            ]
        ).drop([f"_ph_{c}" for c in PACE_COLS] + [f"_pf_{c}" for c in PACE_COLS])
        # opponent-ELO rolls
        r = rolls.select(
            "game_id",
            pl.col("team").alias(key),
            pl.col("opp_elo_roll_avg").alias(f"{side}_opp_elo_roll_avg"),
            pl.col("opp_elo_roll_median").alias(f"{side}_opp_elo_roll_median"),
            pl.col("opp_elo_roll_sum").alias(f"{side}_opp_elo_roll_sum"),
        )
        out = out.join(r, on=["game_id", key], how="left")
        # side inputs + team metadata
        if side_inputs is not None:
            cols = [
                c
                for c in (*SIDE_INPUT_COLS, *TEAM_META_COLS)
                if c in side_inputs.columns
            ]
            out = out.join(
                _prefixed(side_inputs, tuple(cols), f"{side}_", key), on=key, how="left"
            )

    out = out.join(consensus_lines(lines), on="game_id", how="left")
    if weather is not None:
        out = out.join(
            weather.select(
                ["game_id", *[c for c in WEATHER_COLS if c in weather.columns]]
            ),
            on="game_id",
            how="left",
        )

    # not-yet-joined groups carry their documented dtype, never a Null column
    missing = [c for c in LINE_COLUMNS if c not in out.columns]
    out = out.with_columns(
        [pl.lit(None, dtype=_column_dtype(c)).alias(c) for c in missing]
    )
    return out.select(list(LINE_COLUMNS)).sort(["season", "week", "game_id"])
