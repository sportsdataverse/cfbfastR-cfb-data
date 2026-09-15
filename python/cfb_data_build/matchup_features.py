"""Matchup-feature primitives: play tagging, WEPA, glm application, drive frame, pace.

Port of the per-play / per-drive units of the R matchup pipeline (the source's
``2025_cfb_update.R`` lines 795-1004 and 1130-1243). Every function is pure
polars in, polars out; the R semantics that matter for parity are:

* ``ifelse(cond, 1, 0)`` on an NA condition is NA -> a flag is the boolean
  condition **cast** to an integer (null stays null), never ``when/otherwise``
  (which would silently turn NA into 0). ``%in%`` and ``grepl`` on NA are FALSE
  -> those conditions are ``fill_null(False)`` before any negation.
* R's ``&`` / ``|`` on NA follow Kleene logic, as polars does.
* WEPA is ``EPA * prod(1 + flag_i * w_i)`` with an NA flag contributing a
  factor of 1 (``replace_na(flag * w, 0) + 1``).
* ``predict.glm(type = "response")`` on a rank-deficient fit drops the aliased
  (NA) coefficients; a null in any feature yields a null prediction
  (``na.action = na.pass``).
* the drive frame is ``summarise()`` output: first play per drive in data
  order, groups emitted in C-locale key order, nulls last; ``prev_drive_result``
  lags over THAT frame, season-wide.

Two flag names are misleading and kept verbatim because the shipped weights
are keyed on them: ``qb_rush`` is a NON-QB rush and ``early_down_rush`` carries
no rush condition. Renaming them would be a silent re-weighting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

PLAY_KEYS = ("game_id", "id_play", "game_play_number")

#: CFBD spelling -> the delivered display name. Applied to every team column at
#: the pbp boundary (``augment_plays``) and to every other input before a join,
#: so all joins share one spelling -- the source's own name fault line.
TEAM_NAME_MAPPING = {
    "UConn": "Connecticut",
    "UTSA": "UT San Antonio",
    "UL Monroe": "Louisiana Monroe",
    "Southern Miss": "Southern Mississippi",
    "Sam Houston": "Sam Houston State",
    "Massachusetts": "UMass",
    "Appalachian State": "App State",
}
TEAM_COLUMNS = (
    "home",
    "away",
    "pos_team",
    "def_pos_team",
    "offense_play",
    "defense_play",
)


def to_display_names(frame: pl.DataFrame, *cols: str) -> pl.DataFrame:
    """Map CFBD spellings to the delivered names in every named column."""
    return frame.with_columns([pl.col(c).replace(TEAM_NAME_MAPPING) for c in cols])


def canonical_ids(frame: pl.DataFrame) -> pl.DataFrame:
    """One id width for every dataset: ``game_id`` / ``season`` / ``week`` as Int64.

    The pbp release ships them Int32 and CFBD Int64; a join between the two
    silently widens, so the two sibling datasets would publish different
    widths. Fix it once at the boundary.
    """
    return frame.with_columns(
        [
            pl.col(c).cast(pl.Int64)
            for c in ("game_id", "season", "week")
            if c in frame.columns
        ]
    )


#: Rush-expectation glm terms, in formula order (``tools/run_pass.R``).
RUSH_EXPECT_FEATURES = (
    "pos_team_score",
    "def_pos_team_score",
    "score_diff",
    "half",
    "period",
    "TimeSecsRem",
    "down",
    "distance",
    "yards_to_goal",
    "ep_before",
    "wp_before",
    "rz_play",
)

#: Scoring-opportunity glm terms, in formula order (``tools/scoring_opp_rate.R``).
SCORING_OPP_FEATURES = (
    "half_secs_rem",
    "game_secs_rem",
    "start_yards_to_goal",
    "pos_team_timeouts",
    "def_pos_team_timeouts",
)

DRIVE_KEYS = ("season", "start_date", "drive_id", "pos_team", "def_pos_team")
DRIVE_FIRST_PLAY_COLS = (
    "new_drive_pts",
    "drive_start_period",
    "TimeSecsRem",
    "adj_TimeSecsRem",
    "yards_to_goal",
    "drive_end_yards_to_goal",
    "pos_team_timeouts",
    "def_pos_team_timeouts",
    "drive_result",
)


def dedupe_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """R ``distinct(game_id, id_play, game_play_number, .keep_all = TRUE)``.

    The release parquet carries ~10k duplicate play rows per season; the
    pipeline keeps the first occurrence in file order.
    """
    return canonical_ids(
        pbp.unique(subset=list(PLAY_KEYS), keep="first", maintain_order=True)
    )


# --------------------------------------------------------------------------- A
def _has(col: str, text: str) -> pl.Expr:
    """R ``grepl(text, col)``: FALSE on NA."""
    return pl.col(col).str.contains(text, literal=True).fill_null(False)


def _is_in(col: str, values: list[str]) -> pl.Expr:
    """R ``col %in% values``: FALSE on NA."""
    return pl.col(col).is_in(values).fill_null(False)


def _c(name: str) -> pl.Expr:
    return pl.col(name)


def _flag_conditions() -> dict[str, pl.Expr]:
    # Order and names mirror the R mutate() block; the shipped weights are keyed
    # on these names, so the order also defines the weight-vector order.
    epa, down, dist, ytg, wpb = (
        _c("EPA"),
        _c("down"),
        _c("distance"),
        _c("yards_to_goal"),
        _c("wp_before"),
    )
    rush, pas, sack, fum = (
        _c("rush") == 1,
        _c("pass") == 1,
        _c("sack") == 1,
        _c("fumble_vec") == 1,
    )
    pen = _c("penalty_detail")
    not_int = ~_has("play_type", "Interception")
    not_sack = ~_has("play_type", "Sack")
    return {
        "passing": pas,
        "passing_ex_int": pas & not_int,
        "rush": rush,
        "pos_rush": rush & (epa > 0),
        "neg_rush": rush & (epa <= 0),
        "short_yd_rush": rush & (dist < 5),
        "long_yd_rush": rush & (dist >= 5),
        "qb_rush": rush & (_c("position_rush") != "QB"),
        "completed_passing": _is_in(
            "play_type", ["Pass Reception", "Passing Touchdown"]
        ),
        "incompleted_passing": _c("play_type") == "Pass Incompletion",
        "off_holding": pen == "Offensive Holding",
        "defensive_pi": (pen == "Pass Interference") & (epa > 0),
        "false_start": pen == "False Start",
        "roughing": pen == "Roughing the Passer",
        "defensive_holding": pen == "Defensive Holding",
        "offensive_unnecessary_roughness": _is_in(
            "penalty_detail", ["Unnecessary Roughness"]
        )
        & (epa < 0),
        "offensive_unsportsmanlike": _is_in(
            "penalty_detail", ["Unsportsmanlike Conduct"]
        )
        & (epa < 0),
        "defensive_unnecessary_roughness": _is_in(
            "penalty_detail", ["Unnecessary Roughness"]
        )
        & (epa >= 0),
        "defensive_unsportsmanlike": _is_in(
            "penalty_detail", ["Unsportsmanlike Conduct"]
        )
        & (epa >= 0),
        "offensive_pi": (pen == "Pass Interference") & (epa <= 0),
        "off_hold_or_false_start": _is_in(
            "penalty_detail", ["Offensive Holding", "False Start"]
        ),
        "sack": sack,
        "fumble": fum,
        "sack_fumble": sack & fum,
        "non_sack_fumble": (_c("sack") == 0) & fum,
        "non_fumble_sack": sack & (_c("fumble_vec") == 0),
        "int": pas & _has("play_type", "Interception"),
        "return_td": (_has("play_type", "Return") | _has("play_type", "Recovery"))
        & _has("play_type", "Touchdown"),
        "punt": _has("play_type", "Punt"),
        "blocked_punt": _has("play_type", "Blocked Punt"),
        "fg": _has("play_type", "Field Goal"),
        "kickoff": _has("play_type", "Kickoff"),
        "first_down": down == 1,
        "second_down": down == 2,
        "third_down": down == 3,
        "fourth_down": (rush | pas) & (down == 4),
        "third_down_ex_sack_int": (down == 3) & not_int & not_sack,
        "third_down_pos": (down == 3) & (epa > 0),
        "third_down_long_ex_sack_int": (down == 3) & (dist > 5) & not_int & not_sack,
        "first_down_rush": (down == 1) & rush,
        "second_down_rush": (down == 2) & rush,
        "third_down_rush": (down == 3) & rush,
        "fourth_down_rush": (down == 4) & rush,
        "first_down_pass": (down == 1) & pas,
        "second_down_pass": (down == 2) & pas,
        "third_down_pass": (down == 3) & pas,
        "fourth_down_pass": (down == 4) & pas,
        "neutral_second_down_rush": (down == 2)
        & rush
        & (wpb > 0.05)
        & (_c("wp_after") < 0.95),
        "early_down_rush": down <= 2,
        "early_down_sack": (down <= 2) & sack & (_c("fumble_vec") == 0),
        "red_zone": ytg <= 20,
        "goal_to_go": _c("Goal_To_Go") == True,  # noqa: E712 -- null-preserving, unlike a bare column
        "goalline": ytg <= 3,
        "plus_territory": ytg <= 50,
        "low_wp": (wpb <= 0.05) | (wpb >= 0.95),
        "garbage_time": (
            ((_c("score_diff") >= 28) & (_c("period") == 1))
            | ((_c("score_diff") >= 24) & (_c("period") == 2))
            | ((_c("score_diff") >= 21) & (_c("period") == 3))
            | ((_c("score_diff") >= 16) & (_c("period") == 4))
        ),
        "asym_low_wp": (wpb <= 0.2) | (wpb >= 0.95),
        "asym_garbage_time": ((wpb <= 0.2) | (wpb >= 0.95)) & (_c("period") == 4),
        "offense_home": _c("pos_team") == _c("home"),
        "offense_away": _c("pos_team") == _c("away"),
    }


FLAG_NAMES: tuple[str, ...] = tuple(_flag_conditions())


def tag_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Append the 60 ``off_<flag>_weight`` and identical ``def_<flag>_weight`` 0/1 columns."""
    conds = _flag_conditions()
    flags = [
        cond.cast(pl.Int8).alias(f"off_{name}_weight") for name, cond in conds.items()
    ]
    out = pbp.with_columns(flags)
    return out.with_columns(
        [pl.col(f"off_{name}_weight").alias(f"def_{name}_weight") for name in conds]
    )


# --------------------------------------------------------------------------- B
def load_wepa_weights(path: str | Path) -> dict[str, float]:
    """The 120 shipped weights keyed ``off_<flag>_weight`` / ``def_<flag>_weight``."""
    weights = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        f"{side}_{name}_weight" for side in ("off", "def") for name in FLAG_NAMES
    }
    if set(weights) != expected:
        raise ValueError(
            f"wepa weights do not match the flag inventory: missing={sorted(expected - set(weights))}"
            f" extra={sorted(set(weights) - expected)}"
        )
    return {k: float(v) for k, v in weights.items()}


def apply_wepa(tagged: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    """``off_wepa`` / ``def_wepa`` = EPA x prod(1 + flag x weight), NA flag -> factor 1."""

    def product(side: str) -> pl.Expr:
        expr = pl.lit(1.0)
        for name in FLAG_NAMES:
            col = f"{side}_{name}_weight"
            expr = expr * (
                (pl.col(col).cast(pl.Float64) * weights[col]).fill_null(0.0) + 1.0
            )
        return (expr * pl.col("EPA")).alias(f"{side}_wepa")

    return tagged.with_columns(product("off"), product("def"))


# --------------------------------------------------------------------------- C / D apply
@dataclass(frozen=True)
class GlmCoefficients:
    """A fitted binomial-logit glm reduced to what ``predict`` needs."""

    intercept: float
    coefficients: dict[str, float]  # aliased (rank-deficient) terms omitted
    features: list[str]  # every formula term, in order, aliased included
    aliased: list[str]
    family: str
    link: str

    @property
    def active(self) -> dict[str, float]:
        return self.coefficients


def load_coefficients(path: str | Path) -> GlmCoefficients:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    coefs = raw["coefficients"]
    features = [k for k in coefs if k != "(Intercept)"]
    aliased = [k for k in features if coefs[k] is None]
    if raw["family"] != "binomial" or raw["link"] != "logit":
        raise ValueError(
            f"only binomial/logit is supported, got {raw['family']}/{raw['link']}"
        )
    return GlmCoefficients(
        intercept=float(coefs["(Intercept)"]),
        coefficients={k: float(coefs[k]) for k in features if coefs[k] is not None},
        features=features,
        aliased=aliased,
        family=raw["family"],
        link=raw["link"],
    )


def apply_logit(coef: GlmCoefficients) -> pl.Expr:
    """R ``predict.glm(type = "response")`` as an expression over the feature columns.

    Terms are accumulated in formula order (as ``X %*% beta`` does); a null in
    any NON-aliased feature propagates to a null prediction.
    """
    eta = pl.lit(coef.intercept)
    for name in coef.features:
        if name in coef.coefficients:
            eta = eta + pl.col(name).cast(pl.Float64) * coef.coefficients[name]
    return (1.0 / (1.0 + (-eta).exp())).alias("prediction")


# --------------------------------------------------------------------------- D
def build_drive_frame(
    plays: pl.DataFrame, scoring_opp: GlmCoefficients
) -> pl.DataFrame:
    """First-play-of-drive snapshot with ``scoring_opp`` and its expectation.

    ``plays`` is whatever subset the caller aggregates over (the pipeline uses a
    team's plays as-of a date); the lag in ``prev_drive_result`` runs over the
    key-sorted output of THAT subset, exactly as the R ``summarise -> filter ->
    lag`` chain does.
    """
    keys = list(DRIVE_KEYS)
    ind = ((pl.col("down") == 1) & (pl.col("yards_to_goal") < 40)).cast(pl.Int8)
    grouped = (
        plays.with_columns(ind.alias("_ind"))
        .group_by(keys, maintain_order=True)
        .agg(
            [pl.col(c).first() for c in DRIVE_FIRST_PLAY_COLS]
            # R sum() without na.rm: any NA poisons the group
            + [
                pl.when(pl.col("_ind").is_null().any())
                .then(None)
                .otherwise((pl.col("_ind").sum() > 0).cast(pl.Int8))
                .alias("scoring_opp_ind_no_td")
            ]
        )
        .sort(keys, nulls_last=True)
        .rename(
            {
                "yards_to_goal": "start_yards_to_goal",
                "TimeSecsRem": "half_secs_rem",
                "adj_TimeSecsRem": "game_secs_rem",
            }
        )
        .filter(
            (pl.col("start_yards_to_goal") > 40)
            & pl.col("pos_team_timeouts").is_not_null()
            & pl.col("def_pos_team_timeouts").is_not_null()
        )
    )
    return (
        grouped.with_columns(
            prev_drive_result=pl.col("drive_result").shift(1),
            # R case_when(new_drive_pts >= 6 ~ 1, ind == 1 ~ 1, TRUE ~ 0): an NA
            # condition falls through, so a null indicator labels 0 -- unlike the
            # ifelse() flags above this one is NOT null-preserving, by the source
            scoring_opp=pl.when(pl.col("new_drive_pts") >= 6)
            .then(1)
            .when(pl.col("scoring_opp_ind_no_td") == 1)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8),
        )
        .with_columns(
            scoring_opp_prediction=apply_logit(scoring_opp),
        )
        .with_columns(
            scoring_opp_oe=pl.col("scoring_opp").cast(pl.Float64)
            - pl.col("scoring_opp_prediction"),
        )
    )


# --------------------------------------------------------------------------- E
def game_seconds_remaining(plays: pl.DataFrame) -> pl.DataFrame:
    """R ``mk_gsr``: ``adj_TimeSecsRem`` when present, else ``(4 - period) * 900 + clock``."""
    if "adj_TimeSecsRem" in plays.columns:
        return plays.with_columns(gsr=pl.col("adj_TimeSecsRem").cast(pl.Float64))
    sec_in_period = pl.col("clock_minutes").cast(pl.Float64) * 60 + pl.col(
        "clock_seconds"
    ).cast(pl.Float64)
    period = pl.col("period").cast(pl.Int32)
    return plays.with_columns(
        gsr=pl.when(sec_in_period.is_not_null() & period.is_not_null() & (period <= 4))
        .then((4 - period).cast(pl.Float64) * 900 + sec_in_period)
        .otherwise(None)
    )


def play_pace(plays: pl.DataFrame) -> pl.DataFrame:
    """Per rush/pass play: seconds elapsed since the previous play of the SAME drive.

    Plays are ordered by game, drive, then descending game clock (stable on
    ties, nulls last); the first play of every drive carries null and negative
    gaps clamp to 0.
    """
    ordered = (
        game_seconds_remaining(plays)
        .filter((pl.col("rush") == 1) | (pl.col("pass") == 1))
        .sort(
            ["game_id", "drive_id", "gsr"],
            descending=[False, False, True],
            nulls_last=True,
            maintain_order=True,
        )
    )
    prev_gsr = pl.col("gsr").shift(1).over(["game_id", "drive_id"])
    return ordered.with_columns(
        sec_since_prev=pl.when(prev_gsr.is_null() | pl.col("gsr").is_null())
        .then(None)
        .otherwise(pl.max_horizontal(prev_gsr - pl.col("gsr"), pl.lit(0.0)))
    )


def pace_history(pbp: pl.DataFrame) -> pl.DataFrame:
    """As-of-date pace per (season, team, game): the pipeline's ``pace_hist`` table.

    The historical variant of the loop's pace (``tools/backfill_pace.R``): over
    ALL plays of the season (no ``ppa`` filter, every team that appears as
    offense or defense), a team's row for game G aggregates ``sec_since_prev``
    over its games dated strictly before G -- mean from running sums, median
    over the prior plays -- and its first game is null. Offense rows key on
    ``offense_play``, defense rows on ``defense_play``; the two sides are
    outer-joined, so a team with plays on only one side still gets a row.
    """
    paced = play_pace(canonical_ids(pbp)).with_columns(
        play_date=pl.col("start_date").str.slice(0, 10).str.to_date()
    )
    sides = []
    for side, team_col in (("off", "offense_play"), ("def", "defense_play")):
        rows: list[dict] = []
        by_team = paced.select(
            "season",
            pl.col(team_col).alias("team"),
            "game_id",
            "play_date",
            "sec_since_prev",
        )
        for (season, team), grp in by_team.group_by(
            ["season", "team"], maintain_order=True
        ):
            games = (
                grp.group_by(["game_id", "play_date"], maintain_order=True)
                .agg(
                    s=pl.col("sec_since_prev").sum(),
                    n=pl.col("sec_since_prev").is_not_null().sum(),
                )
                .sort("play_date", maintain_order=True)
                .with_columns(
                    cum_s=pl.col("s").cum_sum().shift(1, fill_value=0.0),
                    cum_n=pl.col("n").cum_sum().shift(1, fill_value=0),
                )
            )
            secs = grp.select("play_date", "sec_since_prev").drop_nulls(
                "sec_since_prev"
            )
            for game_id, date, cum_s, cum_n in games.select(
                "game_id", "play_date", "cum_s", "cum_n"
            ).iter_rows():
                prior = secs.filter(pl.col("play_date") < date)["sec_since_prev"]
                rows.append(
                    {
                        "season": season,
                        "team": team,
                        "game_id": game_id,
                        f"{side}_sec_per_play_mean": (cum_s / cum_n)
                        if cum_n > 0
                        else None,
                        f"{side}_sec_per_play_median": prior.median()
                        if prior.len()
                        else None,
                    }
                )
        sides.append(
            pl.DataFrame(
                rows,
                schema={
                    "season": pbp.schema["season"],
                    "team": pl.Utf8,
                    "game_id": pbp.schema["game_id"],
                    f"{side}_sec_per_play_mean": pl.Float64,
                    f"{side}_sec_per_play_median": pl.Float64,
                },
            )
        )
    off, deff = sides
    return off.join(deff, on=["season", "team", "game_id"], how="full", coalesce=True)


# --------------------------------------------------------------------------- F
#: The per-team feature family, in the pipeline's column order (offense block,
#: defense block, then the four pace columns).
_SIDE_FEATURES = (
    "plays_per_game",
    "3d_per_game",
    "epa",
    "pass_epa",
    "rush_epa",
    "rroe",
    "1st_down_rush_rate",
    "scoring_opp_rate_oe",
    "pts_per_scoring_opp",
    "starting_fp",
    "wepa",
    "success_rate",
    "early_success_rate",
    "late_success_rate",
    "pass_success_rate",
    "rush_success_rate",
    "3rd_down_pct",
)
TEAM_FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"off_{f}" for f in _SIDE_FEATURES),
    *(f"def_{f}" for f in _SIDE_FEATURES),
    "off_sec_per_play_mean",
    "off_sec_per_play_median",
    "def_sec_per_play_mean",
    "def_sec_per_play_median",
)


def augment_plays(
    pbp: pl.DataFrame, weights: dict[str, float], rush_expect: GlmCoefficients
) -> pl.DataFrame:
    """Dedupe, tag, weight and score a season's pbp into the frame unit F aggregates.

    Maps every team column to the display spelling, then adds ``off_wepa`` /
    ``def_wepa``, ``rp_prediction`` / ``rroe``, ``gsr`` and ``play_date`` (the UTC calendar day of ``start_date``, which is what the
    pipeline's ``as.Date()`` comparisons see); the 120 flag columns are dropped
    again -- nothing downstream reads them.
    """
    out = apply_wepa(tag_plays(dedupe_plays(pbp)), weights)
    out = to_display_names(out, *[c for c in TEAM_COLUMNS if c in out.columns])
    out = out.drop([c for c in out.columns if c.endswith("_weight")])
    out = out.with_columns(rp_prediction=apply_logit(rush_expect))
    out = out.with_columns(
        rroe=pl.col("rush").cast(pl.Float64) - pl.col("rp_prediction"),
        play_date=pl.col("start_date").str.slice(0, 10).str.to_date(),
    )
    return game_seconds_remaining(out)


def _mean(frame: pl.DataFrame, col: str) -> float | None:
    return frame[col].cast(pl.Float64).mean() if frame.height else None


def _median(frame: pl.DataFrame, col: str) -> float | None:
    return frame[col].cast(pl.Float64).median() if frame.height else None


def _plays_per_game_median(frame: pl.DataFrame) -> float | None:
    if not frame.height:
        return None
    return frame.group_by("game_id").len()["len"].cast(pl.Float64).median()


def _side_features(
    plays: pl.DataFrame, drives: pl.DataFrame, team: str, side: str
) -> dict[str, float | None]:
    """One side's 17 features over a team's plays (unit F, one loop body)."""
    play_col, drive_col = (
        ("offense_play", "pos_team")
        if side == "off"
        else ("defense_play", "def_pos_team")
    )
    mine = plays.filter(pl.col(play_col) == team)
    my_drives = drives.filter(pl.col(drive_col) == team)
    down = pl.col("down")
    return {
        f"{side}_plays_per_game": _plays_per_game_median(mine),
        f"{side}_3d_per_game": _plays_per_game_median(mine.filter(down == 3)),
        f"{side}_epa": _mean(mine, "ppa"),
        f"{side}_pass_epa": _mean(mine.filter(pl.col("pass") == 1), "ppa"),
        f"{side}_rush_epa": _mean(mine.filter(pl.col("rush") == 1), "ppa"),
        f"{side}_rroe": _mean(mine, "rroe"),
        f"{side}_1st_down_rush_rate": _mean(mine.filter(down == 1), "rush"),
        f"{side}_scoring_opp_rate_oe": _mean(my_drives, "scoring_opp_oe"),
        f"{side}_pts_per_scoring_opp": _mean(
            my_drives.filter(pl.col("scoring_opp") == 1), "new_drive_pts"
        ),
        f"{side}_starting_fp": _mean(
            my_drives.filter(pl.col("prev_drive_result") == "PUNT"),
            "start_yards_to_goal",
        ),
        f"{side}_wepa": _mean(mine, f"{side}_wepa"),
        f"{side}_success_rate": _mean(mine, "success"),
        f"{side}_early_success_rate": _mean(mine.filter(down <= 2), "success"),
        f"{side}_late_success_rate": _mean(mine.filter(down == 3), "success"),
        f"{side}_pass_success_rate": _mean(mine.filter(pl.col("pass") == 1), "success"),
        f"{side}_rush_success_rate": _mean(mine.filter(pl.col("rush") == 1), "success"),
        # the share of downs 1-3 that were third downs -- a share, not a
        # conversion rate; the name is the pipeline's
        f"{side}_3rd_down_pct": _mean(
            mine.filter(down <= 3).with_columns(_third=(down == 3).cast(pl.Float64)),
            "_third",
        ),
    }


def _pace(plays: pl.DataFrame, team: str, side: str) -> dict[str, float | None]:
    play_col = "offense_play" if side == "off" else "defense_play"
    paced = play_pace(plays.filter(pl.col(play_col) == team))
    return {
        f"{side}_sec_per_play_mean": _mean(paced, "sec_since_prev"),
        f"{side}_sec_per_play_median": _median(paced, "sec_since_prev"),
    }


def team_game_features(
    plays: pl.DataFrame,
    games: pl.DataFrame,
    teams: list[str],
    scoring_opp: GlmCoefficients,
    *,
    first_game_same_day: bool = False,
) -> pl.DataFrame:
    """Per (team, game) as-of features: one row per game a listed team plays.

    ``plays`` is the ``augment_plays`` frame (plays with a null ``ppa`` are
    dropped, as the pipeline does); ``games`` needs ``game_id``, ``start_date``
    (Date), ``home_team``, ``away_team``. Game i of a team's season sees plays
    dated strictly before it. Its FIRST game sees that same day's
    regular-season plays when ``first_game_same_day`` is True -- the source's
    own behaviour, which reads the game being predicted, kept ONLY for the
    R-parity tests. The default (False) leaves the first game null.
    """
    plays = plays.filter(pl.col("ppa").is_not_null())
    rows: list[dict] = []
    for team in teams:
        team_games = games.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        ).sort("start_date", maintain_order=True)
        team_plays = plays.filter((pl.col("home") == team) | (pl.col("away") == team))
        for i, (game_id, game_date) in enumerate(
            team_games.select("game_id", "start_date").iter_rows()
        ):
            if i == 0:
                subset = (
                    team_plays.filter(
                        (pl.col("play_date") == game_date)
                        & (pl.col("season_type") == "regular")
                    )
                    if first_game_same_day
                    else team_plays.clear()
                )
            else:
                subset = team_plays.filter(pl.col("play_date") < game_date)
            row: dict = {"game_id": game_id, "team": team}
            if subset.height == 0:
                row.update({c: None for c in TEAM_FEATURE_COLUMNS})
            else:
                drives = build_drive_frame(subset, scoring_opp)
                row.update(_side_features(subset, drives, team, "off"))
                row.update(_side_features(subset, drives, team, "def"))
                row.update(_pace(subset, team, "off"))
                row.update(_pace(subset, team, "def"))
            rows.append(row)
    schema = {"game_id": games.schema["game_id"], "team": pl.Utf8}
    schema.update({c: pl.Float64 for c in TEAM_FEATURE_COLUMNS})
    return pl.DataFrame(rows, schema=schema)
