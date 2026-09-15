"""Unit G/H of the matchup pipeline: pregame ELO handling, opponent-ELO rolls, betting.

Port of ``2025_cfb_update.R`` 1463-1548 and 2183-2440. Inputs are the CFBD
``/games`` rows (``schedules_unified.fetch_cfbd_games`` payload, or the parquet
fixture) reduced by :func:`elo_games`; nothing here touches the network.

The pipeline's rules, in order:

1. a game's pregame ELO is CFBD's, else the team's final postgame ELO of the
   prior season (preseason carry-forward);
2. then, per season, the 10th percentile of pregame ELO over FBS-vs-FBS games
   replaces (a) an FBS team's still-missing value when there is no prior
   final either and (b) EVERY FCS team's value, present or not;
3. opponent-ELO rolls: for each team, the sum / mean / median of the
   (filled) pregame ELO of the opponents it faced EARLIER in the season, by
   real date -- never the current game;
4. a regular-season week-1 row has no prior opponents, so it takes the team's
   final roll of the prior season, else the 10th percentile of those finals
   over prior-season FBS teams (sum falls back to 11 x the ELO percentile).

Team names are CFBD's; the matchup line applies its own display mapping.
"""

from __future__ import annotations

import polars as pl

ELO_GAME_COLUMNS = (
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "home_team",
    "away_team",
    "home_division",
    "away_division",
    "home_pregame_elo",
    "away_pregame_elo",
    "home_postgame_elo",
    "away_postgame_elo",
)

#: R's ``1200`` when no FBS-vs-FBS game carries an ELO at all.
_DEFAULT_ELO_10TH = 1200.0


def elo_games(games: pl.DataFrame) -> pl.DataFrame:
    """Reduce a CFBD games frame to the ELO columns with a Date ``game_date``."""
    return games.select(ELO_GAME_COLUMNS).with_columns(
        game_date=pl.col("start_date").str.slice(0, 10).str.to_date(),
        home_pregame_elo=pl.col("home_pregame_elo").cast(pl.Float64),
        away_pregame_elo=pl.col("away_pregame_elo").cast(pl.Float64),
        home_postgame_elo=pl.col("home_postgame_elo").cast(pl.Float64),
        away_postgame_elo=pl.col("away_postgame_elo").cast(pl.Float64),
    )


def _long(games: pl.DataFrame, value: str) -> pl.DataFrame:
    """Stack home and away rows: (game_id, season, week, game_date, team, <value>, opp_<value>)."""
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        parts.append(
            games.select(
                "game_id",
                "season",
                "week",
                "game_date",
                team=pl.col(f"{side}_team"),
                division=pl.col(f"{side}_division"),
                value=pl.col(f"{side}_{value}"),
                opp_value=pl.col(f"{other}_{value}"),
            )
        )
    return pl.concat(parts)


def prior_final_elo(prev_games: pl.DataFrame) -> pl.DataFrame:
    """Each team's LAST non-null postgame ELO of the prior season: (team, prev_elo)."""
    return (
        _long(prev_games, "postgame_elo")
        .drop_nulls("value")
        .sort(["team", "game_date"], maintain_order=True)
        .group_by("team", maintain_order=True)
        .agg(prev_elo=pl.col("value").last())
    )


def fbs_elo_10th(games: pl.DataFrame) -> pl.DataFrame:
    """Per season, the 10th percentile (R type-7) of pregame ELO over FBS-vs-FBS games."""
    fbs = games.filter((pl.col("home_division") == "fbs") & (pl.col("away_division") == "fbs"))
    stacked = pl.concat(
        [
            fbs.select("season", elo="home_pregame_elo"),
            fbs.select("season", elo="away_pregame_elo"),
        ]
    ).drop_nulls("elo")
    return stacked.group_by("season").agg(elo_10th=pl.col("elo").quantile(0.10, interpolation="linear"))


def fill_pregame_elo(games: pl.DataFrame, prev_final: pl.DataFrame) -> pl.DataFrame:
    """Rules 1 and 2: carry-forward, then the FBS/FCS percentile substitution."""
    pct = fbs_elo_10th(games)
    global_10th = pct["elo_10th"].drop_nulls()
    global_10th = float(global_10th.median()) if global_10th.len() else _DEFAULT_ELO_10TH
    out = games
    for side in ("home", "away"):
        out = (
            out.join(
                prev_final.rename({"team": f"{side}_team", "prev_elo": f"_{side}_prev"}),
                on=f"{side}_team",
                how="left",
            )
            .with_columns(pl.coalesce(f"{side}_pregame_elo", f"_{side}_prev").alias(f"{side}_pregame_elo"))
            .drop(f"_{side}_prev")
        )
    out = out.join(pct, on="season", how="left").with_columns(elo_10th=pl.col("elo_10th").fill_null(global_10th))
    for side in ("home", "away"):
        elo, div = pl.col(f"{side}_pregame_elo"), pl.col(f"{side}_division")
        out = out.join(
            prev_final.rename({"team": f"{side}_team", "prev_elo": f"_{side}_prev"}),
            on=f"{side}_team",
            how="left",
        )
        out = out.with_columns(
            pl.when(elo.is_null() & (div == "fbs"))
            .then(pl.coalesce(f"_{side}_prev", "elo_10th"))
            .when(elo.is_null() | (div == "fcs"))
            .then(pl.col("elo_10th"))
            .otherwise(elo)
            .alias(f"{side}_pregame_elo")
        ).drop(f"_{side}_prev")
    return out.drop("elo_10th")


def _expanding(values: pl.Series) -> tuple[list, list, list]:
    """R ``.roll_prior``: stats over the elements BEFORE each position, NA when none."""
    sums, avgs, meds = [], [], []
    for i in range(values.len()):
        prior = values[:i].drop_nulls()
        if prior.len() == 0:
            sums.append(None), avgs.append(None), meds.append(None)
        else:
            (
                sums.append(prior.sum()),
                avgs.append(prior.mean()),
                meds.append(prior.median()),
            )
    return sums, avgs, meds


def opponent_elo_rolls(games_filled: pl.DataFrame) -> pl.DataFrame:
    """Rule 3 over one season (or several): (game_id, season, week, team, opp_elo_roll_*)."""
    long = _long(games_filled, "pregame_elo").sort(["season", "team", "game_date", "week"], maintain_order=True)
    frames = []
    for _, grp in long.group_by(["season", "team"], maintain_order=True):
        s, a, m = _expanding(grp["opp_value"])
        frames.append(
            grp.select(
                "game_id",
                "season",
                "week",
                "game_date",
                "team",
                opp_elo=pl.col("opp_value"),
            ).with_columns(
                opp_elo_roll_sum=pl.Series(s, dtype=pl.Float64),
                opp_elo_roll_avg=pl.Series(a, dtype=pl.Float64),
                opp_elo_roll_median=pl.Series(m, dtype=pl.Float64),
            )
        )
    return pl.concat(frames)


def fill_week1_rolls(rolls: pl.DataFrame, prev_rolls: pl.DataFrame, prev_games_filled: pl.DataFrame) -> pl.DataFrame:
    """Rule 4: week-1 rows take the prior season's final rolls, else their 10th percentile."""
    prev_final = (
        prev_rolls.sort(["team", "game_date", "week"], maintain_order=True)
        .group_by("team", maintain_order=True)
        .agg(
            final_sum=pl.col("opp_elo_roll_sum").last(),
            final_avg=pl.col("opp_elo_roll_avg").last(),
            final_median=pl.col("opp_elo_roll_median").last(),
        )
    )
    prev_fbs_teams = pl.concat(
        [
            prev_games_filled.filter(pl.col("home_division") == "fbs").select(team="home_team"),
            prev_games_filled.filter(pl.col("away_division") == "fbs").select(team="away_team"),
        ]
    ).unique()
    fbs_finals = prev_final.join(prev_fbs_teams, on="team", how="semi")
    prev_elo_10th = fbs_elo_10th(prev_games_filled)["elo_10th"]
    prev_elo_10th = (
        float(prev_elo_10th[0]) if prev_elo_10th.len() and prev_elo_10th[0] is not None else _DEFAULT_ELO_10TH
    )

    def q(col: str, fallback: float) -> float:
        v = fbs_finals[col].drop_nulls()
        return float(v.quantile(0.10, interpolation="linear")) if v.len() else fallback

    q_sum, q_avg, q_med = (
        q("final_sum", prev_elo_10th * 11),
        q("final_avg", prev_elo_10th),
        q("final_median", prev_elo_10th),
    )
    out = rolls.join(prev_final, on="team", how="left")
    week1 = pl.col("week") == 1
    return out.with_columns(
        opp_elo_roll_sum=pl.when(week1)
        .then(pl.coalesce("opp_elo_roll_sum", "final_sum", pl.lit(q_sum)))
        .otherwise("opp_elo_roll_sum"),
        opp_elo_roll_avg=pl.when(week1)
        .then(pl.coalesce("opp_elo_roll_avg", "final_avg", pl.lit(q_avg)))
        .otherwise("opp_elo_roll_avg"),
        opp_elo_roll_median=pl.when(week1)
        .then(pl.coalesce("opp_elo_roll_median", "final_median", pl.lit(q_med)))
        .otherwise("opp_elo_roll_median"),
    ).drop("final_sum", "final_avg", "final_median")


def is_playoff(notes: pl.Expr) -> pl.Expr:
    """CFBD ``notes`` naming a College Football Playoff game (bowls do not)."""
    return notes.fill_null("").str.contains("(?i)college football playoff|national championship")


def matchup_scope(games: pl.DataFrame) -> pl.DataFrame:
    """FBS-vs-FBS games: the set the opponent-ELO rolls are defined over.

    The pipeline recomputes every frozen season's rolls from its static
    master, which holds only FBS-vs-FBS games, so a team's FCS opener never
    counts toward its roll and a week-2 row after an FCS opener is null. Bowls
    stay in: the master's older seasons carry them (the delivered 2025 week-1
    rows include each team's 2024 bowl opponent), and a season's bowls fall
    after every row the line keeps, so they cannot leak into a kept row of the
    same season. The line drops bowl ROWS separately (:func:`is_playoff`).
    """
    return games.filter((pl.col("home_division") == "fbs") & (pl.col("away_division") == "fbs"))


def season_elo(games: pl.DataFrame, prev_games: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """All four rules for one season given the prior season's games.

    Returns ``(games_filled, rolls)``: the season's games (ALL of them) with
    filled ``home/away_pregame_elo``, and the per (game_id, team) opponent-ELO
    rolls over the :func:`matchup_scope` games of each season. Prior-season
    finals for the carry-forward come from every prior-season game.
    """
    cur, prev = elo_games(games), elo_games(prev_games)
    prev_final = prior_final_elo(prev)
    cur_filled = fill_pregame_elo(cur, prev_final)
    # the prior season's rolls are rebuilt from its frozen master rows, whose
    # ELO carried no further carry-forward
    prev_filled = fill_pregame_elo(prev, prev_final.clear())
    scope_cur = matchup_scope(games).select("game_id")
    scope_prev = matchup_scope(prev_games).select("game_id")
    rolls = fill_week1_rolls(
        opponent_elo_rolls(cur_filled.join(scope_cur, on="game_id", how="semi")),
        opponent_elo_rolls(prev_filled.join(scope_prev, on="game_id", how="semi")),
        prev_filled,
    )
    return cur_filled, rolls


# --------------------------------------------------------------------------- H
def consensus_lines(lines: pl.DataFrame) -> pl.DataFrame:
    """Per game, the mean over providers of spread / spread_open / over_under / over_under_open.

    A ``"pk"`` (pick) spread is 0; anything non-numeric is null and ignored by
    the mean, as R's ``as.numeric`` + ``mean(na.rm = TRUE)`` do.
    """

    def num(col: str) -> pl.Expr:
        s = pl.col(col).cast(pl.Utf8)
        return pl.when(s.str.to_lowercase() == "pk").then(pl.lit("0")).otherwise(s).cast(pl.Float64, strict=False)

    return (
        lines.with_columns(
            spread=num("spread"),
            spread_open=num("spread_open"),
            over_under=num("over_under"),
            over_under_open=num("over_under_open"),
        )
        .group_by("game_id", maintain_order=True)
        .agg(
            spread=pl.col("spread").mean(),
            spread_open=pl.col("spread_open").mean(),
            over_under=pl.col("over_under").mean(),
            over_under_open=pl.col("over_under_open").mean(),
        )
    )
