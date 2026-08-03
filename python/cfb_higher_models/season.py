"""Season-level surfaces: resume, luck, strength of schedule / record, sims.

Sits on the nflseedR-derived engine already ported into sdv-py
(`cfb_simulations` + `cfb_season_odds` + `cfb_resume`), fed with ratings that
have been priced in POINTS rather than raw EPA -- see points_scale for why that
matters (special teams looked 4x bigger than offence while predicting 4.7x
less).

The four questions this answers, kept deliberately distinct because they are
routinely conflated:

    strength of schedule   how hard were the opponents?
    strength of record     how impressive is this record GIVEN that schedule?
    luck                   how far is the record from what the team's own
                           play-level performance implies?
    projection             what happens from here?

SOS and SOR are not the same statistic and neither is "luck". A 10-2 team with
a brutal schedule has a strong SOR; a 10-2 team whose margins say 8-4 is lucky.
Both can be true at once, and a single "rating" that blends them answers
neither question.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .points_scale import PointsModel, fit_points_model


def _pl(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def team_game_expectations(frame: pl.DataFrame, model: PointsModel) -> pl.DataFrame:
    """Long form: one row per team-game with expected and actual margin.

    Expected margin comes from the points model, so "expected" is in the same
    units as the result and the difference is directly interpretable as points
    of over/under-performance.
    """
    pred = model.predict(frame)
    base = frame.select("game_id", "season", "week", "home_id", "away_id", "margin")
    base = base.with_columns(pl.Series("exp_margin", pred))
    home = base.select(
        "game_id",
        "season",
        "week",
        pl.col("home_id").alias("team_id"),
        pl.col("away_id").alias("opp_id"),
        pl.col("margin"),
        pl.col("exp_margin"),
    )
    away = base.select(
        "game_id",
        "season",
        "week",
        pl.col("away_id").alias("team_id"),
        pl.col("home_id").alias("opp_id"),
        (-pl.col("margin")).alias("margin"),
        (-pl.col("exp_margin")).alias("exp_margin"),
    )
    return pl.concat([home, away]).with_columns(
        (pl.col("margin") > 0).cast(pl.Float64).alias("win"),
        # Win probability implied by the expected margin, under the model's own
        # residual spread -- the bridge from points to expected wins.
        (1.0 / (1.0 + (-pl.col("exp_margin") / (model.resid_sd * 0.5513)).exp())).alias(
            "exp_win"
        ),
    )


def team_strength_points(frame: pl.DataFrame, model: PointsModel) -> pl.DataFrame:
    """Each team's own strength in points, independent of whom it played.

    Strength of schedule must be built from OPPONENT strength. An earlier
    version defined SOS as the mean of ``-exp_margin``, which contains the
    team's OWN rating -- so a terrible team, whose expected margin against
    everyone is deeply negative, scored a huge "SOS". The 2024 leaderboard it
    produced was Southern Miss (0-9), Purdue (0-9), Ball State (2-7) and UMass
    (0-8): a ranking of bad teams wearing a schedule-strength label. Reading
    the extremes is what caught it.
    """
    rows = []
    for side in ("home", "away"):
        cols = {c: f"{c}_{side}" for c in model.coef}
        if not all(v in frame.columns for v in cols.values()):
            continue
        strength = None
        for c, v in cols.items():
            term = pl.col(v).fill_null(0.0) * model.coef[c]
            strength = term if strength is None else strength + term
        rows.append(
            frame.select(
                "season",
                pl.col(f"{side}_id").alias("team_id"),
                strength.alias("strength_points"),
            )
        )
    if not rows:
        raise ValueError("team_strength_points: no rating columns matched the model")
    return (
        pl.concat(rows)
        .group_by(["season", "team_id"])
        .agg(pl.col("strength_points").mean())
    )


def season_resume(
    frame: pl.DataFrame, model: PointsModel | None = None
) -> pl.DataFrame:
    """SOS, SOR, luck and expected wins, one row per team-season.

    * ``sos_points``    mean OPPONENT strength in points -- built from the
                        opponents' own ratings, never from this team's expected
                        margin (see :func:`team_strength_points`).
    * ``exp_wins``      sum of per-game win probabilities from the model.
    * ``wins``          actual, WITHIN THE FRAME (see caveat below).
    * ``luck_wins``     wins - exp_wins. Positive means the record outruns the
                        play-level performance.
    * ``sor_points``    strength of record: the average quality of TEAM that
                        would be expected to produce this record against this
                        schedule. Approximated as the rating shift needed to
                        make exp_wins equal actual wins.

    CAVEAT -- RECORDS ARE PARTIAL. ``build_game_frame`` drops week 1 (no prior
    week to be as-of), unrated early-season rows, and any game where either
    team lacks a snapshot. So a team shows ~9-11 games, not 12-13, and ``wins``
    is its record over the SCORED SUBSET. Liberty's 2023 reads 11-0 against an
    actual 13-0. Rates (luck per game, SOS, SOR) are meaningful; raw win totals
    are not season records and must not be presented as such.
    """
    model = model or fit_points_model(frame)
    tg = team_game_expectations(frame, model)
    strength = team_strength_points(frame, model)
    # Attach each opponent's OWN strength, then average it -- that is the
    # schedule. Joining on the opponent id is the whole difference between
    # measuring the schedule and measuring the team.
    tg = tg.join(
        strength.rename({"team_id": "opp_id", "strength_points": "opp_strength"}),
        on=["season", "opp_id"],
        how="left",
    )
    agg = tg.group_by(["season", "team_id"]).agg(
        pl.len().alias("games"),
        pl.col("win").sum().alias("wins"),
        pl.col("exp_win").sum().alias("exp_wins"),
        pl.col("margin").mean().alias("margin_pg"),
        pl.col("exp_margin").mean().alias("exp_margin_pg"),
        pl.col("opp_strength").mean().alias("sos_points"),
    )
    return (
        agg.with_columns(
            (pl.col("wins") - pl.col("exp_wins")).alias("luck_wins"),
            (pl.col("margin_pg") - pl.col("exp_margin_pg")).alias("luck_points_pg"),
            (pl.col("wins") / pl.col("games")).alias("win_pct"),
        )
        .with_columns(
            # SOR: how many points better than average a team would have to be to
            # earn this record on this schedule. Solved by a simple shift on the
            # logistic used for exp_win.
            (
                (pl.col("wins") / pl.col("games")).clip(0.01, 0.99).log()
                - (1 - (pl.col("wins") / pl.col("games")).clip(0.01, 0.99)).log()
            ).alias("_logit")
        )
        .with_columns(
            (pl.col("_logit") * model.resid_sd * 0.5513 + pl.col("sos_points")).alias(
                "sor_points"
            )
        )
        .drop("_logit")
    )


def simulate_season(
    season: int,
    *,
    n_sims: int = 10000,
    playoff_seeds: int = 12,
    seed: int = 0,
    as_of_date=None,
):
    """Monte-Carlo the rest of a season through the ported nflseedR engine.

    Thin pass-through to :func:`sportsdataverse.cfb.cfb_season_odds`, which
    reuses the whole standings / conference-title / CFP-bracket machinery. Kept
    here so the season surfaces live in one place and so the ratings feeding it
    can later be swapped for the points-scale ones without callers changing.
    """
    from sportsdataverse.cfb.cfb_season_odds import cfb_season_odds

    return _pl(
        cfb_season_odds(
            season,
            n_sims=n_sims,
            playoff_seeds=playoff_seeds,
            seed=seed,
            as_of_date=as_of_date,
        )
    )


def luck_report(resume: pl.DataFrame, *, season: int, n: int = 10) -> str:
    """Most and least fortunate teams -- the sanity check on a luck metric.

    A luck metric that does not surface recognisably fortunate seasons is
    measuring something else. Reading the extremes is the cheapest available
    validation.
    """
    s = resume.filter(pl.col("season") == season).sort("luck_wins", descending=True)
    L = [f"luckiest {season}:"]
    for r in s.head(n).to_dicts():
        L.append(
            f"  {r['team_id']:>8}  {r['wins']:.0f}-{r['games'] - r['wins']:.0f}  "
            f"exp {r['exp_wins']:.1f}  luck {r['luck_wins']:+.1f}  "
            f"sos {r['sos_points']:+.1f}"
        )
    L.append(f"unluckiest {season}:")
    for r in s.tail(n).to_dicts():
        L.append(
            f"  {r['team_id']:>8}  {r['wins']:.0f}-{r['games'] - r['wins']:.0f}  "
            f"exp {r['exp_wins']:.1f}  luck {r['luck_wins']:+.1f}  "
            f"sos {r['sos_points']:+.1f}"
        )
    return "\n".join(L)
