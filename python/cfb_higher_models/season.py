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
    fbs_only: bool = True,
):
    """Monte-Carlo the rest of a season through the ported nflseedR engine.

    Wraps :func:`sportsdataverse.cfb.cfb_season_odds`, which reuses the whole
    standings / conference-title / CFP-bracket machinery.

    ``fbs_only`` EXISTS BECAUSE THE RAW SURFACE IS WRONG. ``cfb_season_odds``
    builds its team set from the schedule, which contains every opponent an FBS
    team played -- 704 teams in 2023 against 133 in ``cfb_ratings``. The 571
    unrated ones are FCS/D2/D3/NAIA schools, and
    ``make_ratings_compute_results`` documents that "teams absent from it are
    treated as league-average (0.0)". So a NAIA program is simulated as a
    MEDIAN FBS TEAM, wins its single scheduled game, and enters the playoff
    field. Measured on a 2023 run, ~24% of championship probability went to
    non-FBS schools -- South Dakota State, Ave Maria, Colorado Mines, Arizona
    Christian -- while Michigan (the actual champion) held 0.569.

    "Missing -> league average" is a sound default for a team with sparse data
    and a catastrophic one for a team that does not belong in the population.
    Both look identical at the lookup: a failed join.

    With ``fbs_only=True`` the returned frame is restricted to teams that carry
    a real rating, and the probability columns are renormalised so they sum
    over the FBS field. Set it False only to reproduce the raw surface.
    """
    from sportsdataverse.cfb.cfb_season_odds import cfb_season_odds

    out = _pl(
        cfb_season_odds(
            season,
            n_sims=n_sims,
            playoff_seeds=playoff_seeds,
            seed=seed,
            as_of_date=as_of_date,
        )
    )
    if not fbs_only or not out.height:
        return out

    from sportsdataverse.cfb import load_cfb_ratings

    rated = _pl(load_cfb_ratings([season]))
    keep = set(rated["team_id"].cast(pl.Utf8).to_list())
    before = out.height
    out = out.filter(pl.col("team_id").cast(pl.Utf8).is_in(list(keep)))
    if out.height == 0:
        raise ValueError(
            f"fbs_only dropped every team for {season} -- team_id namespaces "
            "probably disagree between cfb_season_odds and cfb_ratings"
        )
    # Renormalise: the dropped teams were holding real probability mass.
    for col in ("cfp_champ_prob", "playoff_prob", "conf_title_prob"):
        if col in out.columns:
            total = float(out[col].sum() or 0.0)
            if col == "cfp_champ_prob" and total > 0:
                out = out.with_columns((pl.col(col) / total).alias(col))
    print(
        f"simulate_season {season}: kept {out.height}/{before} teams (dropped {before - out.height} unrated non-FBS)"
    )
    return out


def make_blend_compute_results(
    pred_by_game: dict[int, float], *, margin_sd: float, seed: int = 0
):
    """A `cfb_simulations` sampler driven by the BLEND instead of the closed form.

    The shipped simulator samples ``Normal(closed_form_margin, margin_sd)``. The
    closed form measures MAE 15.17 out-of-sample; the blend measures 12.68 and
    is the only model here whose advantage survives significance testing. Every
    playoff and conference probability the engine produces inherits whichever
    margin model feeds it, so this is where the improvement actually reaches a
    user-facing number.

    ``pred_by_game`` maps game_id -> expected home margin, precomputed for the
    games to be simulated. Precomputing is deliberate: the blend needs the GBM
    feature spine and the filter state, and rebuilding those inside a sampler
    that runs once per simulated week per iteration would be pathological.

    Games missing from the map fall back to the engine's own default rather
    than to zero -- a silently-zero margin would make a mismatch look like a
    coin flip, which is the kind of quiet wrongness that reads as plausible.
    """
    rng = np.random.default_rng(seed)

    def compute_results(teams, games, week_num, **kwargs):
        g = games if isinstance(games, pl.DataFrame) else pl.from_pandas(games)
        if "result" not in g.columns or "week" not in g.columns:
            return {"teams": teams, "games": games}
        todo = (pl.col("week") == week_num) & pl.col("result").is_null()
        ids = g.filter(todo)["game_id"].to_list() if "game_id" in g.columns else []
        if not ids:
            return {"teams": teams, "games": games}
        draws = {
            gid: float(np.round(rng.normal(pred_by_game[gid], margin_sd)))
            for gid in ids
            if gid in pred_by_game
        }
        if not draws:
            return {"teams": teams, "games": games}
        g = g.with_columns(
            pl.when(todo & pl.col("game_id").is_in(list(draws)))
            .then(
                pl.col("game_id").map_elements(
                    lambda x: draws.get(x), return_dtype=pl.Float64
                )
            )
            .otherwise(pl.col("result"))
            .alias("result")
        )
        return {"teams": teams, "games": g}

    return compute_results


def blend_predictions_for(
    game_frame: pl.DataFrame, *, seasons: list[int]
) -> tuple[dict[int, float], float]:
    """game_id -> blended expected margin, plus the residual sd to sample with.

    Thin wrapper over :func:`ensemble.build_blend_frame` so the simulator does
    not need to know how the blend is assembled.
    """
    from .ensemble import build_blend_frame

    out, w = build_blend_frame(game_frame, seasons=seasons)
    resid_sd = float(np.std(out["pred_margin"].to_numpy() - out["margin"].to_numpy()))
    print(f"blend for simulation: {w}, residual sd {resid_sd:.2f}")
    return (
        dict(zip(out["game_id"].to_list(), out["pred_margin"].to_list())),
        resid_sd,
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
