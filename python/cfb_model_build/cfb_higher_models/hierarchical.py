"""Team strength as a two-level random effect, with the pooling strength FITTED.

Pre-registration: ``PREREG_hierarchical.md`` (committed before this file).

WHAT THIS REPLACES
------------------
Four separately hand-tuned constants in the shipped CFB ratings stack, each
approximating one piece of the same model:

    ridge lambda on the opponent adjustment   -> a Gaussian prior on team effects
    attenuation curve (slope by games played) -> measurement-error shrinkage
    prior-season carryover n/(n+k), k=4       -> the season-boundary prior
    per-team HFA shrinkage, k=40              -> partial pooling to the league mean

All four are shrinkage. A hierarchical model does all of them at once with ONE
pooling strength (``tau_team``) estimated from the data, which is also why it
removes the failure class that produced the lambda=325 no-op: there is no
free-floating penalty constant left to be silently wrong.

MODEL (fixed before fitting; see the pre-registration)
------------------------------------------------------
    sigma_obs  ~ HalfNormal(20)      # measured residual sd is 16.4
    sigma_conf ~ HalfNormal(7)
    tau_team   ~ HalfNormal(10)      # THE POOLING STRENGTH
    hfa        ~ Normal(2.5, 3)
    rho        ~ Normal(0.5, 0.3)    # prior-season carryover, estimated
    c_j        ~ Normal(0, sigma_conf)
    theta_i    ~ Normal(c_conf(i) + rho * prev_i, tau_team)
    margin     ~ Normal(theta_home - theta_away + hfa*(1-neutral), sigma_obs)

LEAKAGE BOUNDARY
----------------
A fit for ``(season s, week W)`` sees games of season ``s`` with ``week < W``
and nothing else. ``prev_i`` is the end-of-season posterior mean of season
``s-1``, so the carryover chain runs strictly forward. Both are asserted, not
assumed -- see :func:`fit_asof`.

SHAPE DISCIPLINE
----------------
The team universe and the observation array are FIXED SIZE across every fit
(observations padded and masked). Without this, jax recompiles the NUTS kernel
for every week and the walk-forward costs an order of magnitude more.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import polars as pl

# CPU host device count must be set before jax initialises, or `chain_method
# ="parallel"` silently degrades to one device and the 4 chains run serially.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

#: Priors. Stated here, in one place, because a posterior is not reproducible
#: from the data alone (bayesian.md section 4).
PRIOR_SIGMA_OBS = 20.0
PRIOR_SIGMA_CONF = 7.0
PRIOR_TAU_TEAM = 10.0
PRIOR_HFA_LOC, PRIOR_HFA_SCALE = 2.5, 3.0
PRIOR_RHO_LOC, PRIOR_RHO_SCALE = 0.5, 0.3

#: Sampler gates. A run that fails any of these has no result to report.
RHAT_MAX = 1.01
ESS_MIN = 400.0


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Corpus:
    """Fixed-shape encoding of the schedule. Indices are stable across fits."""

    games: pl.DataFrame  # game_id, season, week, home_ix, away_ix, neutral, margin
    team_ids: np.ndarray  # index -> team_id
    conf_names: list[str]  # index -> conference name
    conf_of: dict[int, np.ndarray]  # season -> (n_teams,) conference index
    max_obs: int

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    @property
    def n_conf(self) -> int:
        return len(self.conf_names)


def build_corpus(games: pl.DataFrame) -> Corpus:
    """Encode completed games into fixed team/conference index space.

    ``games`` is the raw ``load_cfb_schedule`` frame. Kept: games with at least
    one FBS participant, in the ``regular`` or ``postseason`` season types.
    Non-FBS opponents are kept as their OWN entities pooled into an ``OTHER``
    conference -- which is exactly the case partial pooling exists for -- but
    FCS-vs-FCS games are dropped (246 teams / 934 games per season instead of
    774 / 3,829, at no cost to any FBS rating).

    ``week`` IS NOT MONOTONE IN TIME. ESPN restarts the counter per season type,
    so a January bowl of season 2023 carries ``week == 1`` and would land inside
    the "week < 8" fit window of an October game -- a four-month future leak.
    ``t`` is the monotone index every as-of filter must use: ``week`` for the
    regular season, ``100 + week`` for the postseason.
    """
    g = (
        games.filter(
            pl.col("season_type").is_in(["regular", "postseason"]) & (pl.col("fbs_participant") == True)  # noqa: E712
        )
        .select(
            pl.col("game_id").cast(pl.Int64),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("season_type").cast(pl.Utf8),
            pl.col("home_id").cast(pl.Int64),
            pl.col("away_id").cast(pl.Int64),
            pl.col("neutral_site").cast(pl.Boolean),
            pl.col("home_conference").cast(pl.Utf8),
            pl.col("away_conference").cast(pl.Utf8),
            (pl.col("home_points") - pl.col("away_points")).cast(pl.Float64).alias("margin"),
        )
        .drop_nulls(["home_id", "away_id", "margin"])
        .with_columns(
            pl.when(pl.col("season_type") == "postseason")
            .then(pl.col("week") + 100)
            .otherwise(pl.col("week"))
            .alias("t")
        )
    )

    team_ids = np.sort(np.unique(np.concatenate([g["home_id"].to_numpy(), g["away_id"].to_numpy()])))
    tix = {int(t): i for i, t in enumerate(team_ids)}

    long = pl.concat(
        [
            g.select(
                pl.col("season"),
                pl.col("home_id").alias("team_id"),
                pl.col("home_conference").alias("conf"),
            ),
            g.select(
                pl.col("season"),
                pl.col("away_id").alias("team_id"),
                pl.col("away_conference").alias("conf"),
            ),
        ]
    ).with_columns(pl.col("conf").fill_null("OTHER"))
    # "OTHER" is always present, even when the FBS filter leaves no null
    # conference: it is the bucket a team with no season row falls into.
    conf_names = sorted(set(long["conf"].to_list()) | {"OTHER"})
    cix = {c: i for i, c in enumerate(conf_names)}
    other = cix["OTHER"]

    conf_of: dict[int, np.ndarray] = {}
    # Mode per (season, team): a team's conference is constant within a season,
    # but the two sides of a game can disagree on a null, so take the modal
    # non-null value rather than `first`.
    modal = (
        long.group_by(["season", "team_id", "conf"])
        .len()
        .sort("len", descending=True)
        .group_by(["season", "team_id"])
        .first()
    )
    for season in sorted(g["season"].unique().to_list()):
        arr = np.full(len(team_ids), other, dtype=np.int32)
        sub = modal.filter(pl.col("season") == season)
        for t, c in zip(sub["team_id"].to_list(), sub["conf"].to_list()):
            arr[tix[int(t)]] = cix[c]
        conf_of[int(season)] = arr

    g = g.with_columns(
        pl.col("home_id").replace_strict(tix, return_dtype=pl.Int32).alias("home_ix"),
        pl.col("away_id").replace_strict(tix, return_dtype=pl.Int32).alias("away_ix"),
        pl.col("neutral_site").cast(pl.Float64).alias("neutral"),
    ).select("game_id", "season", "week", "t", "home_ix", "away_ix", "neutral", "margin")

    # Fixed observation capacity = the largest single-season game count, so one
    # JIT compile serves every fit.
    max_obs = int(g.group_by("season").len()["len"].max())
    return Corpus(g, team_ids, conf_names, conf_of, max_obs)


def _pack(sub: pl.DataFrame, max_obs: int):
    """Pad a fit window to ``max_obs`` rows and return arrays + a mask."""
    n = sub.height
    if n > max_obs:
        raise ValueError(f"fit window {n} exceeds max_obs {max_obs}")
    h = np.zeros(max_obs, np.int32)
    a = np.zeros(max_obs, np.int32)
    nz = np.zeros(max_obs, np.float64)
    y = np.zeros(max_obs, np.float64)
    m = np.zeros(max_obs, bool)
    h[:n] = sub["home_ix"].to_numpy()
    a[:n] = sub["away_ix"].to_numpy()
    nz[:n] = sub["neutral"].to_numpy()
    y[:n] = sub["margin"].to_numpy()
    m[:n] = True
    return h, a, nz, y, m


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def _model(h, a, nz, y, mask, conf_ix, prev, n_teams, n_conf, *, centered: bool):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(PRIOR_SIGMA_OBS))
    sigma_conf = numpyro.sample("sigma_conf", dist.HalfNormal(PRIOR_SIGMA_CONF))
    tau_team = numpyro.sample("tau_team", dist.HalfNormal(PRIOR_TAU_TEAM))
    hfa = numpyro.sample("hfa", dist.Normal(PRIOR_HFA_LOC, PRIOR_HFA_SCALE))
    rho = numpyro.sample("rho", dist.Normal(PRIOR_RHO_LOC, PRIOR_RHO_SCALE))

    if centered:
        conf = numpyro.sample("conf", dist.Normal(0.0, sigma_conf).expand([n_conf]).to_event(1))
    else:
        conf_raw = numpyro.sample("conf_raw", dist.Normal(0.0, 1.0).expand([n_conf]).to_event(1))
        conf = numpyro.deterministic("conf", sigma_conf * conf_raw)

    loc = conf[conf_ix] + rho * prev
    if centered:
        theta = numpyro.sample("theta", dist.Normal(loc, tau_team).expand([n_teams]).to_event(1))
    else:
        theta_raw = numpyro.sample("theta_raw", dist.Normal(0.0, 1.0).expand([n_teams]).to_event(1))
        theta = numpyro.deterministic("theta", loc + tau_team * theta_raw)

    mu = theta[h] - theta[a] + hfa * (1.0 - nz)
    with numpyro.handlers.mask(mask=jnp.asarray(mask)):
        numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=y)


def assert_mcmc_healthy(idata, *, label: str) -> dict[str, float]:
    """Raise unless every gate in the pre-registration passes.

    Raises rather than asserts: ``python -O`` strips assert statements, and a
    gate that vanishes under an optimisation flag is not a gate.
    """
    import arviz as az

    s = az.summary(idata, var_names=["sigma_obs", "sigma_conf", "tau_team", "hfa", "rho", "theta"])
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    diag = {
        "max_rhat": float(s["r_hat"].max()),
        "min_ess_bulk": float(s["ess_bulk"].min()),
        "min_ess_tail": float(s["ess_tail"].min()),
        "divergences": float(div),
    }
    problems = []
    if div:
        problems.append(f"{div} divergent transitions")
    if diag["max_rhat"] >= RHAT_MAX:
        problems.append(f"max r_hat {diag['max_rhat']:.4f} >= {RHAT_MAX}")
    if diag["min_ess_bulk"] <= ESS_MIN:
        problems.append(f"min ess_bulk {diag['min_ess_bulk']:.0f} <= {ESS_MIN}")
    if diag["min_ess_tail"] <= ESS_MIN:
        problems.append(f"min ess_tail {diag['min_ess_tail']:.0f} <= {ESS_MIN}")
    if problems:
        raise RuntimeError(f"MCMC did not converge [{label}]: " + "; ".join(problems))
    return diag


@dataclass
class Fit:
    theta: np.ndarray  # posterior mean team strength, (n_teams,)
    hfa: float
    tau_team: float
    sigma_obs: float
    diag: dict[str, float]
    param: str  # "centered" / "non-centered" / "eb"


def _run_nuts(
    args,
    *,
    centered: bool,
    seed: int,
    warmup: int,
    draws: int,
    chains: int,
    target_accept: float = 0.8,
):
    import arviz as az
    import jax
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(
        lambda *a: _model(*a, centered=centered), target_accept_prob=target_accept
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), *args)
    return mcmc, az.from_numpyro(mcmc)


def fit_asof(
    corpus: Corpus,
    season: int,
    week: int,
    prev: np.ndarray,
    *,
    method: str = "mcmc",
    seed: int = 0,
    warmup: int = 1000,
    draws: int = 1000,
    chains: int = 4,
    target_accept: float = 0.8,
) -> Fit:
    """Fit team strength using ONLY season-``season`` games before ``week``.

    ``prev`` is the previous season's posterior mean (zeros for the first
    season). The as-of boundary is asserted here, in code.
    """
    sub = corpus.games.filter((pl.col("season") == season) & (pl.col("t") < week)).sort("game_id")
    if sub.height == 0:
        raise ValueError(f"no games before t={week} of {season}")
    # THE BOUNDARY, asserted rather than trusted. On `t`, not `week`: a
    # postseason game's `week` restarts at 1 (see build_corpus).
    if int(sub["t"].max()) >= week:
        raise AssertionError("as-of violation: fit window contains t >= W")
    if sub["season"].n_unique() != 1:
        raise AssertionError("as-of violation: fit window spans seasons")

    h, a, nz, y, mask = _pack(sub, corpus.max_obs)
    conf_ix = corpus.conf_of[season]
    args = (h, a, nz, y, mask, conf_ix, prev, corpus.n_teams, corpus.n_conf)

    if method == "eb":
        return fit_eb(h[mask], a[mask], nz[mask], y[mask], conf_ix, prev, corpus)

    # Fit BOTH parameterizations; both must clear every gate; among the
    # survivors keep the higher ESS (bayesian.md section 3 -- selecting on ESS
    # alone would pick a more confident wrong answer).
    fits: list[Fit] = []
    errors: list[str] = []
    for centered in (True, False):
        label = f"{season}w{week}/{'centered' if centered else 'non-centered'}"
        try:
            mcmc, idata = _run_nuts(args, centered=centered, seed=seed, warmup=warmup, draws=draws, chains=chains)
            diag = assert_mcmc_healthy(idata, label=label)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        post = mcmc.get_samples()
        fits.append(
            Fit(
                theta=np.asarray(post["theta"]).mean(0),
                hfa=float(np.asarray(post["hfa"]).mean()),
                tau_team=float(np.asarray(post["tau_team"]).mean()),
                sigma_obs=float(np.asarray(post["sigma_obs"]).mean()),
                diag=diag,
                param="centered" if centered else "non-centered",
            )
        )
    if not fits:
        raise RuntimeError(f"both parameterizations failed the gates: {errors}")
    return max(fits, key=lambda f: f.diag["min_ess_bulk"])


# ---------------------------------------------------------------------------
# the no-sampler control
# ---------------------------------------------------------------------------
def fit_eb(h, a, nz, y, conf_ix, prev, corpus: Corpus, *, iters: int = 40) -> Fit:
    """Same two-level model, closed form: EM on the variance components.

    The posterior mean of a Gaussian hierarchical model is a ridge whose penalty
    is the variance RATIO -- so estimating ``tau_team`` by EM is the same
    shrinkage the sampler finds, without a sampler. If this matches the MCMC
    arm, the sampler earned nothing and the report says so.
    """
    T, C = corpus.n_teams, corpus.n_conf
    n = len(y)
    X = np.zeros((n, T + 1))
    X[np.arange(n), h] = 1.0
    X[np.arange(n), a] -= 1.0
    X[:, T] = 1.0 - nz  # HFA column
    XtX = X.T @ X
    Xty = X.T @ y

    sigma2, tau2, sconf2, rho = 16.4**2, 100.0, 25.0, 0.5
    conf = np.zeros(C)
    theta = np.zeros(T + 1)
    onehot = np.zeros((T, C))
    onehot[np.arange(T), conf_ix] = 1.0
    has_prev = prev != 0.0

    for _ in range(iters):
        m = np.concatenate([conf[conf_ix] + rho * prev, [PRIOR_HFA_LOC]])
        d = np.concatenate([np.full(T, 1.0 / tau2), [1.0 / PRIOR_HFA_SCALE**2]])
        A = XtX / sigma2 + np.diag(d)
        b = Xty / sigma2 + d * m
        V = np.linalg.inv(A)
        theta = V @ b
        th = theta[:T]
        # variance components
        resid_t = th - m[:T]
        tau2 = max(float(np.mean(resid_t**2 + np.diag(V)[:T])), 1e-3)
        r = y - X @ theta
        sigma2 = max(float(np.mean(r**2)) + float(np.trace(X @ V @ X.T)) / n, 1e-3)
        # conference level: shrunk group means
        cnt = onehot.sum(0)
        gm = (onehot * (th - rho * prev)[:, None]).sum(0) / np.maximum(cnt, 1)
        conf = gm * (sconf2 / (sconf2 + tau2 / np.maximum(cnt, 1)))
        sconf2 = max(float(np.mean(conf**2)), 1e-3)
        # carryover strength, estimated the same way k=4 never was
        if has_prev.sum() > 10:
            num = float(np.dot(prev[has_prev], (th - conf[conf_ix])[has_prev]))
            den = float(np.dot(prev[has_prev], prev[has_prev])) + 1e-9
            rho = float(np.clip(num / den, 0.0, 1.0))

    return Fit(
        theta=theta[:T],
        hfa=float(theta[T]),
        tau_team=float(np.sqrt(tau2)),
        sigma_obs=float(np.sqrt(sigma2)),
        diag={"max_rhat": float("nan"), "min_ess_bulk": float("nan"), "min_ess_tail": float("nan"), "divergences": 0.0},
        param="eb",
    )


def predict(fit: Fit, games: pl.DataFrame) -> np.ndarray:
    """Predicted home margin for an already-index-encoded game frame."""
    h = games["home_ix"].to_numpy()
    a = games["away_ix"].to_numpy()
    nz = games["neutral"].to_numpy()
    return fit.theta[h] - fit.theta[a] + fit.hfa * (1.0 - nz)


def walk_forward(
    corpus: Corpus,
    seasons: list[int],
    *,
    method: str = "mcmc",
    verbose: bool = True,
    **fit_kw,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Forward-only pass: every game predicted from strictly earlier games.

    Season ``s`` is fit from its own completed games only; the previous
    season's END-OF-SEASON posterior enters as the ``prev`` prior (that is the
    carryover, with its strength ``rho`` estimated rather than set to k=4).

    The carryover chain runs forward, so no future season can reach a prior.
    Returns (per-game predictions, per-fit diagnostics, voided as-of points).
    """
    preds: list[pl.DataFrame] = []
    diags: list[dict] = []
    voids: list[dict] = []
    prev = np.zeros(corpus.n_teams)
    for s in seasons:
        ts = sorted(corpus.games.filter(pl.col("season") == s)["t"].unique().to_list())
        for tt in ts[1:]:  # the first slot has no in-season history
            try:
                fit = fit_asof(corpus, s, tt, prev, method=method, **fit_kw)
            except RuntimeError as exc:
                # PRE-REGISTERED: a fit that fails the sampler gates is VOID --
                # it yields no prediction, and the common-set rule then drops
                # those games from EVERY arm so the comparison stays
                # like-for-like. Recorded, never silently swallowed.
                voids.append({"season": s, "t": tt, "error": str(exc)})
                if verbose:
                    print(f"  {s} t={tt:>3} VOID (gates): {exc}", flush=True)
                continue
            te = corpus.games.filter((pl.col("season") == s) & (pl.col("t") == tt))
            preds.append(
                te.select("game_id", "season", "week", "t", "margin").with_columns(
                    pl.Series("pred_margin", predict(fit, te))
                )
            )
            diags.append(
                {
                    "season": s,
                    "t": tt,
                    "n_fit": int((corpus.games["season"] == s).sum()),
                    "param": fit.param,
                    "hfa": fit.hfa,
                    "tau_team": fit.tau_team,
                    "sigma_obs": fit.sigma_obs,
                    **fit.diag,
                }
            )
            if verbose:
                print(
                    f"  {s} t={tt:>3} {fit.param:<12} n_te={te.height:>4} "
                    f"hfa={fit.hfa:5.2f} tau={fit.tau_team:6.2f} "
                    f"rhat={fit.diag['max_rhat']:.4f} ess={fit.diag['min_ess_bulk']:.0f} "
                    f"div={fit.diag['divergences']:.0f}",
                    flush=True,
                )
        # The carryover fit uses the WHOLE of season s and is never void in
        # practice (it has the most data of any fit); if it were, the chain
        # would break silently, so it is allowed to raise.
        final = fit_asof(corpus, s, max(ts) + 1, prev, method=method, **fit_kw)
        prev = final.theta
        if verbose:
            print(f"  {s} FINAL carryover prior set (theta sd {prev.std():.2f})", flush=True)
    if voids:
        print(
            f"VOID as-of points: {len(voids)}/{len(voids) + len(preds)} "
            f"({100 * len(voids) / (len(voids) + len(preds)):.1f}%) -- "
            f"{sorted({v['t'] for v in voids})}",
            flush=True,
        )
    return pl.concat(preds), pl.DataFrame(diags), pl.DataFrame(voids)
