"""Team strength as a filtered state, not a season-to-date average.

WHAT THIS REPLACES
------------------
Four separately hand-tuned constants, each approximating a piece of one model:

    ridge lambda on the opponent adjustment   -> a prior on team effects
    attenuation curve (slope by games played) -> measurement-error correction
    prior-season carryover n/(n+k), k=4       -> the season-boundary prior
    per-team HFA shrinkage, k=40              -> partial pooling

All four are shrinkage. A Kalman filter does all of them at once, with the
shrinkage DERIVED from the ratio of process to observation variance rather
than fitted per-hack. Early-season uncertainty is native: a team with two
games has a wide posterior, so its rating stays near the prior, which IS the
attenuation curve without four hand-set buckets.

It also removes a pipeline: the filter produces an as-of rating before every
game by construction, so the 22-seasons x 15-snapshots weekly rebuild exists
only to approximate what this computes in one pass.

WHY A FILTER AND NOT MCMC
-------------------------
The model is linear-Gaussian, so the Kalman recursion is EXACT -- there is no
approximation for sampling to improve on, and it runs in milliseconds where
MCMC would take minutes per fit. Sampling earns its place only if the
likelihood stops being Gaussian (heavy-tailed margins are the real candidate;
see the residual check in `diagnose`).

MODEL
-----
    state      x_t  = team strength vector (one entry per team)
    evolution  x_t  = rho * x_{t-1} + w,   w ~ N(0, Q)
    obs        y    = x_home - x_away + hfa + v,   v ~ N(0, R)

`rho < 1` mean-reverts toward the league average, which is what teams actually
do between seasons; a pure random walk would let a good team drift forever.
Q is larger across a season boundary (roster turnover) than within one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class StateSpaceConfig:
    """Filter parameters. Estimated by predictive likelihood, not chosen.

    Defaults are only a starting point for :func:`fit_params`; shipping them
    unfitted would recreate exactly the hand-tuned-constant problem this
    module exists to remove.
    """

    r_obs: float = 16.4**2  # observation variance; measured residual sd is 16.4
    q_week: float = 0.35  # within-season drift per week
    q_season: float = 12.0  # extra variance across a season boundary
    rho_season: float = 0.72  # mean reversion between seasons
    p0: float = 100.0  # prior variance for an unseen team
    hfa: float = 2.42  # measured league home-field advantage, in points


@dataclass
class FilterState:
    teams: dict[int, int] = field(default_factory=dict)
    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    P: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def index(self, team_id: int, p0: float) -> int:
        """Team index, growing the state for a team seen for the first time."""
        if team_id in self.teams:
            return self.teams[team_id]
        i = len(self.teams)
        self.teams[team_id] = i
        self.mu = np.append(self.mu, 0.0)
        P = np.zeros((i + 1, i + 1))
        if i:
            P[:i, :i] = self.P
        P[i, i] = p0
        self.P = P
        return i


def _predict(st: FilterState, *, q: float, rho: float = 1.0) -> None:
    """Time update: drift the state and inflate its uncertainty."""
    if rho != 1.0:
        st.mu *= rho
        st.P *= rho * rho
    st.P[np.diag_indices_from(st.P)] += q


def _update(st: FilterState, h: int, a: int, y: float, r: float) -> float:
    """Measurement update for one game. Returns the predictive residual.

    H = e_h - e_a, so this is a rank-1 Kalman update -- exact, and O(n^2) in
    the number of teams rather than the O(n^3) a naive matrix solve would
    cost.
    """
    Ph = st.P[:, h] - st.P[:, a]
    s = float(Ph[h] - Ph[a] + r)
    resid = float(y - (st.mu[h] - st.mu[a]))
    K = Ph / s
    st.mu += K * resid
    st.P -= np.outer(K, Ph)
    return resid


def run_filter(games: pl.DataFrame, cfg: StateSpaceConfig, *, collect: bool = True) -> tuple[FilterState, pl.DataFrame]:
    """Filter forward through games in chronological order.

    Returns the final state and, when ``collect``, one row per game carrying
    the PRE-GAME rating of each side. Those are as-of by construction: the
    prediction is emitted before the update that uses the result, so there is
    no way for a game to inform its own forecast.
    """
    g = games.sort(["season", "week"])
    st = FilterState()
    rows = []
    prev_season = None
    prev_week = None
    for rec in g.iter_rows(named=True):
        season, week = rec["season"], rec["week"]
        if prev_season is None:
            pass
        elif season != prev_season:
            _predict(st, q=cfg.q_season, rho=cfg.rho_season)
        elif week != prev_week:
            _predict(st, q=cfg.q_week * max(1, week - (prev_week or week)))
        prev_season, prev_week = season, week

        h = st.index(int(rec["home_id"]), cfg.p0)
        a = st.index(int(rec["away_id"]), cfg.p0)
        hfa = 0.0 if rec.get("neutral_site") else cfg.hfa
        pred = float(st.mu[h] - st.mu[a]) + hfa
        if collect:
            rows.append(
                {
                    "game_id": rec["game_id"],
                    "season": season,
                    "week": week,
                    "ss_pred": pred,
                    "ss_home": float(st.mu[h]),
                    "ss_away": float(st.mu[a]),
                    # posterior sd of the matchup -- the model's own statement
                    # of how well it knows THIS pairing
                    "ss_sd": float(np.sqrt(max(st.P[h, h] + st.P[a, a] - 2 * st.P[h, a], 0.0))),
                    "margin": rec["margin"],
                }
            )
        _update(st, h, a, float(rec["margin"]) - hfa, cfg.r_obs)
    return st, (pl.DataFrame(rows) if rows else pl.DataFrame())


def predictive_loglik(games: pl.DataFrame, cfg: StateSpaceConfig) -> float:
    """Total one-step-ahead predictive log-likelihood.

    The objective for fitting the variance parameters. It is the natural
    criterion for a filter: every game is scored by the forecast made before
    it, so maximising it cannot overfit in the way a whole-sample likelihood
    would.
    """
    _, out = run_filter(games, cfg)
    if not out.height:
        return -np.inf
    resid = out["margin"].to_numpy() - out["ss_pred"].to_numpy()
    var = out["ss_sd"].to_numpy() ** 2 + cfg.r_obs
    return float(-0.5 * np.sum(np.log(2 * np.pi * var) + resid**2 / var))


def fit_params(games: pl.DataFrame, *, base: StateSpaceConfig | None = None, verbose: bool = True) -> StateSpaceConfig:
    """Estimate q_week / q_season / rho_season by predictive likelihood.

    A coarse coordinate search rather than a gradient method: the surface is
    smooth and low-dimensional, three passes converge, and it avoids a
    dependency for a 20-line optimisation. r_obs is pinned to the MEASURED
    residual variance (sd 16.4, verified homoscedastic in `distributional`)
    rather than fitted, because it is identifiable from the data directly.
    """
    cfg = base or StateSpaceConfig()
    # Widened after the first fit pinned TWO parameters to their edges (q_week
    # at the 0.05 floor, q_season at the 50.0 ceiling). A boundary solution is
    # not an estimate -- it is the search saying it wanted to keep going, and
    # reporting one as "fitted" would be exactly the hand-tuned-constant
    # problem this module exists to remove.
    grids = {
        "q_week": [0.0, 0.005, 0.02, 0.05, 0.15, 0.35, 0.7, 1.5, 3.0],
        "q_season": [2.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0, 400.0],
        "rho_season": [0.3, 0.45, 0.6, 0.72, 0.8, 0.85, 0.9, 0.95, 1.0],
    }
    # Always include the incumbent value in each coordinate's candidate set.
    # Widening the grids once WITHOUT this dropped rho=0.8 (the previous
    # optimum sat between the new 0.72 and 0.85 points) and the "wider" search
    # returned a WORSE loglik, -7298.3 against -7292.9. A search that cannot
    # reproduce its own starting point is not a widening.
    best = predictive_loglik(games, cfg)
    for _ in range(3):
        for name, grid in grids.items():
            cur = getattr(cfg, name)
            candidates = sorted({*grid, cur})
            scores = []
            for v in candidates:
                setattr(cfg, name, v)
                scores.append((predictive_loglik(games, cfg), v))
            ll, v = max(scores)
            if ll > best:
                best, cur = ll, v
            setattr(cfg, name, cur)
    if verbose:
        print(f"  fitted: q_week={cfg.q_week} q_season={cfg.q_season} rho_season={cfg.rho_season}  loglik={best:.1f}")
        for name, grid in grids.items():
            v = getattr(cfg, name)
            if v in (min(grid), max(grid)):
                print(
                    f"  !! {name}={v} sits at a grid EDGE -- the optimum may be "
                    "outside the search range. Widen before quoting it."
                )
    return cfg


def diagnose(out: pl.DataFrame, cfg: StateSpaceConfig) -> dict[str, float]:
    """Is the Gaussian assumption holding? Decides whether MCMC is warranted.

    Standardised residuals should be N(0,1). Excess kurtosis is the tell for
    heavy tails -- if margins have them, a t-likelihood (which needs sampling)
    would be the justified next step rather than a speculative one.
    """
    from scipy import stats

    resid = out["margin"].to_numpy() - out["ss_pred"].to_numpy()
    z = resid / np.sqrt(out["ss_sd"].to_numpy() ** 2 + cfg.r_obs)
    return {
        "resid_sd": float(resid.std()),
        "z_mean": float(z.mean()),
        "z_sd": float(z.std()),
        "excess_kurtosis": float(stats.kurtosis(z)),
        "shapiro_p": float(stats.shapiro(z[:4000]).pvalue) if len(z) > 20 else np.nan,
    }
