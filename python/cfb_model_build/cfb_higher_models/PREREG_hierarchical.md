# CFB hierarchical team strength — stocktake §8 item 13, first entry

**Status:** PRE-REGISTRATION COMMITTED 2026-09-02. Results appended below only after
the sampler gates are read.

Worktree: `C:\Users\saiem\Documents\GitHub-Data\sdv-dev\cfbfastR-dev\cfb-data-wt-hier`
Branch: `feat/cfb-hierarchical-team-strength` off `origin/main` @ `25976c86`.

---

## 0. Why this model and not another one

The published CFB ratings surface (`cfb_ratings` → `cfb_game_predict`) is a ridge +
fixed-point system whose only fitted numbers are module constants in
`cfb_prediction_constants.py`. The 2026-08-03 Bayesian evaluation
(`notes/2026-08-03-cfb-bayesian-evaluation.md`) established, from measured gains, that
**four separately hand-tuned constants in that stack are all the same model**:

| hand-tuned constant | what it actually is | measured gain |
|---|---|---|
| attenuation curve `slope_by_games` (10.6 / 26.1 / 42.0 / 54.5) | errors-in-variables correction | −0.58 MAE |
| prior-season carryover `n/(n+k)`, k=4 | empirical-Bayes shrinkage to a prior | −0.23 MAE |
| per-team HFA, k=40 | partial pooling to the league mean | −0.22 vs no pooling |
| ridge λ on the opponent adjustment | a Gaussian prior on team effects | (the λ=325 no-op cost a full republish) |

A two-level hierarchical model estimates all four as one shrinkage parameter **from the
data**, which is also why it removes the failure class that produced the λ no-op: there is
no free-floating penalty constant left to be silently wrong. That note's estimate of the
gain was **≈0.15 MAE against a measured 0.067 MAE minimum detectable effect**, i.e.
testable. It has never been built. `state_space.py` (the Kalman half) exists in the
package but is referenced by nothing and has no measured number — an orphan.

CFB is also the shape partial pooling is *for*: ~134 FBS teams × ~12 games, in conferences
that barely cross-play. A flat estimator either over-fits a thin schedule or is dominated
by strength-of-schedule artifacts.

## 1. Model (fixed before fitting; no tuning is permitted before the primary readout)

Fit per as-of point `(season s, week W)` on **completed games of season `s` with
`week < W` only**. Team universe is fixed across every fit so JIT shapes are stable.

```text
sigma_obs   ~ HalfNormal(20)            # game-margin noise; measured residual sd is 16.4
sigma_conf  ~ HalfNormal(7)             # between-conference spread
tau_team    ~ HalfNormal(10)            # within-conference team spread  <- THE POOLING STRENGTH
hfa         ~ Normal(2.5, 3)            # points, estimated not assumed
rho         ~ Normal(0.5, 0.3)          # prior-season carryover, ESTIMATED (subsumes k=4)
c_j         ~ Normal(0, sigma_conf)                       # conference effect
theta_i     ~ Normal(c_{conf(i)} + rho * prev_i, tau_team)  # team strength
margin_g    ~ Normal(theta_home - theta_away + hfa*(1-neutral), sigma_obs)
```

`prev_i` is the **posterior mean of team i at the end of season s−1** (0 for a team with no
prior season). The chain runs forward only, so no future season can enter a prior.

Prediction for a week-W game: `E[theta_home] − E[theta_away] + E[hfa]·(1−neutral)`.

**Priors are a modelling decision and are stated here, before the fit** (per
`sdv-modeling/references/bayesian.md` §4). `HalfNormal` scales are set near the observed
spreads; no `Uniform(0, 100)` anywhere.

**Parameterization.** Fit centered AND non-centered (bayesian.md §3: centered is ~18×
more efficient when the data is informative, and only non-centered survives when `tau` is
small). **Both must clear every gate in §3; among those that pass, keep the higher ESS.**
Selecting on ESS alone would pick a more confident wrong answer.

**Cheap control (no sampler).** The same two-level model in closed form — method-of-moments
`tau`, James–Stein shrinkage `tau²/(tau²+se²)` — is scored as its own arm. If the closed
form matches the sampler, the sampler earned nothing and the report says so.

## 2. Pre-registered evaluation

* **Criterion:** held-out **MAE of predicted home margin**, walk-forward, as-of.
  Secondary (reported, not gating): Brier, calibration slope, max calibration error, corr.
* **Scoring set:** every arm is scored on the **identical** game set — the inner join of
  the harness game frame (`build_game_frame`, `require_rating=True`, `min_week=2`) across
  all arms. Market comparisons are additionally inner-joined to games with a usable
  closing line, and *every* arm is re-scored on that smaller set for the market row.
* **Seasons:** corpus **2014–2025**. 2014 is fit-only (it seeds the first carryover prior);
  **2015–2025 are the scored held-out seasons**. Every scored game is out-of-sample by
  construction: its prediction is made from games strictly earlier in time.
* **Arms:**
  | arm | what |
  |---|---|
  | `constant` | trivial baseline: the mean home margin of prior seasons |
  | `shipped` | **the pre-registered baseline** — the published closed-form surface, `backtest.shipped_predictor` (`net_points_scale`·(Δ`rt_adj_net`) + HFA) |
  | `hier-mcmc` | this model, NUTS |
  | `hier-eb` | this model, closed-form empirical Bayes |
  | `MARKET` | closing line, the ceiling — **never a feature** |
* **Handicap disclosed:** the shipped constants were fit on 2004–2025, so the baseline has
  seen every scored season. That advantages the baseline; the hierarchical arm has not.

### Where a hierarchical model could realistically land

Measured reference points, all from this program:

* market closing line: **MAE 12.03** (2021–2025, 3,004 games; slope 0.98, bias −0.03) — the ceiling
* best measured GBM on the 211-feature spine: **13.28** (provisional; measured pre-ridge-fix)
* closed-form refit + attenuation curve: **13.69**
* shipped surface: **14.79** (2018–2025) / **14.89** (2014–2025)
* constant home edge: **15.82**

A score-only hierarchical model sees **less** per-game information than the shipped surface
(final margins, not opponent-adjusted EPA) and far less than the GBM. **The honest band it
can occupy is ~13.3–14.5 MAE.** It is not expected to beat the 211-feature GBM and it
certainly cannot reach 12.03. What it can plausibly do is match-or-beat the *published*
surface while replacing four hand-set constants with one estimated pooling strength.

### The threshold that makes it a win — set now

Primary comparison: `hier` vs `shipped`, identical games, via
`cfb_higher_models.significance` (paired game-level **and** season-clustered).

| verdict | condition |
|---|---|
| **WIN — wire it as an engine** | ΔMAE ≤ **−0.15** AND season-clustered p < 0.05 |
| **SUGGESTIVE — report only, ship nothing** | ΔMAE ≤ −0.067 (the MDE) AND game-level p < 0.05, season-clustered p ≥ 0.05 |
| **NULL — the report is the deliverable** | anything else, including a win smaller than the 0.067 MDE |

The 0.067 MDE is measured, not assumed: two seeds of the same model, noise clustered
game-level (5,089 clusters, SE 0.0239) and season-level (9 clusters, SE 0.0237) both give
0.067 at 80% power. Week-level clustering would *raise* it to 0.141 and is excluded.

**No tuning happens before this readout.** If the primary comparison is NULL, that is the
result. Any post-hoc variant is reported separately and labelled post-hoc.

## 3. Sampler gates — read BEFORE any predictive number

A run that has not converged has no result to report. Per fit, per parameterization:

| diagnostic | threshold |
|---|---|
| `max r_hat` | **< 1.01** |
| `min ess_bulk` (pooled) | **> 400** |
| `min ess_tail` (pooled) | **> 400** |
| divergent transitions | **exactly 0** |

Enforcement: `assert_mcmc_healthy` **raises** (not `assert` — `python -O` strips those).
A failing as-of point is void. **If more than 1% of as-of points fail, the whole run is
void and no MAE is reported for it.** Diagnostics are printed for every fit and the
worst-case row across all fits is reported in §5 before any MAE appears.

## 4. Leakage and correctness checks (this ecosystem has been bitten by each)

1. **As-of boundary.** The fit input for `(s, W)` is asserted to satisfy
   `max(week) < W` and `season == s`; priors come only from seasons `< s`. Asserted in
   code, not by inspection.
2. **`through_week` is inclusive of week W** — verified empirically at 97.0% inclusive vs
   58.7% exclusive (`data.assert_asof_boundary`), so a week-W game must join
   `through_week == W − 1`. The harness's `build_game_frame` already enforces this and is
   reused rather than re-implemented; `assert_asof_boundary` runs on every build.
3. **Cumulative ops reset per group.** Games-played counts and carryover priors are
   computed per `(team, season)`; no cumulative quantity crosses a season boundary except
   the explicit `rho * prev_i` term, which is a *prior*, not a running sum.
4. **Silent no-op.** The fitted pooling must be shown to *do* something: assert the
   posterior spread of `theta` differs from the raw per-team margin spread, and that
   `tau_team`'s posterior is bounded away from both 0 and its prior scale. A component that
   ran without error has not been shown to have done anything.
5. **ID dtype discipline.** `team_id` pinned `Int64` at the boundary; join-key dtype
   agreement asserted before every join.
6. **Market is never an input.** Structurally separate module (`market.py`), used only to
   score.

## 5. Data vintage — stated, because the published assets are not current

Training and scoring read the **published** release assets as of **2026-09-02**:
`load_cfb_schedule`, `load_cfb_ratings_weekly`, `load_cfb_team_summaries_weekly`,
`load_cfb_betting_lines`.

**CFB model play-by-play was corrected several times this week** (rushing decomposition,
field-position flags, `text_dupe`); those fixes are on `sdv-py` `main` but **the published
assets have not been rebuilt**. So:

* the **hierarchical arm** consumes only final scores + schedule + conference, which no pbp
  parser fix can change — it is vintage-insensitive;
* the **`shipped` baseline arm** and the **rated-game scoring subset** are derived from
  `cfb_ratings_weekly` / `cfb_team_summaries_weekly` and **are** vintage-dependent. Numbers
  below are against the pre-rebuild vintage and must be re-read after the republish.

Do not treat the published asset as current.

---

<!-- RESULTS APPENDED BELOW ONLY AFTER THE GATES IN §3 ARE READ -->
