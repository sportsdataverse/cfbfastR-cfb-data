# CFB model registry

The authoritative list of every model and published artifact this repo trains.
Machine-checked by `tests/test_model_registry.py`, but know exactly what that
buys. It asserts that a numbered model stage is mentioned somewhere in this file,
and that a row citing an in-repo package has a stage exposing it. Two gaps it
does NOT catch:

- It matches on **package** names (`model_training`, `cpoe`), not per-model rows.
  Deleting the `ep` row alone still passes, because sibling rows mention
  `model_training` too.
- A row citing **no in-repo package** -- an sdv-py entry point, say -- is skipped
  entirely (`continue` at the `if not cited` branch), so it can lack a stage
  without failing.

Treat the test as a floor against wholly-undocumented stages, not as proof that
every row is complete or current.

Moved out of `CLAUDE.md` (2026-08-28): a table that a test parses is repository
data, not agent instructions, and it does not belong in an instructions file that
is read for guidance. It lives at the repo root under `models/` rather than
`docs/models/`, because `cfb_model_reports` regenerates that directory and
overwrites `README.md` on every run -- a hand-maintained file there would be
silently clobbered.

Rows are **mandatory for new published models/artifacts**. "frozen" is a valid
cadence, but it must be explicit.

Rows are **mandatory for new published models/artifacts**; "frozen" is a valid cadence but
must be explicit. Keeper `.ubj` files ship in sdv-py's bundled `cfb/models/`;
`.github/workflows/cfb_model_pipeline.yml` (annual cron Feb 5 + dispatch) retrains
EP/WP-spread/QBR/FD/CPOE on the full `cfbfastR-cfb-raw` finals corpus (2004–present,
~18.6k games), publishes what it trained to `espn_cfb_model_artifacts` +
`espn_cfb_model_pbp`, and regenerates `docs/models/` via `cfb_model_reports`. The 2026-06
era refresh (`docs/models/era_model_refresh.md`) promoted `qbr_era` / `fg_era` /
`wp_spread_backfilled` into the bundle.

**Card contract rebuild, 2026-09-09 (`model_version` 2026.08.27 -> 2026.09.09).** Every
model in the bundle now publishes a card. `cfb_cp_model` and `fd_model` had none at all;
the other seven were written 2026-08-02, before `_era_contract()` landed on 2026-09-07
(#71), so all seven published `era_contract: null`. Cards were regenerated from the
shipped boosters' own `feature_names` via `rebuild-cards` -- **no model was retrained**
and every `.ubj` checksum was verified byte-identical before upload. Five models carry a
contract (`fg`, `qbr`, `fd` one-hot; `two_pt`, `xpass` ordinal; cuts `[2006, 2013, 2020]`);
`ep`, `wp_naive`, `wp_spread` and `cfb_cp_model` correctly carry none, having no era
feature. Consumers can now read the era rule from the published card instead of keeping
private copies -- the duplication that caused #70.

| model | artifact(s) | release tag | training data (seasons/source) | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| ep (next-score EP, 7-class) | `ep_model.ubj` | `espn_cfb_model_artifacts` + sdv-py bundle | 2004–2025 finals corpus (~2.2M plays) | `model_training/train_ep.py` (`train-ep`) | LOSO EP cal-MAE 0.014 pts; era dummies rejected (cal regression) | 2026-06-17 | annual + dispatch |
| wp_spread | `wp_spread.ubj` (= promoted `wp_spread_backfilled`) | `espn_cfb_model_artifacts` + sdv-py bundle | 2004–2025 corpus + `cfb_line_odds` consensus-spread backfill (2,167 games) | `model_training/train_wp.py` (`train-wp --variant spread`); `spread_backfill.py` | LOSO AUC 0.916; backfill logloss 0.3616→0.3486 | 2026-06 (era refresh) | annual + dispatch |
| wp_naive | `wp_naive.ubj` | sdv-py bundle only (not in the cron's train steps) | 2004–2025 corpus, spread-free | `train-wp --variant naive` | corr-vs-spread 0.94 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| cp / CPOE (game state) | `cfb_cp_model.ubj` | `espn_cfb_model_artifacts` + sdv-py bundle | 2004+ pass plays with a known outcome, completions and incompletions alike (`completion` is the target, not a filter; window lowered from 2014, 2026-06) | `python -m cfb_model_build.cpoe --loso --variant game_state` | LOSO CV at train; measured on 2025-26 rows GroupKFold-by-game logloss 0.6687 / Brier 0.2380 | 2026-06-17 | annual + dispatch |
| cp / CPOE (air yards) | `cfb_cp_model_air_yards.ubj` | `espn_cfb_model_artifacts` + sdv-py bundle | 2025+ pass plays carrying ESPN air yards, both outcomes (27,673 in the 2026-09-07 fit over the `cfbfastR-cfb-raw` finals corpus, seasons 2025-2026; ESPN emits catch/target spots from 2025 only) | `python -m cfb_model_build.cpoe --loso --variant air_yards` | GroupKFold-by-game logloss 0.5395 / Brier 0.1806, vs the game-state model's 0.6687 / 0.2380 on the same 65,284-play corpus. (AIR_YARDS.md also quotes 0.5398 over 27,804 plays -- that is a SEPARATE confirmation run against the published `sportsdataverse-data` release parquet rather than the finals corpus, hence the slightly different population; the numbers here are the ones the fitting script produced.) | 2026-09-07 | annual + dispatch |
| qbr | `qbr_model.ubj` (= promoted `qbr_era`) | `espn_cfb_model_artifacts` + sdv-py bundle | 2004–2025 corpus + ESPN-QBR reference (`cfbfastR-cfb-raw` `scrape_cfb_qbr.py`) | `model_training/train_qbr.py` (`train-qbr`) | LOSO RMSE 17.294 / r² 0.612 (era beat baseline 17.604) | 2026-06 (era refresh) | annual + dispatch |
| fg | `fg_model.ubj` (= promoted `fg_era`) | sdv-py bundle (uploaded only when trained in a run) | 2004–2025 corpus placekicks | `model_training/train_fg.py` (`train-fg`) | LOSO logloss 0.5247 (era beat 0.5265); cal-err 0.008 | 2026-06 (era refresh) | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| fourth_down | `fd_model.ubj` (76-class yards distribution) | `espn_cfb_model_artifacts` + sdv-py bundle | full-corpus 4th-down plays | `model_training/fourth_down/` (`train-fd --validate`) | first-down cal-MAE 0.00272; era variant worse → not promoted | 2026-06-22 | annual + dispatch |
| two_pt | `two_pt_model.ubj` | sdv-py bundle (uploaded only when trained in a run) | corpus 2-pt attempts (ordinal era) | `model_training/train_two_pt.py` (`train-two-pt`) | LOSO cal-err 0.028 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| xpass | `xpass_model.ubj` | sdv-py bundle (uploaded only when trained in a run) | corpus pre-snap dropbacks | `model_training/train_xpass.py` (`train-xpass`) | LOSO cal-err 0.0073 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| punt distribution | `punt_distribution.parquet` | sdv-py bundle | 126k corpus punts | `python -m cfb_model_build.model_training.punt_distribution` | reproduces shipped artifact corr 0.9994 | 2026-06 | frozen (reproducible on demand, no cron) |
| rb_eval (xREPA) | GAM `.pkl` + card | `espn_cfb_model_artifacts` (per release body) | corpus rushing plays (pygam) | `python -m cfb_model_build.rb_eval` | TODO (no headline gate in `docs/models/index.qmd`) | 2026-06-18 | manual |
| pregame_wp (Five Factors) | `pgwp_model.ubj` (bundled at `python/pregame_wp/models/`) | `espn_cfb_model_artifacts` only when the opt-in `run_t4` step runs (needs `CFBD_API_KEY`) | CFBD box scores 2012–2020 **(UNRESOLVED 2026-09-07: `docs/models/pregame_wp.md` says the fit is over 2005–2025 / 37,774 team-games. `cli.py`'s `--seasons` DEFAULT is `2012:2020`, which is what this row records, while the committed box cache spans 2005–2025. Both statements are true about different things and neither establishes what the shipped `pgwp_model.ubj` was actually trained on — do not 'align' them by picking one; check the artifact.)** | `python -m cfb_model_build.pregame_wp build-boxes` + `train` | LOSO WP cal-err 0.0115; PtsDiff r² 0.535 | 2026-06-22 | manual (opt-in T4) |
| model_pbp (scored PBP) | `cfb_model_pbp_full.parquet` | `espn_cfb_model_pbp` | cfb-raw finals scored with the freshly trained cp model | `python -m cfb_model_build.cfb_model_pbp` | TODO (no documented publish gate; folded into `model_pbp` by `R/espn_cfb_16_model_pbp.R`) | TODO (last pipeline run not recorded here) | annual + dispatch |
| cfb_ratings | `cfb_ratings_{season}.parquet` + oracle card | `cfb_ratings` | released `espn_cfb_pbp` (2004+) | sdv-py `cfb_ratings()` via `cfb_model_publish ratings` (`cfb_ratings_cron.yml`) | refuses 0-row seasons; ridge refit per run; card written per publish | refit every run | daily in-season (13:00 UTC Aug–Jan), off-season idempotent newest-season refresh |
| cfb_recruiting_proj | per-season parquet + oracle card | `cfb_recruiting_proj` | roster features (247 talent, blue-chip ratio, returning production, prior wins) | sdv-py `cfb_recruiting_projection()` via `cfb_model_publish recruiting` (`cfb_recruiting_proj_cron.yml`) | refuses 0-row seasons; card written per publish | as-of ridge refit per run | monthly (5th, Dec + Jan–Aug) |
| matchup scoring_opp (drive scoring-opportunity glm) | `cfb_matchup/artifacts/scoring_opp_coef.json` (+ `_meta.json`) | bundled in-repo; consumed by `cfb_matchup_features` / `cfb_matchup_line` (stages 41/42) | 262,065 first-play-of-drive rows starting beyond the 40, the source pipeline's 2014–2023 drive frame (sha256 in `_meta.json`; frame regenerated by `ops/oneoff/20260915_matchup_oracle_capture/dump_training_frames.R`) | `python -m cfb_model_build.cfb_matchup train-scoring-opp --out-dir <candidate>` (stage 35; `--promote` to write the bundle) | parity: refit on the shipped rows reproduces every coefficient ≤ 1e-6 rel (observed 5.7e-12; integration); applier ≤ 1e-9 vs R `predict.glm` on 36,151 drives of 2025; level: a candidate must beat the intercept-only log-loss by ≥ 0.015 on the committed sample (observed 0.027) — `tests/cfb_matchup/test_glm_parity.py` | 2026-09-15 (imported R fit) | frozen (retrain on demand via stage 35 + `--force`) |
| matchup rush_expect (P(rush) glm) | `cfb_matchup/artifacts/rush_expect_coef.json` (+ `_meta.json`; `score_diff` aliased) | bundled in-repo; consumed by stages 41/42 (`rroe = rush − P(rush)`) | TODO — seasons not recorded by the source (1,304,773 rush/pass plays; sha256 in `_meta.json`, frame via `dump_training_frames.R`) | `python -m cfb_model_build.cfb_matchup train-rush-expect --out-dir <candidate>` (stage 35; `--promote` to write the bundle) | parity: refit reproduces every non-aliased coefficient ≤ 1e-6 rel (observed 1.8e-11; integration); applier ≤ 1e-9 vs R on 283,172 plays of 2025 and on the committed 5,000-play sample (unit); level: candidate beats intercept-only log-loss by ≥ 0.03 on the sample (observed 0.045) | 2026-09-15 (imported R fit) | frozen (retrain on demand; era drift in rush rate is a known caveat, see meta) |
| matchup wepa_weights (120 situational EPA weights) | `cfb_matchup/artifacts/wepa_weights.json` (+ `_meta.json`) | bundled in-repo; consumed by stages 41/42 (`apply_wepa`) | random search over 500 candidate weight vectors scored in-sample by `lm(home MOV ~ home/away off/def WEPA)` adj-R² on the source's 2014–2023 games | `python -m cfb_model_build.cfb_matchup train-wepa-weights --seasons 2014-2024 --holdout 2025 --n 500 --seed 4 --out-dir <candidate>` (stage 35; the source's random search, scored by adj-R² of home MOV on the four as-of WEPA means, with every candidate also scored on the held-out seasons) | applier: WEPA ≤ 1e-9 vs R on 283,172 plays of 2025; scorer: adj-R² and sigma ≤ 1e-6 vs the source's `evaluate_weights` on 2025 for five of its candidates + the shipped weights (`tests/cfb_matchup/test_wepa_search.py`, integration); the shipped weights are the argmax of the source's 500-candidate table (in-sample adj-R² 0.2509 on 2014–2023); **held-out 2025 adj-R² 0.2036** (1,459 games) — the first out-of-sample number these weights have had | 2024-06-02 (source fit) | frozen — the 2026-09-15 search on the 2014–2024 release corpus (500 candidates, seed 4, 2025 held out) peaked at held-out adj-R² 0.156 / in-sample 0.142, while the shipped weights score 0.204 held-out and 0.202 in-sample on that same corpus (`score-wepa --seasons 2014-2024`), so nothing was promoted; rerun with `train-wepa-weights` when the corpus grows |
| pregame closed form (sdv-py `cfb_game_predict`; GOP projection) | `models/pregame_fit.json` -> sdv-py `CFB_CONSTANTS["modern"]` (`net_points_scale`, `hfa_points`, `slope_by_games`, and `margin_sd` = the json's `margin_sd_curve`, the residual sd of the served curve; `adj_net_sd` -> `_FITTED_ADJ_NET_SD`) and GOP `astro/src/resources/sdv.ts` (`gop_net_adj_epa`: the flat form fitted on the summaries' `net_adj_epa`, the rating GOP feeds it, 1.37x narrower than `adj_net`) | none: the constants ship in code | as-of games (week W joined to `through_week` W-1), train 2014-2023 (5,673), near-holdout 2024-2025 (1,202; never in the fit, but sdv-py #598's adjusted-EPA shrinkage was tuned on 2023-2025); `cfb_team_summaries_weekly` assets of 2026-09-27 13:22-13:53Z (leak-free, #100, carries #598), `cfb_ratings_weekly` 2026-09-27 21:57-22:18Z (#105), `cfb_schedules` | `python -m cfb_model_build.cfb_higher_models fit-pregame --seasons 2014 ... 2025 --holdout 2024 2025` (stage 34), with a fresh `CFB_HM_CACHE` directory (or `.cache/higher_models/` deleted): the cache keys on the season range only, so after a republish it serves the old assets | sdv-py `tests/cfb/test_cfb_prediction_backtest.py` 2024 floors MAE <= 14.65, Brier <= 0.2298, acc >= 0.6089, spread agreement <= 6.41, calibration slope in [0.74, 1.12] (measured 13.35 / 0.2087 / 0.6481 / 5.79 / 0.926). Near-holdout, serving formula: MAE 13.32 vs the previous constants' 13.29 (dMAE +0.03 [-0.01, +0.06]), Brier 0.2039 vs 0.2040 (dBrier -0.0001 [-0.0008, +0.0005]): a tie, and the previous fit had seen 2024-25. GOP arm: MAE 13.00 vs 13.39 (dMAE -0.39 [-0.72, -0.08]), Brier 0.1967 vs 0.2056 (dBrier -0.0089 [-0.0139, -0.0039]), calibration slope 0.885 (about 10% steep). CIs: 2,000 game bootstraps, seed 0 | 2026-09-27 | after any republish of `cfb_team_summaries_weekly` or `cfb_ratings_weekly` (fit ~5 s on a warm cache, ~1 min cold), and each new season with the holdout rolled forward |

**2026-09-01 (deepdive PR #56).** `model_pbp` gained five additive athlete columns —
`passer_player_id`, `rusher_player_name`/`_id`, `receiver_player_name`/`_id` (Int64 ids pinned
at the boundary in `cfb_model_pbp/build.py`, null where ESPN tags no participant). Gate
`cfb_model_pbp/build.py::check_athlete_ids`: the newest season >= 2005 must carry an id on >= 0.9 of
named plays per role -- observed in pbp_full 2025 passer 0.979 / rusher 0.980 / receiver 0.972 (the
2-3% residue is regex-fallback names with no ESPN id); never lowered to pass. No gate changes; the
`espn_cfb_model_pbp` release picks them up on the next stage-10 + publish run, and sdv-py's
`load_cfb_model_pbp` returns-schema must be extended in the same step (its live test asserts
exact column equality against the published asset). `model_training export-analysis` writes
the per-model **analysis frames** `python/artifacts/analysis/analysis_{ep,wp,xpass,cp}.parquet`
+ `analysis_manifest.json` — play ids beside the exact trainer feature matrix — a build-tree
artifact (not published) consumed by `docs/models/deepdive.qmd`; the CI pipeline runs it
right after `ingest`.

**2026-09-02 — a defect in the SHIPPED pregame surface, found while measuring something else.**
`cfb_game_predict` / `cfb_ratings`' closed-form margin is **worse than a constant in weeks
2-4**: MAE **18.14** against the constant-home-edge baseline's **16.39**, measured
walk-forward as-of on 6,480 games (2015-2025) with `require_rating=True`. Cause is the flat
`net_points_scale` (44.54), which is roughly 4x too large in September when ratings rest on
one or two games — the errors-in-variables failure `slope_by_games` was hand-built to patch
and which the shipped constants do not use. It is well-behaved later (14.94 / 14.33 / 14.30
for weeks 5-8 / 9-12 / 13+). **A consumer reading an early-season `cfb_ratings` margin should
treat weeks 2-4 as unusable.** Not fixed here; recorded so it is visible at the artifact.
Evidence: `python -m cfb_model_build.cfb_higher_models hierarchical` prints the same table.

**2026-09-27 — the 2026-09-02 figures above and below were measured on LEAKED features.**
Every through-week `cfb_team_summaries_weekly` snapshot carried that season's bowl and CFP
games (#100). Re-measured on the leak-free republish with identical code, games and
`cfb_ratings_weekly` (the leaked arm reads the pre-#100 weekly assets from git `76f3b6b61`). Not a table:
the registry tests parse every pipe row in this file as a model row.

- hierarchical (`hier-eb`), walk-forward 2015-2025: 12.84 on 6,505 games leaked, 12.84 on 6,200 leak-free.
- `backtest.shipped_margin` on those games: 14.94 leaked, 14.76 leak-free.
- closing line on those games: 12.29 leaked, 12.30 leak-free.
- `experiments --quick` `gbm_all` (walk-forward, min_train 3): 12.51 on 5,408 leaked, 12.77 on 5,148 leak-free.
- `experiments --quick` `ridge_all`: 12.54 leaked, 13.10 leak-free.
- shipped constants through the serving formula, weeks 2-4: 16.42 on 689 leaked, 16.33 on 423 leak-free (constant baseline 16.36).

The feature models score 0.26-0.56 MAE worse on the leak-free vintage. That gap is leak
plus population (the leak made more week-2/3 games "rated", so the vintages score
different game sets) plus a redefinition: the leak-free assets also carry sdv-py #598's
adjusted EPA. The hierarchical model reads final scores only, so
its predictions do not move: it scores 12.84 on both sets, so for that model the
population change alone moves MAE by < 0.01. Its `assert_hierarchical_gate` thresholds all
pass as coded (beat-shipped 1.92, CI [1.68, 2.16]; 11/11 seasons), but that baseline is
`backtest.shipped_margin`, a formula no surface serves (flat scale, `2 * hfa_epa`). Against the
formula sdv-py serves (games-played curve + `hfa_points`) the hierarchical model's edge is
**1.21** MAE (14.05 vs 12.84; 11/11 seasons), which **fails the 1.85 gate**. The baseline fix
is a separate follow-up. The "worse than a constant
in weeks 2-4" defect was a property of `shipped_margin`'s flat scale: the served
games-played curve now ties the constant there. The "lean 60-feature GBM 13.06" was not
re-run. The pregame constants were refit from this data: see the "pregame closed form" row.

**2026-09-02 (hierarchical team strength) — no registry row, by the recorded rule.**
A two-level hierarchical team-strength model landed in stage 34
(`cfb_higher_models/hierarchical.py`, pre-registered at `PREREG_hierarchical.md`, gated by
`assert_hierarchical_gate`). Measured walk-forward as-of on 6,480 games (2015-2025): **12.83**
MAE vs the shipped surface's 14.97, the lean 60-feature GBM's 13.06, and the closing line's
12.27 (season-clustered p < 0.0001, 11/11 seasons). It gets **no row**, because
`NON_PUBLISHING_STAGES` in `tests/test_model_registry.py` says stage 34 "earns a registry row
the day one of its models is published, and not before" and this model publishes nothing —
no tag, no asset. It also has no stored fitted constants (`tau_team`, HFA and the carryover
`rho` are estimated per fit), so there is nothing here that could go stale the way
`_RIDGE_LAMBDA = 325` did. Full report:
`ClaudeCowork/ledgers/2026-09-02-next-ten/reports/cfb-hierarchical.md`.
