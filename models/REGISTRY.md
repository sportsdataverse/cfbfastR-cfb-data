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

| model | artifact(s) | release tag | training data (seasons/source) | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| ep (next-score EP, 7-class) | `ep_model.ubj` | `espn_cfb_model_artifacts` + sdv-py bundle | 2004–2025 finals corpus (~2.2M plays) | `model_training/train_ep.py` (`train-ep`) | LOSO EP cal-MAE 0.014 pts; era dummies rejected (cal regression) | 2026-06-17 | annual + dispatch |
| wp_spread | `wp_spread.ubj` (= promoted `wp_spread_backfilled`) | `espn_cfb_model_artifacts` + sdv-py bundle | 2004–2025 corpus + `cfb_line_odds` consensus-spread backfill (2,167 games) | `model_training/train_wp.py` (`train-wp --variant spread`); `spread_backfill.py` | LOSO AUC 0.916; backfill logloss 0.3616→0.3486 | 2026-06 (era refresh) | annual + dispatch |
| wp_naive | `wp_naive.ubj` | sdv-py bundle only (not in the cron's train steps) | 2004–2025 corpus, spread-free | `train-wp --variant naive` | corr-vs-spread 0.94 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| cp / CPOE | `cfb_cp_model.ubj` | `espn_cfb_model_artifacts` + sdv-py bundle | 2004+ completions (window lowered from 2014, 2026-06) | `python -m cfb_model_build.cpoe --loso` | LOSO CV at train (no headline gate published) | 2026-06-17 | annual + dispatch |
| qbr | `qbr_model.ubj` (= promoted `qbr_era`) | `espn_cfb_model_artifacts` + sdv-py bundle | 2004–2025 corpus + ESPN-QBR reference (`cfbfastR-cfb-raw` `scrape_cfb_qbr.py`) | `model_training/train_qbr.py` (`train-qbr`) | LOSO RMSE 17.294 / r² 0.612 (era beat baseline 17.604) | 2026-06 (era refresh) | annual + dispatch |
| fg | `fg_model.ubj` (= promoted `fg_era`) | sdv-py bundle (uploaded only when trained in a run) | 2004–2025 corpus placekicks | `model_training/train_fg.py` (`train-fg`) | LOSO logloss 0.5247 (era beat 0.5265); cal-err 0.008 | 2026-06 (era refresh) | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| fourth_down | `fd_model.ubj` (76-class yards distribution) | `espn_cfb_model_artifacts` + sdv-py bundle | full-corpus 4th-down plays | `model_training/fourth_down/` (`train-fd --validate`) | first-down cal-MAE 0.00272; era variant worse → not promoted | 2026-06-22 | annual + dispatch |
| two_pt | `two_pt_model.ubj` | sdv-py bundle (uploaded only when trained in a run) | corpus 2-pt attempts (ordinal era) | `model_training/train_two_pt.py` (`train-two-pt`) | LOSO cal-err 0.028 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| xpass | `xpass_model.ubj` | sdv-py bundle (uploaded only when trained in a run) | corpus pre-snap dropbacks | `model_training/train_xpass.py` (`train-xpass`) | LOSO cal-err 0.0073 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| punt distribution | `punt_distribution.parquet` | sdv-py bundle | 126k corpus punts | `python -m cfb_model_build.model_training.punt_distribution` | reproduces shipped artifact corr 0.9994 | 2026-06 | frozen (reproducible on demand, no cron) |
| rb_eval (xREPA) | GAM `.pkl` + card | `espn_cfb_model_artifacts` (per release body) | corpus rushing plays (pygam) | `python -m cfb_model_build.rb_eval` | TODO (no headline gate in `docs/models/index.qmd`) | 2026-06-18 | manual |
| pregame_wp (Five Factors) | `pgwp_model.ubj` (bundled at `python/pregame_wp/models/`) | `espn_cfb_model_artifacts` only when the opt-in `run_t4` step runs (needs `CFBD_API_KEY`) | CFBD box scores 2012–2020 | `python -m cfb_model_build.pregame_wp build-boxes` + `train` | LOSO WP cal-err 0.0115; PtsDiff r² 0.535 | 2026-06-22 | manual (opt-in T4) |
| model_pbp (scored PBP) | `cfb_model_pbp_full.parquet` | `espn_cfb_model_pbp` | cfb-raw finals scored with the freshly trained cp model | `python -m cfb_model_build.cfb_model_pbp` | TODO (no documented publish gate; folded into `model_pbp` by `R/espn_cfb_16_model_pbp.R`) | TODO (last pipeline run not recorded here) | annual + dispatch |
| cfb_ratings | `cfb_ratings_{season}.parquet` + oracle card | `cfb_ratings` | released `espn_cfb_pbp` (2004+) | sdv-py `cfb_ratings()` via `cfb_model_publish ratings` (`cfb_ratings_cron.yml`) | refuses 0-row seasons; ridge refit per run; card written per publish | refit every run | daily in-season (13:00 UTC Aug–Jan), off-season idempotent newest-season refresh |
| cfb_recruiting_proj | per-season parquet + oracle card | `cfb_recruiting_proj` | roster features (247 talent, blue-chip ratio, returning production, prior wins) | sdv-py `cfb_recruiting_projection()` via `cfb_model_publish recruiting` (`cfb_recruiting_proj_cron.yml`) | refuses 0-row seasons; card written per publish | as-of ridge refit per run | monthly (5th, Dec + Jan–Aug) |

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
