# CLAUDE.md — cfbfastR-cfb-data

R data repo. Reshapes per-game enriched `final` JSON from `cfbfastR-cfb-raw` into release
parquet/csv/rds. Sibling of `cfbfastR-cfb-raw` (Python/uv).

## Commands
- `Rscript -e 'testthat::test_dir("tests/testthat")'` — offline reshape tests (fixture-driven).
- `Rscript R/espn_cfb_0N_*.R -s YYYY -e YYYY` — build one dataset for a season range.
- `bash scripts/daily_cfb_R_processor.sh -s YYYY -e YYYY` — build all datasets.
- `Rscript R/releases_init.R` — one-time release-tag creation on both publish repos.

## Conventions
- **R reshape is reshape, not re-enrich.** The R pipeline (`R/espn_cfb_0N_*.R`) reads a
  `final` JSON block, rectangularizes, conforms, writes — play-by-play enrichment stays
  upstream in sdv-py. PBP conforms to cfbfastR's `.pbp_apply_output_schema()` when the
  installed cfbfastR exposes it (graceful pass-through otherwise). **Modeling now lives
  HERE** under `python/` (moved out of `-raw`, 2026-06-17) — see "Model training" below.
- Reshape functions are **pure + unit-tested** on `tests/testthat/fixtures/final_*.json`.
  Network is isolated to `fetch_*` in `R/_data_utils.R`.
- Bind with `data.table::rbindlist(fill = TRUE)`; select with `dplyr::any_of()`;
  `check.names = FALSE` to preserve dotted/slashed column names. JSON null -> NA.
- Publish dataset releases to `sportsdataverse/sportsdataverse-data` only (via `pb_upload_both`).
  Tags: `espn_cfb_*` (PBP = `espn_cfb_pbp`).
- Datasets NOT produced: `officials`, betting `propbets` (unavailable for CFB). `power_index`
  / `linescores` are recent-seasons-only.
- Commit message: `"CFB Data Updated (Start: YYYY End: YYYY)"`. Never add AI co-author trailers.

## Model training (Python, `python/`)

The native model suite moved here from `-raw` (2026-06-17). Run from `python/`:

| Package | Entry point | Dep group |
|---|---|---|
| `model_training` (EP/WP/QBR/FG/2pt/xpass + fourth-down, era models) | `python -m model_training` | — |
| `rb_eval` | `python -m rb_eval` | `gam` (pygam) |
| `pregame_wp` | `python -m pregame_wp` | `pregame-wp` (scipy/sklearn) |
| `cpoe` | `python -m cpoe` | — |

Cross-repo dependency: `.github/workflows/cfb_model_pipeline.yml` runs
`cfbfastR-cfb-raw`'s `python/scrape_cfb_qbr.py` (sparse checkout at `_raw`) for the
ESPN-QBR reference — renaming/moving that script in `-raw` breaks the QBR train step.

Supporting packages: `cfb_data_ingest`, `cfb_model_pbp`, `cfb_model_publish`,
`cfb_model_reports`. Figures: `uv sync --group figures` (plotnine). GAM tests (`rb_eval`):
`uv sync --group gam`; they skip cleanly otherwise. Integration checklist:
`python/model_training/HANDOFF.md`. `R/espn_cfb_16_model_pbp.R` folds model output into the
published `model_pbp` dataset.

## Model registry

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
| cp / CPOE | `cfb_cp_model.ubj` | `espn_cfb_model_artifacts` + sdv-py bundle | 2004+ completions (window lowered from 2014, 2026-06) | `python -m cpoe --loso` | LOSO CV at train (no headline gate published) | 2026-06-17 | annual + dispatch |
| qbr | `qbr_model.ubj` (= promoted `qbr_era`) | `espn_cfb_model_artifacts` + sdv-py bundle | 2004–2025 corpus + ESPN-QBR reference (`cfbfastR-cfb-raw` `scrape_cfb_qbr.py`) | `model_training/train_qbr.py` (`train-qbr`) | LOSO RMSE 17.294 / r² 0.612 (era beat baseline 17.604) | 2026-06 (era refresh) | annual + dispatch |
| fg | `fg_model.ubj` (= promoted `fg_era`) | sdv-py bundle (uploaded only when trained in a run) | 2004–2025 corpus placekicks | `model_training/train_fg.py` (`train-fg`) | LOSO logloss 0.5247 (era beat 0.5265); cal-err 0.008 | 2026-06 (era refresh) | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| fourth_down | `fd_model.ubj` (76-class yards distribution) | `espn_cfb_model_artifacts` + sdv-py bundle | full-corpus 4th-down plays | `model_training/fourth_down/` (`train-fd --validate`) | first-down cal-MAE 0.00272; era variant worse → not promoted | 2026-06-22 | annual + dispatch |
| two_pt | `two_pt_model.ubj` | sdv-py bundle (uploaded only when trained in a run) | corpus 2-pt attempts (ordinal era) | `model_training/train_two_pt.py` (`train-two-pt`) | LOSO cal-err 0.028 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| xpass | `xpass_model.ubj` | sdv-py bundle (uploaded only when trained in a run) | corpus pre-snap dropbacks | `model_training/train_xpass.py` (`train-xpass`) | LOSO cal-err 0.0073 | 2026-06-22 | on-demand: `cfb_model_pipeline.yml` with `train_extras=true` |
| punt distribution | `punt_distribution.parquet` | sdv-py bundle | 126k corpus punts | `python -m model_training.punt_distribution` | reproduces shipped artifact corr 0.9994 | 2026-06 | frozen (reproducible on demand, no cron) |
| rb_eval (xREPA) | GAM `.pkl` + card | `espn_cfb_model_artifacts` (per release body) | corpus rushing plays (pygam) | `python -m rb_eval` | TODO (no headline gate in `docs/models/index.qmd`) | 2026-06-18 | manual |
| pregame_wp (Five Factors) | `pgwp_model.ubj` (bundled at `python/pregame_wp/models/`) | `espn_cfb_model_artifacts` only when the opt-in `run_t4` step runs (needs `CFBD_API_KEY`) | CFBD box scores 2012–2020 | `python -m pregame_wp build-boxes` + `train` | LOSO WP cal-err 0.0115; PtsDiff r² 0.535 | 2026-06-22 | manual (opt-in T4) |
| model_pbp (scored PBP) | `cfb_model_pbp_full.parquet` | `espn_cfb_model_pbp` | cfb-raw finals scored with the freshly trained cp model | `python -m cfb_model_pbp` | TODO (no documented publish gate; folded into `model_pbp` by `R/espn_cfb_16_model_pbp.R`) | TODO (last pipeline run not recorded here) | annual + dispatch |
| cfb_ratings | `cfb_ratings_{season}.parquet` + oracle card | `cfb_ratings` | released `espn_cfb_pbp` (2004+) | sdv-py `cfb_ratings()` via `cfb_model_publish ratings` (`cfb_ratings_cron.yml`) | refuses 0-row seasons; ridge refit per run; card written per publish | refit every run | daily in-season (13:00 UTC Aug–Jan), off-season idempotent newest-season refresh |
| cfb_recruiting_proj | per-season parquet + oracle card | `cfb_recruiting_proj` | roster features (247 talent, blue-chip ratio, returning production, prior wins) | sdv-py `cfb_recruiting_projection()` via `cfb_model_publish recruiting` (`cfb_recruiting_proj_cron.yml`) | refuses 0-row seasons; card written per publish | as-of ridge refit per run | monthly (5th, Dec + Jan–Aug) |

## Inputs / outputs
- Input: `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/json/final/{id}.json`
  enumerated via `cfb_schedule_master.parquet` from the same repo.
- Output: `cfb/{dataset}/{parquet,rds,csv}/{stem}_{year}.*` + `cfb/{dataset}/cfb_{dataset}_in_data_repo.csv`.

## Reference
Data dictionary: `DATASETS.md`. Plan: `cfbfastR-cfb-raw/docs/superpowers/plans/2026-06-03-cfbfastR-cfb-data.md`.
