# CLAUDE.md — cfbfastR-cfb-data

R data repo. Reshapes per-game enriched `final` JSON from `cfbfastR-cfb-raw` into release
parquet/csv/rds. Sibling of `cfbfastR-cfb-raw` (Python/uv).

## Commands
- `Rscript -e 'testthat::test_dir("tests/testthat")'` — offline reshape tests (fixture-driven).
- `Rscript R/espn_cfb_0N_*.R -s YYYY -e YYYY` — build one dataset for a season range.
- `bash scripts/daily_cfb_R_processor.sh -s YYYY -e YYYY` — build all datasets.
- `Rscript R/releases_init.R` — one-time release-tag creation on both publish repos.
- `uv run python python/espn_injuries_daily_snapshot.py [-l nfl ...] [--publish]` — daily
  ESPN injuries snapshot, all 8 leagues (see "ESPN injuries" below). Build-only by default.

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
| `model_training` (EP/WP/QBR/FG/2pt/xpass + fourth-down, era models) | `python -m cfb_model_build.model_training` | — |
| `rb_eval` | `python -m cfb_model_build.rb_eval` | `gam` (pygam) |
| `pregame_wp` | `python -m cfb_model_build.pregame_wp` | `pregame-wp` (scipy/sklearn) |
| `cpoe` | `python -m cfb_model_build.cpoe` | — |

Cross-repo dependency: `.github/workflows/cfb_model_pipeline.yml` runs `cfbfastR-cfb-raw`'s
QBR scraper (sparse checkout at `_raw`) for the ESPN-QBR reference. The step tries
`python/espn_cfb_08_qbr_scrape.py` (the current numbered stage; the implementation lives in
`python/cfb_raw_scrape/scrape_cfb_qbr.py`), falling back to the two legacy names
(`espn_cfb_13_qbr_scrape.py`, top-level `scrape_cfb_qbr.py`) so any cfb-raw generation works.
Drop the fallbacks once cfb-raw's numbering is stable.

Supporting packages: `cfb_data_ingest`, `cfb_model_pbp`, `cfb_model_publish`,
`cfb_model_reports`. Figures: `uv sync --group figures` (plotnine). GAM tests (`rb_eval`):
`uv sync --group gam`; they skip cleanly otherwise. Integration checklist:
`python/model_training/HANDOFF.md`. `R/espn_cfb_16_model_pbp.R` folds model output into the
published `model_pbp` dataset.

## Model registry

The registry lives at [`models/REGISTRY.md`](models/REGISTRY.md) -- one row per
model/artifact, with training data, fitting script, gates at publish, last
retrain and cadence. Rows are **mandatory for new published models/artifacts**.
`tests/test_model_registry.py` is a floor, not a guarantee: it matches on PACKAGE
names rather than per-model rows, and skips a row citing no in-repo package (an
sdv-py entry point). It catches a wholly-undocumented stage; it does not prove
every row is complete. See the header of `models/REGISTRY.md`.

## ESPN injuries — daily append snapshot, all 8 leagues

`python/espn_injuries_daily_snapshot.py` is the ONLY route to injury data, and it is
deliberately cross-league even though this is the CFB producer.

- **The per-game stage cannot work.** `R/espn_cfb_14_injuries_creation.R` +
  `python/espn_cfb_14_injuries_creation.py` read the game-summary `injuries` key, which
  ESPN always ships as `[]` (verified across the 12 most-recent raw finals). They are
  structurally incapable of emitting rows; no rerun changes that. Left in place for the
  R/Python parity chain — do not "fix" or re-run them expecting output.
- **Where the data actually is:** `espn_<lg>_injuries()` is league-level and takes no
  arguments, so a full 8-league snapshot is ~8 requests, not a per-team fan-out.
- **Why here:** `espn_cfb_injuries` is the only injuries tag in the ecosystem, so injuries
  keep one owner rather than becoming eight near-identical stages across eight `-data`
  repos. This stage fills that tag and seven siblings.
- **APPEND, not overwrite.** The endpoint reports current state with no history — the
  `as_of_date` series *is* the dataset. Each run reads the prior release asset, drops any
  rows already stamped with today's date, and re-uploads the merged history, so re-running
  the same day replaces that day rather than duplicating it.
- **Never publish an empty snapshot.** A league returning zero athlete rows is logged and
  skipped; no zero-row asset is written. mbb/wbb hit this every offseason day, by design.
- Output: `espn_{league}_injuries` / `injuries_{season}.parquet`, one row per
  `(as_of_date, league, team, athlete, injury)`. `team_id` / `athlete_id` are pinned to
  `Int64` (ESPN ships every id as a numeric string, and omits `athlete.id` on this
  endpoint — it is recovered from the player links).
- Build-only by default; `--publish` uploads, `--dry-run` plans. Tests:
  `tests/test_espn_injuries_snapshot.py` (offline, no network).

## Inputs / outputs
- Input: `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/json/final/{id}.json`
  enumerated via `cfb_schedule_master.parquet` from the same repo.
- Output: `cfb/{dataset}/{parquet,rds,csv}/{stem}_{year}.*` + `cfb/{dataset}/cfb_{dataset}_in_data_repo.csv`.

## Reference
Data dictionary: `DATASETS.md`. Plan: `cfbfastR-cfb-raw/docs/superpowers/plans/2026-06-03-cfbfastR-cfb-data.md`.
