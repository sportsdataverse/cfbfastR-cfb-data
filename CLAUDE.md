# CLAUDE.md — cfbfastR-cfb-data

R data repo. Reshapes per-game enriched `final` JSON from `cfbfastR-cfb-raw` into release
parquet/csv/rds. Sibling of `cfbfastR-cfb-raw` (Python/uv).

## Commands
- `Rscript -e 'testthat::test_dir("tests/testthat")'` — offline reshape tests (fixture-driven).
- `Rscript R/espn_cfb_0N_*.R -s YYYY -e YYYY` — build one dataset for a season range.
- `bash scripts/daily_cfb_R_processor.sh -s YYYY -e YYYY` — build all datasets.
- `Rscript R/releases_init.R` — one-time release-tag creation on both publish repos.

## Conventions
- **Reshape, not re-enrich.** All modelling lives upstream in `-raw`/sdv-py. Here: read a
  `final` JSON block, rectangularize, conform, write. PBP conforms to cfbfastR's
  `.pbp_apply_output_schema()` when the installed cfbfastR exposes it (graceful pass-through
  otherwise).
- Reshape functions are **pure + unit-tested** on `tests/testthat/fixtures/final_*.json`.
  Network is isolated to `fetch_*` in `R/_data_utils.R`.
- Bind with `data.table::rbindlist(fill = TRUE)`; select with `dplyr::any_of()`;
  `check.names = FALSE` to preserve dotted/slashed column names. JSON null -> NA.
- Publish dataset releases to `sportsdataverse/sportsdataverse-data` only (via `pb_upload_both`).
  Tags: `espn_cfb_*` (PBP = `espn_cfb_pbp`).
- Datasets NOT produced: `officials`, betting `propbets` (unavailable for CFB). `power_index`
  / `linescores` are recent-seasons-only.
- Commit message: `"CFB Data Updated (Start: YYYY End: YYYY)"`. Never add AI co-author trailers.

## Inputs / outputs
- Input: `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/json/final/{id}.json`
  enumerated via `cfb_schedule_master.parquet` from the same repo.
- Output: `cfb/{dataset}/{parquet,rds,csv}/{stem}_{year}.*` + `cfb/{dataset}/cfb_{dataset}_in_data_repo.csv`.

## Reference
Data dictionary: `DATASETS.md`. Plan: `cfbfastR-cfb-raw/docs/superpowers/plans/2026-06-03-cfbfastR-cfb-data.md`.
