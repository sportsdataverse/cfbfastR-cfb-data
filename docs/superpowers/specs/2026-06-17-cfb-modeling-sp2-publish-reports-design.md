# CFB Modeling Migration — SP2 Design

**Publish (model-PBP dataset + model artifacts) + per-model Markdown reports**

- **Date:** 2026-06-17
- **Status:** Approved design (pre-implementation)
- **Repo:** `cfbfastR-cfb-data` (the modeling subsystem landed in SP1, PR #3, merge `0c1cd4a`)
- **Branch:** `feat/cfb-modeling-sp2-publish-reports`
- **This spec covers:** SP2 first slice. CI/orchestration is a deferred later slice; SP3 (decommission modeling from `cfbfastR-cfb-raw`) is separate.

---

## 1. Context & motivation

SP1 stood up the Python modeling subsystem under `python/`: it ingests `final.json` by URL, trains the model suite (EP / WP-spread / WP-naive / QBR / CPOE / fourth-down / RB-eval), and builds a model-PBP parquet — **all local outputs (gitignored)**. SP1 publishes nothing.

SP2 turns those local outputs into **published, browsable products**, reusing the existing R publish machinery for format parity:

- `R/_data_utils.R` exposes a generic, data-shape-agnostic publish path: `write_dataset(df, dataset, season, stem)` (`_data_utils.R:116-130`) serializes parquet/rds/csv with JSON-stringified list columns (`stringify_list_cols`, `:100-115`), and `publish_dataset(dataset, season, stem, tag)` (`:161-173`) uploads each format via `pb_upload_both` (`:132-141`, piggyback → `PUBLISH_REPOS = sportsdataverse/sportsdataverse-data`). Tags follow `espn_cfb_<dataset>`, registered in `R/releases_init.R`.
- The per-track Python `figures.py` modules already emit calibration / feature-importance PNG+CSV (cfbfastR garnet styling); `validate.py`/`loso.py` modules compute the metrics.

**Decision (maintainer):** SP2 = publish the model-PBP **dataset** + the trained model **artifacts** to releases, and generate nflfastR/cfbfastR-style **per-model Markdown reports** committed to the repo. Run-locally-then-publish; CI orchestration deferred.

## 2. Locked decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | What SP2 publishes | **model-PBP dataset + all model artifacts + per-model Markdown reports** | The full "make the SP1 outputs consumable" slice. |
| D2 | Report format + home | **Markdown committed under `docs/models/`** (metrics tables + embedded figure PNGs), GitHub-rendered + versioned | Mirrors nflfastR's in-repo model docs; lightweight (no Quarto/RMarkdown toolchain). |
| D3 | Model artifacts scope + tag | **All suite models** (EP, WP-spread, WP-naive, QBR, CPOE, fourth-down `.ubj` + RB-eval `.pkl`) + each `model_card.json`, under **one** `espn_cfb_model_artifacts` tag | One release to register/consume; cards carry per-model provenance. pregame-wp included only if it trained in the run (analytic artifact). |
| D4 | Python/R split | **Python owns modeling/reports/artifacts; R for dataset parity** | Metrics/figures/cards are all Python; R's value-add is the rds/csv stringify + piggyback. Each language does what it already does best. |
| D5 | Model-PBP dataset publish | **Thin R wrapper** reads the Python model-PBP parquet → `write_dataset` → `publish_dataset(...,"espn_cfb_model_pbp")` | rds/csv/parquet parity with every other cfb-data dataset, via the proven path. |
| D6 | Run mode | **Run-locally-then-publish** (consumes SP1's `python/artifacts/` + warmed cache); **CI deferred** | Smallest shippable SP2; the heavy ingest/train already runs locally in SP1. |

## 3. Verified grounding facts (from SP2 exploration)

- **Generic R publish entry points** (reusable as-is): `write_dataset(df, dataset, season, stem)` `_data_utils.R:116-130`; `publish_dataset(dataset, season, stem, tag)` `:161-173`; `pb_upload_both(file, tag, repos=PUBLISH_REPOS)` `:132-141`; `stringify_list_cols` `:100-115`. `PUBLISH_REPOS = c("sportsdataverse/sportsdataverse-data")`. Tags registered in `R/releases_init.R:8-34` (25 `espn_cfb_*` tags; idempotent `pb_release_create`).
- **No model artifacts are published today**; no model-report generation exists today (only figure PNG/CSV emitters).
- **Figures (Python):** `model_training/figures.py::write_calibration(table, stem, title, subtitle, cal_error) -> (png, csv)`; `fourth_down/figures.py::write_fd_figures(...)`; `rb_eval/figures.py::write_xrepa_calibration(...)`; `cpoe/figures.py` plot helpers. cfbfastR garnet `#500f1b` styling. (`figures`/`gam` optional dependency groups.)
- **Metrics (Python):** each track's `validate.py` (+ `cpoe/loso.py`, `rb_eval` `loso_cv`) computes the per-model metrics (calibration error, log-loss/Brier, weighted R², QBR correlation). SP2 reports run these.
- **CI today:** `daily_cfb.yml` (cron + `repository_dispatch: daily_cfb_data` from cfb-raw, parses `Start:/End:` from the commit message) → `scripts/daily_cfb_R_processor.sh`. No Python in CI. (SP2 does NOT touch CI — deferred.)

## 4. Components

```text
cfbfastR-cfb-data/
  python/
    cfb_model_reports/            # NEW — metrics + figures + Markdown report assembly
      __init__.py
      metrics.py                  # per-track validate/loso runners -> metric dicts (one fn per model family)
      report.py                   # assemble per-model Markdown (tables + embedded figure links + card provenance)
      cli.py, __main__.py         # `python -m cfb_model_reports --artifacts <dir> --cache <dir> --out docs/models`
    cfb_model_publish/            # NEW — publish model artifacts
      __init__.py
      artifacts.py                # gh-release upload of *.ubj/.pkl + cards -> espn_cfb_model_artifacts (supports --dry-run)
      cli.py, __main__.py         # `python -m cfb_model_publish artifacts --artifacts <dir> [--dry-run]`
    tests/cfb_model_reports/      # NEW — hermetic report-assembly tests (synthetic metrics + tiny figures)
    tests/cfb_model_publish/      # NEW — --dry-run upload tests
  R/
    espn_cfb_16_model_pbp.R       # NEW — read python model-PBP parquet -> write_dataset -> publish_dataset(espn_cfb_model_pbp)
    releases_init.R               # EDIT — register espn_cfb_model_pbp + espn_cfb_model_artifacts
  docs/models/                    # NEW (committed) — per-model reports + figures
    README.md                     # index linking all model reports
    {ep,wp,qbr,cpoe,fourth_down,rb_eval}.md
    figures/                      # committed calibration/importance PNG (+ CSV sidecars)
```

- **`cfb_model_reports`** is the bulk of SP2's net-new code: pure Python that reads the trained models + validation data, computes metrics, renders figures, and writes Markdown. No publishing — its output (docs/models/) is committed.
- **`cfb_model_publish`** is a thin `gh release upload` wrapper for the artifact files (no rds/csv needed — they're opaque binaries + JSON cards). `--dry-run` prints the file set + tag without uploading.
- **`espn_cfb_16_model_pbp.R`** is the only R addition: it reads the Python-produced model-PBP parquet into a frame and runs it through the existing `write_dataset`/`publish_dataset` so the published dataset has parquet/rds/csv exactly like every other cfb-data dataset.

## 5. Data flow

```text
SP1 local outputs:  python/artifacts/{ep.ubj,wp_spread.ubj,wp_naive.ubj,qbr.ubj,cp.ubj,fd.ubj,xrepa.pkl,*.model_card.json}
                    python/artifacts/model_pbp_<season>.parquet ;  python/.cache/cfb_final/ (warmed)
        │
        ├── REPORTS (committed):
        │     for model in [ep, wp(spread,naive), qbr, cpoe, fourth_down, rb_eval]:
        │       metrics = cfb_model_reports.metrics.<model>(model, validation_data)   # runs validate/loso
        │       figs    = <track>.figures.*(...)  -> docs/models/figures/<model>_*.png (+ .csv)
        │       report.assemble(model, metrics, figs, card) -> docs/models/<model>.md
        │     report.index(...) -> docs/models/README.md
        │
        ├── DATASET (release):  artifacts/model_pbp_<season>.parquet
        │     -> R espn_cfb_16_model_pbp.R: read parquet -> write_dataset(df,"model_pbp",season,"model_pbp")
        │     -> publish_dataset("model_pbp",season,"model_pbp","espn_cfb_model_pbp")  (sportsdataverse-data)
        │
        └── ARTIFACTS (release):  *.ubj/.pkl + *.model_card.json
              -> cfb_model_publish.artifacts: gh release upload -> espn_cfb_model_artifacts  (--dry-run capable)
```

**Per-model report contents** (D2): a metrics table + embedded calibration/importance figures + provenance (from `model_card.json`: features, hyperparameters, training seasons, trained date). Metrics by family:
- **EP / WP-spread / WP-naive / CPOE:** log-loss + Brier + weighted calibration error + feature importance.
- **fourth-down:** first-down calibration + feature importance (76-class yards model).
- **QBR:** correlation + RMSE vs ESPN QBR.
- **RB-eval (xREPA):** LOSO weighted R² + weighted calibration error.

## 6. Error handling

- **Missing artifact** (a model not present in `python/artifacts/`) → skip that report + that upload, log a clear `skipped: <model> (artifact absent)` line; never abort the batch. Emit a completeness summary (reported / skipped).
- **Publish auth:** `gh release upload` uses the ambient `gh` auth / `GITHUB_PAT`; the R `publish_dataset` reuses `pb_upload_both` (idempotent `overwrite=TRUE`). The artifact publisher supports `--dry-run` (prints the planned uploads, touches nothing).
- **Insufficient validation data** for a metric → omit that metric from the report with an explicit `n/a (insufficient data)` note rather than crashing.
- **Figure-dep absence** (`figures`/`gam` groups not installed) → report generation degrades to metrics-only with a note (figures import-guarded).

## 7. Testing

- **Report assembly (`cfb_model_reports`)** — hermetic: feed synthetic metric dicts + tiny fixture figure paths; assert the rendered Markdown contains the metrics table rows, the embedded figure links, and the provenance block. No real models, no network.
- **Metrics runners** — unit-test each `metrics.<model>` on a tiny synthetic validation frame (reuses the per-track `validate` functions, which already have their own tests) — assert the metric keys + plausible ranges; mark `integration` only if a real model/corpus is required.
- **Artifact publisher** — test `--dry-run`: assert the exact file set + tag it *would* upload from a synthetic `artifacts/` dir. The live `gh release upload` and the R `publish_dataset` are `integration`-marked (need auth + a real release); not in the default suite.
- **R `espn_cfb_16_model_pbp.R`** — covered by the existing `tests/testthat` reshape-fixture pattern if practical; otherwise a smoke check that it reads a fixture parquet and calls `write_dataset` (the publish call gated behind an env flag).
- Acceptance: `uv run pytest -m "not integration"` green incl. the new report/publish tests; a manual `python -m cfb_model_reports` over SP1's local artifacts produces `docs/models/*.md` + figures; `python -m cfb_model_publish artifacts --dry-run` lists the correct files/tag.

## 8. Non-goals (deferred)

- **CI orchestration** — the automated `repository_dispatch`/cron workflow that runs ingest → train → build model-PBP → report → publish on a runner. Its own later slice (with runner/secret setup).
- **Live publish in CI** — SP2 publishes from a local run; wiring publish into an automated workflow is the CI slice.
- **R `presentation/`-style Bluesky posting** of model figures — out of scope.
- **SP3** — decommissioning the modeling code from `cfbfastR-cfb-raw`.

## 9. Risks & open questions

- **R1 — validation data source for reports.** Reports need a validation pass. Decide in the plan: re-run each track's `loso`/`validate` on the warmed cache / model-PBP frame (accurate, slower) vs. read a metrics sidecar the train step could emit. Recommend running `validate`/`loso` at report time over the cache (no new train-time contract).
- **R2 — exact artifact filenames.** SP1's CLIs name the saved models (`ep.ubj`, `cp.ubj`, etc.); the report + publisher must agree on the artifact directory layout + filenames. Pin the exact `python/artifacts/` layout in the plan (read SP1's CLIs).
- **R3 — QBR metric.** Confirm `model_training`'s QBR `validate` exposes correlation/RMSE vs ESPN QBR (it may need the ESPN QBR frame at report time). If unavailable offline, mark the QBR report metric `integration`.
- **R4 — figure embedding paths.** Markdown embeds `figures/<model>_*.png` with repo-relative links so they render on GitHub; confirm the per-track `figures.py` can write into `docs/models/figures/` (they take an out-dir/stem).

## 10. Decision log

1. SP2 scope → **publish capability + per-model Markdown reports** (over dataset-only / publish-only / +full-CI).
2. Report format → **Markdown committed under `docs/models/`** (over HTML/Quarto / release-asset).
3. Artifacts → **all suite models under one `espn_cfb_model_artifacts` tag** (over XGBoost-only / per-track tags).
4. Py/R split → **Python owns modeling/reports/artifacts; R for dataset parity** (over R-orchestrated / Python-only-no-parity).
