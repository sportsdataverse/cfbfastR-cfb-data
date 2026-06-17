# CFB Modeling Migration — SP1 Design

**Python modeling foundation in `cfbfastR-cfb-data` + URL-fetch ingest**

- **Date:** 2026-06-17
- **Status:** Approved design (pre-implementation); hardened by an 8-agent verify+critique pass on 2026-06-17.
- **Source repo:** `cfbfastR-cfb-raw` (scraping)
- **Destination repo:** `cfbfastR-cfb-data` (reshape + — after this work — modeling)
- **This spec covers:** Sub-project 1 (SP1). SP2/SP3 are roadmap context, out of scope for the SP1 plan.

---

## 1. Context & motivation

The SportsDataverse CFB pipeline follows the ecosystem's two-repo `-raw → -data` pattern:

- **`cfbfastR-cfb-raw`** scrapes ESPN, runs the Python enrichment pipeline (`CFBPlayProcess` → EPA/WPA/QBR), commits per-game raw + enriched JSON directly to git, and maintains `cfb/cfb_schedule_master.parquet`.
- **`cfbfastR-cfb-data`** is a pure-R reshape pipeline that reads the raw JSON **by URL** (`raw.githubusercontent.com/.../cfb/json/final/{game_id}.json`), enumerated from the schedule master, and builds + publishes 19 compiled datasets/season (parquet/rds/csv) to `sportsdataverse-data` GitHub Releases via `piggyback`. Triggered by cfb-raw's `repository_dispatch` on push.

A Python CFB modeling suite (EP/WP/QBR, CPOE, fourth-down, RB-eval) was built **inside `cfbfastR-cfb-raw`**, coupling model training to the scraper repo and reading inputs from the **local** committed JSON. Maintainer decision:

> Keep `cfbfastR-cfb-raw` centered on scraping raw JSON. Move all model training, training/final PBP dataset building, and the related tests/figures/model-cards into `cfbfastR-cfb-data`. The moved code should read the JSON **by URL** (via the raw repo's schedule file) to build the compiled rds/parquet dataset of all games' PBP and train the models.

## 2. Overall migration — three sub-projects

| Sub-project | Scope | Status |
|---|---|---|
| **SP1** (this spec) | Stand up a self-contained Python modeling subsystem in cfb-data; move the 4 packages + tests + fixtures; **unify all four onto managed URL/CFBD ingest**; train the models; **build the model-PBP dataset locally** (Phase 3); suite green offline. **Local outputs only.** | Designing now |
| **SP2** | **Publish** the model-PBP dataset (parquet via Python; rds/csv via the existing R writer for format parity) + model artifacts to `sportsdataverse-data` releases; generate figures/reports as orchestrated stages; wire `repository_dispatch`/cron. | Deferred |
| **SP3** | Decommission modeling from cfb-raw: remove the moved code + modeling-only deps; cfb-raw stays scraping-only and green. | Deferred |

**Build order:** SP1 → SP2 → SP3.

## 3. Locked decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Primary training data source | **Read raw final JSON by URL** for the final.json-sourced packages (`model_training`, `rb_eval`, and `cpoe` after rework) | Self-contained; mirrors the existing R `_data_utils.R` fetch; keeps model features under Python control. |
| D2 | What the Python side produces | **Train models AND BUILD a "model PBP" dataset locally** (feature cols + ep/wp/cpoe/epa/wpa predictions). **Publishing is deferred to SP2.** | The model-PBP is a distinctly-purposed artifact vs. the descriptive R `play_by_play`. SP1 builds it locally only. |
| D3 | Where the modeling lives | **cfb-data, new `python/` subsystem** with its own `pyproject.toml`, coexisting with the R pipeline | Mirrors cfb-raw's layout; precedent: `scripts/build_cfb_crosswalk.py`. |
| D4 | SP1 strategy | **Lift-and-shift, then re-point** (Phase 1 verbatim-green; Phase 2 re-point; Phase 3 model-PBP) | Separates "did the move break anything" from behavior change. |
| D5 | cfb-raw end state | **Scraping-only** (executed in SP3) | Removes modeling coupling from the scraper repo. |
| D6 | HTTP layer | **Reuse `sportsdataverse.dl_utils.download`** | `sportsdataverse` is **already a hard dependency** of `model_training` (`constants.py` imports `sportsdataverse.cfb.model_vars`), so this adds no new dependency weight. (Resolves former R3.) |
| D7 | Python layout | **Everything under `python/`**: `python/pyproject.toml`, `python/{packages}`, **`python/tests/`**, `python/tests/fixtures/` | cfb-data already has `tests/testthat/` (R). Repo-root `tests/` would collide with `R CMD check` and break pytest rootdir/pythonpath. Co-locating under `python/` fixes both. |
| D8 | Ingest unification (Fork A) | **Unify all four packages onto managed ingest** in SP1 | `model_training`/`rb_eval` read final.json by URL directly; **`cpoe` is reworked** to derive pass features from the final.json plays frame (drops its per-game-parquet read); **`pregame_wp`** keeps its CFBD season-stats source but under the managed/cached/gated framework (see §5.3). |
| D9 | Model-PBP build (Fork B) | **SP1 Phase 3** (net-new inference) | The model-PBP frame is net-new: load trained models → score → append predictions → EPA/WPA differencing. Its column schema is an SP1 **output contract** (consumed by SP2) and must be frozen before Phase 3 code. |

## 4. Verified grounding facts (8-agent audit, 2026-06-17)

- **Schedule master:** `cfb/cfb_schedule_master.parquet` — 18,619 games (2004–2025), 81 cols, **key `game_id` (Int32)**, **no URL column** (URLs constructed from `game_id`). Written by `scrape_cfb_schedules.py` (dedup on `game_id`).
- **RAW_BASE:** `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb` — the exact base R `_data_utils.R` uses: `final_url(game_id) → {RAW_BASE}/json/final/{game_id}.json`, `season_game_ids_from_master()` enumerates by season. Reuse verbatim.
- **Raw JSON layout (flat per-game):** `cfb/json/final/{game_id}.json`, `cfb/json/raw/{game_id}.json`; category dirs each have a `json/` subdir: `cfb/{betting,game_rosters,play_participants,power_index,team_box_extra}/json/{game_id}.json`.
- **Packages to move:** `model_training/` (incl. `fourth_down/`), `cpoe/`, `pregame_wp/`, `rb_eval/`. **No cfb-raw-only imports inside the package code** — BUT `model_training/constants.py` imports `from sportsdataverse.cfb import model_vars` (hard dep, see D6); `model_training/features.py` asserts equality against that shipped contract (a drift test).
- **`model_training.ingest` contract (preserve exactly):** `build_training_frame(final_dir, seasons=None) -> pl.DataFrame` → `_read_final_plays(final_dir, seasons)` (glob `*.json`, filter `obj.get("season")`, `pl.concat(how="diagonal_relaxed")`) → `clean_plays` → `_coerce_scoring_bools` → `label_next_score_half` → `add_weights`. Also `write_training_frame(final_dir, out_path, seasons=None) -> int` and `add_winner(df) -> df` (uses `homeScore/awayScore/homeTeamName/awayTeamName` per `game_id`). Dotted column surface is large (`start.down`, `type.text`, `start.pos_team.name`, EP_SOURCE/WP_SOURCE dotted cols, both `clock.minutes` and `clock_minutes`).
- **Per-package ingest sources (the key correction):** see §5.3.
- **pytest config (cfb-raw):** `[tool.pytest.ini_options] markers=["live: ..."], testpaths=["tests"], pythonpath=["python"]`. **No `integration` marker registered.** `tests/conftest.py` does `sys.path.insert`. **`tests/model_training/__init__.py` is MISSING** while sibling test dirs have one (asymmetry to fix).
- **Test coupling caveat:** `tests/model_training/test_qbr_scrape.py` does `sys.path.insert(..., parents[2]/"python"); from scrape_cfb_qbr import parse_qbr_payload` at import time — `scrape_cfb_qbr` is a **scraper that stays in cfb-raw**. This file must NOT be lifted verbatim (collection-time `ModuleNotFoundError`).
- **Fixtures to move:** `tests/fixtures/model_training/*` (incl. committed `.ubj` baselines `xgb_ep/wp_naive/wp_spread.ubj`, `fd_model.ubj`, `epa-/wpa-model-test-items.json`, `qbr_endpoint_sample.json`, `fd_fixture_plays.json`), `tests/fixtures/pregame_wp/{ep.csv,punt_sr.csv}`, `tests/fixtures/rb_eval/synth_plays.json`, `+ .gitkeep`. Fixture reads are `parent.parent[.parent]/"fixtures"` — they survive iff `fixtures/` stays a direct child of the tests root.
- **deps (cfb-raw `pyproject.toml`):** PEP 621 `[project]` + PEP 735 `[dependency-groups]` (no `[build-system]`; defaults to setuptools under uv). Core: `sportsdataverse>=0.0.60, pandas>=2.0, polars>=1.0, pyarrow>=15.0, requests>=2.28, tqdm>=4.66, scikit-learn>=1.0, xgboost>=2.0`. Groups: `dev=[pytest>=8.0,joblib>=1.3]`, `figures=[plotnine>=0.13,statsmodels>=0.14,pillow>=10.0]`, `gam=[pygam>=0.9]`, `pregame-wp=[scipy>=1.10,scikit-learn>=1.3]`.
- **cfb-data `.gitignore`:** a bare `cfb/` rule ignores the **entire** `cfb/` tree ("Built dataset artifacts live on release tags, not in git"). Anything placed under `cfb/` is already untracked.

## 5. SP1 design

### 5.1 Architecture — three phases

- **Phase 1 — lift-and-shift.** Copy the moved packages + test files (file-level, NOT verbatim dir — exclude `test_qbr_scrape.py`) + fixtures into `python/`; author `python/pyproject.toml` + `python/conftest.py`; get the suite green offline. Proves the move before behavior change.
- **Phase 2 — unify ingest (re-point).** Add `cfb_data_ingest` (URL fetch + cache + frame assembly). Re-point `model_training`/`rb_eval` to the final.json frame; **rework `cpoe`** to consume the same frame; bring **`pregame_wp`**'s CFBD ingest under the managed/cached/gated framework. Re-point = **populate a local cache dir; the existing `build_training_frame(final_dir=<cache>)` is called with an UNCHANGED signature** (directory swap, not an API refactor — preserves the D4 byte-identical invariant).
- **Phase 3 — model-PBP build (net-new).** Load trained models → score the plays frame → append `ep/wp/cpoe/epa/wpa` (incl. EP→EPA / WP→WPA differencing) → write the model-PBP parquet **locally**. Two-pass ordering: train → then score. Column schema frozen first (D9/R2).

### 5.2 Components

```
cfbfastR-cfb-data/
  python/
    pyproject.toml          # NEW — uv; PEP 621 + PEP 735 groups mirrored from cfb-raw
                            #   [tool.pytest.ini_options] pythonpath=["."] testpaths=["tests"]
                            #   markers=["integration: whole-corpus/network test (deselected by default)", "live: ..."]
                            #   addopts='-m "not integration"'   (default-deselect)
    conftest.py             # NEW — packages_dir / final_cache_dir fixtures (single source of path depth)
    model_training/         # MOVED (incl. fourth_down/)  + add missing __init__ where needed
    cpoe/                   # MOVED + reworked ingest (final.json frame)
    pregame_wp/             # MOVED (CFBD ingest under managed framework)
    rb_eval/                # MOVED
    cfb_data_ingest/        # NEW — schedule enumerate + URL fetch + cache + frame assembly + (Phase 3) scoring
    tests/                  # MOVED here (NOT repo-root) to avoid tests/testthat collision
      model_training/ cpoe/ pregame_wp/ rb_eval/
      fixtures/             # MOVED (model_training/, pregame_wp/, rb_eval/ subdirs + baselines)
  .cache/  (gitignored)     # ingest cache dir (final.json + schedule master); under python/ or cfb/.cache
  artifacts/ (gitignored)   # trained .ubj/.pkl + model_card.json + model-PBP parquet (SP1 local outputs)
```

- **`test_qbr_scrape.py` stays in cfb-raw** (it tests a scraper that stays). Move-set is enumerated per-file.
- **`.gitignore` (new entries):** the cache dir and artifacts dir must be explicitly ignored unless placed under the already-ignored `cfb/`. The plan pins exact paths + lines; outputs must NOT collide with the R writer's `cfb/{dataset}/` namespace.
- **conftest fixtures** replace the brittle `parents[N]` arithmetic (7 literals: 4 `FINAL` + 3 `sys.path/"python"`), defining path depth in one place.

### 5.3 Per-package ingest contracts (the corrected core)

| Package | Current source | SP1 treatment |
|---|---|---|
| `model_training` (EP/WP/QBR) | local `cfb/json/final/*.json` glob via `build_training_frame(final_dir)` | **Re-point**: `cfb_data_ingest` caches final.json by URL into `<cache>`; call `build_training_frame(final_dir=<cache>)` unchanged. |
| `rb_eval` | local `cfb/json/final` (`load_rush_plays(FINAL_DIR)`) | **Re-point**: same cached final.json frame. |
| `cpoe` | per-game `<season>/<game_id>/plays.parquet` (CFBPlayProcess output — NOT in raw repo) | **Rework**: change `extract_pass_features` to consume the final.json plays frame (the same enriched plays cpoe needs: completion, air-yards, down/distance/yardline). Plan must verify the CPOE feature columns exist in final.json plays. |
| `pregame_wp` (5-factor pregame WP) | live `api.collegefootballdata.com` (season team stats / talent), **requires `CFB_DATA_API_KEY`** | **Manage, don't RAW_BASE-repoint**: its inputs are season-level, not in final.json. Bring under the managed/cached framework; document `CFB_DATA_API_KEY` as a secret; **gate its network tests** (network/key gate, not just importorskip). **Plan should evaluate** sourcing five-factor inputs from the R pipeline's published `team_summaries` dataset by URL instead of CFBD (would complete the "unify onto URL" goal). |

### 5.4 Data flow

```
Phase 2 (ingest):
  cfb_schedule_master.parquet  ({RAW_BASE}/cfb_schedule_master.parquet, cached; --schedule local override)
        │  enumerate game_ids where season in {seasons}
        ▼
  GET {RAW_BASE}/json/final/{game_id}.json  (sportsdataverse.dl_utils.download; cache to <cache>; --refresh)
        ▼
  <cache>/  ──>  build_training_frame(final_dir=<cache>, seasons)   [UNCHANGED contract]
        ├──> model_training / rb_eval / cpoe(reworked)  → train → .ubj/.pkl + model_card.json (artifacts/)
        └──> pregame_wp: CFBD (or team_summaries) season inputs → train

Phase 3 (model-PBP, net-new):
  trained models + final.json plays frame
        ▼  load models → predict ep/wp/cpoe → difference EP→EPA, WP→WPA → append
  model-PBP parquet (artifacts/ or cfb/model_pbp/, local)   [publish = SP2]
```

### 5.5 Error handling

- **RAW_BASE (no auth):** missing/404 game → log + skip; emit fetched/skipped/total completeness count; reuse `dl_utils.download` retry/backoff. Reuse `_read_final_plays`' `diagonal_relaxed` concat + season filter so one bad game can't fail assembly.
- **CFBD (auth, pregame_wp only):** missing `CFB_DATA_API_KEY` → clear error; network failures retried; its tests gated so a base/CI install without the key SKIPS (never fails).
- **Cache:** keyed by `game_id`; `--refresh` re-fetches; corrupt cached JSON → re-fetch once then skip. Schedule master cached too (so offline `integration` runs work).

### 5.6 Testing

- Moved suites green offline with fixtures. **Collection-time guard audit:** confirm no moved *package* module imports `plotnine/pygam/statsmodels/pillow` at module scope (must be import-guarded or in-function), else the default `-m "not integration"` run breaks on a base install. `figures`/`gam` optional groups are **installed** for the green run (decision: yes — so `test_figures.py` runs; state this in acceptance).
- **Register the `integration` marker** in `python/pyproject.toml` and default-deselect via `addopts='-m "not integration"'`. **Sweep ALL corpus/network-gated tests** (every `FINAL`/`FINAL_DIR` + `glob('*.json')` skipif, plus pregame_wp CFBD tests) and mark them — NOT a hardcoded list.
- **Re-point `FINAL`** at the gitignored cache dir via a conftest fixture (not `parents[N]/cfb/json/final`, which is nonexistent in cfb-data). Re-derive every `parents[N]` and `parents[N]/"python"` literal for the `python/tests/` depth (or replace with conftest fixtures).
- **`cfb_data_ingest` hermetic tests:** 2–3 final-JSON fixtures + a tiny schedule-master fixture → URL-fetch + cache + frame-assembly tested with no live network. Add a test asserting the assembled frame carries the full required-column set (EP_SOURCE/WP_SOURCE/LBL/`add_winner`/qbr columns).
- Moved tests are tmp_path-hermetic for writes (verified) — no working-tree pollution.

## 6. Acceptance criteria (SP1)

1. `uv run pytest -m "not integration"` is **green** in `python/` (all moved suites incl. `test_figures.py` with `figures`/`gam` installed + new `cfb_data_ingest` tests).
2. `cfb_data_ingest` hermetic tests pass with no live network; the assembled frame asserts the required-column contract.
3. Every corpus/network-gated test (sweep-derived, not a fixed count) is marked `@pytest.mark.integration`, deselected by default, and **runnable offline** via `-m integration` after warming the cache (FINAL re-pointed at the cache dir).
4. A manual run over one small season: fetches final.json by URL → builds the frame → trains all four tracks (`.ubj`/`.pkl` + `model_card.json`) → **(Phase 3)** produces the model-PBP parquet locally with the frozen column schema.
5. **cfb-raw's code/contents are not modified by SP1** (SP1 only READS cfb-raw over URL; modeling code is removed only in SP3).
6. R toolchain untouched: `tests/testthat`, the R creation scripts, and `daily_cfb_R_processor.sh` still work; pytest does not collect R tests.

## 7. Non-goals (deferred)

- **Publishing** the model-PBP dataset or model artifacts to releases (SP2).
- Writing **rds/csv** (handed to the existing R writer for format parity — SP2).
- CI / `repository_dispatch` / cron orchestration (SP2).
- **Orchestrated** figure/report rendering as a pipeline stage (figure *code* + unit tests move in SP1; batch rendering = SP2).
- Removing modeling code + deps from cfb-raw (SP3).
- Reconciling `pregame_wp` onto a fully URL-sourced input (evaluated in SP1; if `team_summaries` doesn't carry the five factors, CFBD stays and full reconciliation defers).

## 8. Risks & open questions

- **R1 — `build_training_frame` contract (must freeze FIRST).** Plan task #1: read `model_training/ingest.py` + `features.py` and enumerate the full required dotted-column set; add a hermetic assertion in `cfb_data_ingest`.
- **R2 — model-PBP schema (must freeze before Phase 3).** The exact column set (features + which predictions + EPA/WPA differencing definitions) is an SP1 **output contract** consumed by SP2; freeze it early, not "in the plan later."
- **R3 — cpoe rework feasibility.** Confirm the final.json plays frame carries every CPOE feature `extract_pass_features` needs (completion, air-yards, etc.) before committing to the rework; fallback is keeping cpoe on local parquet (would partially undo D8).
- **R4 — pregame_wp source.** Decide CFBD-managed vs. `team_summaries`-by-URL; the latter completes "unify onto URL" but depends on `team_summaries` carrying the five-factor inputs.
- **R5 — layout/collection correctness.** `python/tests/` + `pythonpath=["."]` + conftest must make both relative (`from .ingest`) and absolute (`from model_training.model_card`) imports resolve; verify no duplicate-module errors; keep R `tests/testthat` out of pytest's path.
- **R6 — fetch cost.** Full-corpus = ~18.6K JSON GETs; cache makes re-runs cheap and lets gated `integration` tests run offline once warmed.

## 9. Decision log (this brainstorming session)

1. Training data source → **read raw final JSON by URL** (over consume-R-parquet / hybrid).
2. Python output → **train models AND build a "model PBP" dataset** locally; publish deferred to SP2 (over train-only / take-over-PBP).
3. SP1 strategy → **lift-and-shift, then re-point** (over dataset-first / combined).
4. Ingest scope (Fork A) → **unify all four** onto managed ingest in SP1 (over final.json-packages-only).
5. Model-PBP build (Fork B) → **SP1 Phase 3** net-new inference (over its own sub-project).
6. (Hardening) HTTP → reuse `sportsdataverse.dl_utils.download` (it's already a hard dep); layout → everything under `python/` incl. `python/tests/`.
