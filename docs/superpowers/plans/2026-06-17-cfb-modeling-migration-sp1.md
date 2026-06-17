# CFB Modeling Migration — SP1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a self-contained Python modeling subsystem in `cfbfastR-cfb-data`, move the four model packages from `cfbfastR-cfb-raw`, unify their ingest onto a URL-fetched final.json cache, train the models, and build the model-PBP dataset locally — all green offline.

**Architecture:** A new `python/` tree in cfb-data (own `pyproject.toml`, `python/tests/`) holds `model_training` / `cpoe` / `pregame_wp` / `rb_eval` moved verbatim (Phase 1). A new `cfb_data_ingest` package fetches `raw.githubusercontent.com/.../json/final/{game_id}.json` (enumerated from `cfb_schedule_master.parquet`) into a local cache dir; the packages' existing `final_dir` arguments are pointed at that cache — a directory swap, not an API change (Phase 2). A net-new `cfb_model_pbp` builder carries EP/WP/EPA/WPA from final.json and scores net-new CPOE to produce the model-PBP parquet (Phase 3). The R reshape pipeline is untouched.

**Tech Stack:** Python ≥3.11, uv (PEP 621 + PEP 735), polars 1.x, pandas, xgboost, pygam (optional `gam`), plotnine/statsmodels/pillow (optional `figures`), scipy, scikit-learn, `sportsdataverse>=0.0.60`, pytest.

## Global Constraints

- **Python ≥3.11; polars 1.x** API only (no 0.18-era calls).
- **`sportsdataverse>=0.0.60` is a HARD runtime dep** (`model_training/constants.py` imports `sportsdataverse.cfb.model_vars`; `test_constants.py` asserts a drift gate against it).
- **Everything Python under `python/`**; tests under `python/tests/` (never repo-root `tests/`, which is the R `testthat` tree).
- **`integration` marker is registered and default-deselected** via `addopts='-m "not integration"'`.
- **cfb-raw is NOT modified by SP1** (read-only over URL; decommission = SP3).
- **R toolchain untouched:** `tests/testthat`, `R/`, `daily_cfb_R_processor.sh` must keep working; pytest must not collect R files.
- **All local outputs gitignored** (cache, `.ubj`/`.pkl`, `model_card.json`, model-PBP parquet). No publishing in SP1.
- **No AI co-author trailers** on any commit (SportsDataverse rule). Conventional Commits.
- **`RAW_BASE = https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb`** — reuse verbatim (matches R `_data_utils.R`).
- **Per-game final.json shape:** top-level dict `{ "season": int, "plays": [ {dotted-key play}, ... ], ... }`; some games have empty `plays` (skip).

---

## File Structure

**New (created in cfb-data):**
- `python/pyproject.toml` — packaging + pytest config (mirrors cfb-raw + `integration` marker + `addopts`).
- `python/conftest.py` — repo-level pytest bootstrap (pythonpath).
- `python/tests/conftest.py` — `packages_dir` + `final_cache_dir` fixtures (single source of path depth).
- `python/cfb_data_ingest/__init__.py`, `schedule.py`, `fetch.py`, `cli.py`, `__main__.py` — URL fetch + cache + enumeration.
- `python/cfb_model_pbp/__init__.py`, `schema.py`, `build.py`, `cli.py`, `__main__.py` — Phase 3 model-PBP builder.
- `python/tests/cfb_data_ingest/`, `python/tests/cfb_model_pbp/` — hermetic tests.
- `.gitignore` additions for Python artifacts.

**Moved verbatim (from cfb-raw `python/` → cfb-data `python/`):** `model_training/` (incl. `fourth_down/`), `cpoe/`, `pregame_wp/`, `rb_eval/`, and their `tests/<pkg>/` dirs + `tests/fixtures/{model_training,pregame_wp,rb_eval}/`.

**Moved-and-reworked:** `model_training/ingest.py` + `model_training/cli.py` + `model_training/fourth_down/cli.py` (re-point `final_dir`); `cpoe/ingest.py` + `cpoe/features.py` + `cpoe/constants.py` (final.json rework); `rb_eval/cli.py` (`--final-dir` default); several test files (path literals + `integration` marks).

**Excluded (stays in cfb-raw):** `tests/model_training/test_qbr_scrape.py` + `tests/fixtures/model_training/qbr_endpoint_sample.json` (they exercise `scrape_cfb_qbr`, a scraper that stays).

---

# PHASE 1 — Lift-and-shift + scaffold + green offline

### Task 1: Create `python/pyproject.toml`

**Files:**
- Create: `python/pyproject.toml`

**Interfaces:**
- Produces: the uv project + pytest config the whole subsystem builds on (`pythonpath=["."]`, `testpaths=["tests"]`, `markers` incl. `integration`, `addopts='-m "not integration"'`).

- [ ] **Step 1: Write the file**

```toml
[project]
name = "cfbfastr-cfb-data-models"
version = "0.1.0"
description = "CFB modeling subsystem (EP/WP/QBR, CPOE, fourth-down, RB-eval) + URL-fetch ingest + model-PBP build."
requires-python = ">=3.11"
dependencies = [
    "sportsdataverse>=0.0.60",
    "pandas>=2.0",
    "polars>=1.0",
    "pyarrow>=15.0",
    "requests>=2.28",
    "tqdm>=4.66",
    "scikit-learn>=1.0",
    "xgboost>=2.0",
]

[dependency-groups]
dev = ["pytest>=8.0", "joblib>=1.3"]
figures = ["plotnine>=0.13", "statsmodels>=0.14", "pillow>=10.0"]
gam = ["pygam>=0.9"]
pregame-wp = ["scipy>=1.10", "scikit-learn>=1.3"]

[tool.pytest.ini_options]
markers = [
    "integration: whole-corpus or network test requiring fetched/cached JSON or a CFBD key (deselected by default)",
    "live: hits the live ESPN API (gated by CFB_LIVE_TESTS=1)",
]
testpaths = ["tests"]
pythonpath = ["."]
addopts = '-m "not integration"'
```

- [ ] **Step 2: Verify it parses**

Run: `cd python && uv run python -c "import tomllib,pathlib;tomllib.loads(pathlib.Path('pyproject.toml').read_text());print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add python/pyproject.toml
git commit -m "feat(python): scaffold cfb-data modeling subsystem pyproject"
```

---

### Task 2: Conftest fixtures + Python `.gitignore`

**Files:**
- Create: `python/conftest.py`
- Create: `python/tests/conftest.py`
- Create: `python/tests/__init__.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `packages_dir` fixture (→ `python/`) and `final_cache_dir` fixture (→ gitignored cache, env-overridable via `CFB_FINAL_CACHE`) consumed by every relocated path-dependent test.

- [ ] **Step 1: Write `python/conftest.py`** (empty marker file so pytest rootdir is `python/`)

```python
# python/conftest.py — presence pins pytest rootdir to python/ (with pyproject pythonpath=["."])
```

- [ ] **Step 2: Write `python/tests/conftest.py`**

```python
import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def packages_dir() -> pathlib.Path:
    # python/tests/conftest.py -> parents[1] == python/ (the packages root)
    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def final_cache_dir() -> pathlib.Path:
    # Where cfb_data_ingest caches final.json. Gitignored. Overridable for integration runs.
    env = os.environ.get("CFB_FINAL_CACHE")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[1] / ".cache" / "cfb_final"
```

- [ ] **Step 3: Write `python/tests/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Append Python ignores to `.gitignore`**

Add these lines to the existing `.gitignore` (after the `cfb/` line):

```gitignore
# Python modeling subsystem (python/)
__pycache__/
*.pyc
.pytest_cache/
python/.venv/
python/.cache/
python/artifacts/
```

- [ ] **Step 5: Commit**

```bash
git add python/conftest.py python/tests/conftest.py python/tests/__init__.py .gitignore
git commit -m "feat(python): conftest path fixtures + python gitignore entries"
```

---

### Task 3: Move the four packages verbatim

**Files:**
- Create (copy from cfb-raw): `python/model_training/` (incl. `fourth_down/`), `python/cpoe/`, `python/pregame_wp/`, `python/rb_eval/`

**Interfaces:**
- Consumes: nothing. Produces: importable `model_training`, `cpoe`, `pregame_wp`, `rb_eval` packages under `python/`.

- [ ] **Step 1: Copy the package source** (cfb-raw is the source of truth; do NOT modify cfb-raw)

```bash
SRC="/c/Users/saiem/Documents/GitHub-Data/sdv-dev/cfbfastR-dev/cfbfastR-cfb-raw/python"
DST="python"
cp -r "$SRC/model_training" "$DST/model_training"
cp -r "$SRC/cpoe"          "$DST/cpoe"
cp -r "$SRC/pregame_wp"    "$DST/pregame_wp"
cp -r "$SRC/rb_eval"       "$DST/rb_eval"
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} +
```

- [ ] **Step 2: Verify each package imports** (with deps synced)

Run:
```bash
cd python && uv sync --all-groups && \
uv run python -c "import model_training, cpoe, pregame_wp, rb_eval; print('import ok')"
```
Expected: `import ok` (sportsdataverse resolves model_training/constants.py's `model_vars` import).

- [ ] **Step 3: Commit**

```bash
git add python/model_training python/cpoe python/pregame_wp python/rb_eval
git commit -m "feat(python): move EP/WP/QBR, cpoe, pregame_wp, rb_eval packages into cfb-data"
```

---

### Task 4: Move the test dirs + fixtures (excluding the QBR-scrape test)

**Files:**
- Create (copy): `python/tests/model_training/` (excl. `test_qbr_scrape.py`), `python/tests/cpoe/`, `python/tests/pregame_wp/`, `python/tests/rb_eval/`
- Create (copy): `python/tests/fixtures/{model_training,pregame_wp,rb_eval}/` (excl. `model_training/qbr_endpoint_sample.json`)
- Create: `python/tests/model_training/__init__.py` (was missing in cfb-raw)

**Interfaces:**
- Consumes: the moved packages (Task 3). Produces: the relocated test suite.

- [ ] **Step 1: Copy test dirs, excluding the QBR-scrape test + its fixture**

```bash
SRC="/c/Users/saiem/Documents/GitHub-Data/sdv-dev/cfbfastR-dev/cfbfastR-cfb-raw/tests"
DST="python/tests"
cp -r "$SRC/model_training" "$DST/model_training"
cp -r "$SRC/cpoe"           "$DST/cpoe"
cp -r "$SRC/pregame_wp"     "$DST/pregame_wp"
cp -r "$SRC/rb_eval"        "$DST/rb_eval"
cp -r "$SRC/fixtures"       "$DST/fixtures"
rm -f "$DST/model_training/test_qbr_scrape.py" "$DST/fixtures/model_training/qbr_endpoint_sample.json"
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} +
```

- [ ] **Step 2: Add the missing `__init__.py`**

```bash
touch python/tests/model_training/__init__.py
```

- [ ] **Step 3: Collect-only to confirm no import/collection errors** (excludes integration by default)

Run: `cd python && uv run pytest --collect-only -q`
Expected: collection succeeds with NO errors (no `ModuleNotFoundError` for `scrape_cfb_qbr` — that test was excluded). Some tests show as deselected (`integration`).

- [ ] **Step 4: Commit**

```bash
git add python/tests
git commit -m "test(python): relocate modeling test suites + fixtures (excl. qbr-scrape)"
```

---

### Task 5: Fix the `parents[2]/'python'` subprocess literals

**Files:**
- Modify: `python/tests/pregame_wp/test_cli.py:17`
- Modify: `python/tests/rb_eval/test_cli.py:17`

**Interfaces:**
- Consumes: `packages_dir` fixture (Task 2). These tests launch `python -m <pkg> --help` subprocesses with `PYTHONPATH` set to the packages dir.

- [ ] **Step 1: Write the failing test check** — run the two CLI tests as-is to see them fail

Run: `cd python && uv run pytest tests/pregame_wp/test_cli.py tests/rb_eval/test_cli.py -q`
Expected: FAIL — old `parents[2]/"python"` resolves to `python/tests/python` (nonexistent), so the subprocess gets an empty `PYTHONPATH` and `python -m pregame_wp --help` exits nonzero.

- [ ] **Step 2: Repoint each `python_dir` to the packages root**

In both files, replace the path computation (currently `pathlib.Path(__file__).resolve().parents[2] / "python"`) with the packages root. After the move, `python/tests/<pkg>/test_cli.py` → `parents[2]` is `python/` itself, so drop the `/ "python"` segment:

```python
# OLD: python_dir = pathlib.Path(__file__).resolve().parents[2] / "python"
python_dir = pathlib.Path(__file__).resolve().parents[2]  # == python/ (packages root)
```

- [ ] **Step 3: Run to verify pass**

Run: `cd python && uv run pytest tests/pregame_wp/test_cli.py tests/rb_eval/test_cli.py -q`
Expected: PASS (the subprocess now imports the packages from `python/`).

- [ ] **Step 4: Commit**

```bash
git add python/tests/pregame_wp/test_cli.py python/tests/rb_eval/test_cli.py
git commit -m "test(python): repoint CLI-subprocess PYTHONPATH to python/ packages root"
```

---

### Task 6: Mark + re-point the corpus-gated tests as `integration`

**Files:**
- Modify: `python/tests/model_training/test_cli_smoke.py`
- Modify: `python/tests/model_training/test_ingest_build.py`
- Modify: `python/tests/model_training/test_wp_realdata.py`
- Modify: `python/tests/rb_eval/test_smoke.py`

**Interfaces:**
- Consumes: `final_cache_dir` fixture (Task 2). These four tests previously read `parents[2]/"cfb"/"json"/"final"` (nonexistent in cfb-data).

- [ ] **Step 1: Re-point each FINAL path + add the `integration` marker**

In each of the four files, replace the module-level `FINAL`/`FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"` with a read from the cache-dir env (so the test is offline-runnable after warming the cache), and add `@pytest.mark.integration` to each test function. Example for `test_ingest_build.py`:

```python
import os
import pathlib
import polars as pl
import pytest
from model_training.ingest import build_training_frame

FINAL_DIR = pathlib.Path(
    os.environ.get(
        "CFB_FINAL_CACHE",
        pathlib.Path(__file__).resolve().parents[2] / ".cache" / "cfb_final",
    )
)


@pytest.mark.integration
@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="warm the cache: python -m cfb_data_ingest --seasons ...")
def test_build_training_frame_real():
    df = build_training_frame(FINAL_DIR, seasons=None)
    assert df.height > 0
```

Apply the same two-line change (cache-dir `FINAL`/`FINAL_DIR` + `@pytest.mark.integration`) to `test_cli_smoke.py`, `test_wp_realdata.py`, and `rb_eval/test_smoke.py` (keep each file's existing test body/asserts).

- [ ] **Step 2: Confirm they are deselected by default and error-free**

Run: `cd python && uv run pytest tests/model_training/test_ingest_build.py tests/model_training/test_cli_smoke.py tests/model_training/test_wp_realdata.py tests/rb_eval/test_smoke.py -q`
Expected: all **deselected** (0 selected) by `addopts='-m "not integration"'`; no collection errors.

- [ ] **Step 3: Confirm they are selectable + skip cleanly offline**

Run: `cd python && uv run pytest -m integration -q`
Expected: the integration tests are **skipped** (cache empty) with the warm-the-cache reason — not failed, not errored.

- [ ] **Step 4: Commit**

```bash
git add python/tests/model_training/test_cli_smoke.py python/tests/model_training/test_ingest_build.py python/tests/model_training/test_wp_realdata.py python/tests/rb_eval/test_smoke.py
git commit -m "test(python): gate corpus tests behind integration marker + cache-dir FINAL"
```

---

### Task 7: Phase-1 green gate + R-suite isolation check

**Files:**
- Test: the whole `python/tests/` suite + the R `tests/testthat`

- [ ] **Step 1: Run the default Python suite green**

Run: `cd python && uv run pytest -m "not integration" -q`
Expected: PASS — all moved unit tests green (incl. `test_figures.py` with the `figures`/`gam` groups installed via `uv sync --all-groups`); integration tests deselected; zero errors.

- [ ] **Step 2: Confirm pytest does NOT collect the R tests**

Run: `cd python && uv run pytest --collect-only -q | grep -i testthat || echo "no R tests collected"`
Expected: `no R tests collected` (pytest rootdir is `python/`, so `../tests/testthat` is out of scope).

- [ ] **Step 3: Confirm the R suite still runs** (R toolchain untouched)

Run (from repo root): `Rscript -e "testthat::test_dir('tests/testthat')"`
Expected: the R tests run as before (unaffected by the new `python/` tree).

- [ ] **Step 4: Commit a Phase-1 checkpoint marker** (docs note; no code)

```bash
git commit --allow-empty -m "chore(python): Phase 1 complete — modeling subsystem green offline in cfb-data"
```

---

# PHASE 2 — `cfb_data_ingest` + unify all four onto the URL/CFBD ingest

### Task 8: `cfb_data_ingest.schedule` — enumerate game_ids by season

**Files:**
- Create: `python/cfb_data_ingest/__init__.py`
- Create: `python/cfb_data_ingest/schedule.py`
- Test: `python/tests/cfb_data_ingest/test_schedule.py`
- Create: `python/tests/cfb_data_ingest/__init__.py`

**Interfaces:**
- Produces: `RAW_BASE: str`; `season_game_ids(schedule_path_or_url, seasons) -> list[int]` (reads `cfb_schedule_master.parquet`, returns `game_id` Int32 values filtered by the `season` column).

- [ ] **Step 1: Write the failing test**

```python
import polars as pl
from cfb_data_ingest.schedule import season_game_ids


def test_season_game_ids_filters_by_season(tmp_path):
    p = tmp_path / "cfb_schedule_master.parquet"
    pl.DataFrame({"game_id": [1, 2, 3], "season": [2023, 2024, 2024]}).write_parquet(p)
    assert season_game_ids(p, [2024]) == [2, 3]
    assert season_game_ids(p, None) == [1, 2, 3]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cfb_data_ingest/test_schedule.py -v`
Expected: FAIL — `ModuleNotFoundError: cfb_data_ingest`.

- [ ] **Step 3: Write `__init__.py` + `schedule.py`**

`python/cfb_data_ingest/__init__.py`:
```python
__version__ = "0.1.0"
RAW_BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb"
```

`python/cfb_data_ingest/schedule.py`:
```python
from __future__ import annotations

from pathlib import Path

import polars as pl

from . import RAW_BASE

SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule_master.parquet"


def season_game_ids(schedule_path_or_url: str | Path | None, seasons: list[int] | None) -> list[int]:
    """Return game_id values from the schedule master, optionally filtered by season."""
    src = str(schedule_path_or_url) if schedule_path_or_url is not None else SCHEDULE_URL
    lf = pl.scan_parquet(src).select("game_id", "season")
    if seasons is not None:
        lf = lf.filter(pl.col("season").is_in(seasons))
    return lf.collect().get_column("game_id").cast(pl.Int64).to_list()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/cfb_data_ingest/test_schedule.py -v`
Expected: PASS. (Create `python/tests/cfb_data_ingest/__init__.py` as an empty file first.)

- [ ] **Step 5: Commit**

```bash
git add python/cfb_data_ingest/__init__.py python/cfb_data_ingest/schedule.py python/tests/cfb_data_ingest/
git commit -m "feat(ingest): schedule-master game_id enumeration by season"
```

---

### Task 9: `cfb_data_ingest.fetch` — fetch final.json by URL into a cache dir

**Files:**
- Create: `python/cfb_data_ingest/fetch.py`
- Test: `python/tests/cfb_data_ingest/test_fetch.py`

**Interfaces:**
- Consumes: `season_game_ids` (Task 8), `sportsdataverse.dl_utils.download`.
- Produces: `final_url(game_id) -> str`; `fetch_final(seasons, cache_dir, *, schedule=None, refresh=False, downloader=None) -> dict` returning `{"fetched": int, "skipped": int, "missing": int, "total": int}` and writing `<cache_dir>/{game_id}.json` per game (preserving the full `{season, plays, ...}` object).

- [ ] **Step 1: Write the failing test** (hermetic — inject a fake downloader, no network)

```python
import json
import polars as pl
from cfb_data_ingest.fetch import fetch_final


def _fake_downloader(url, **kwargs):
    gid = url.rsplit("/", 1)[-1].removesuffix(".json")
    class R:  # mimic requests.Response surface used by fetch_final
        status_code = 200
        text = json.dumps({"season": 2024, "plays": [{"id": int(gid)}]})
    return R()


def test_fetch_final_writes_cache_and_counts(tmp_path):
    sched = tmp_path / "sched.parquet"
    pl.DataFrame({"game_id": [10, 11], "season": [2024, 2024]}).write_parquet(sched)
    cache = tmp_path / "cache"
    stats = fetch_final([2024], cache, schedule=sched, downloader=_fake_downloader)
    assert stats == {"fetched": 2, "skipped": 0, "missing": 0, "total": 2}
    assert (cache / "10.json").exists() and (cache / "11.json").exists()
    # second run skips cached
    stats2 = fetch_final([2024], cache, schedule=sched, downloader=_fake_downloader)
    assert stats2["skipped"] == 2 and stats2["fetched"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cfb_data_ingest/test_fetch.py -v`
Expected: FAIL — `cannot import name 'fetch_final'`.

- [ ] **Step 3: Write `fetch.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from . import RAW_BASE
from .schedule import season_game_ids


def final_url(game_id: int) -> str:
    return f"{RAW_BASE}/json/final/{game_id}.json"


def _default_downloader(url: str):
    from sportsdataverse.dl_utils import download  # pooled session + retry/backoff
    return download(url)


def fetch_final(
    seasons: list[int] | None,
    cache_dir: str | Path,
    *,
    schedule: str | Path | None = None,
    refresh: bool = False,
    downloader: Callable[..., object] | None = None,
) -> dict:
    """Fetch each season's final.json by URL into cache_dir/{game_id}.json. Fail-soft per game."""
    dl = downloader or _default_downloader
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ids = season_game_ids(schedule, seasons)
    fetched = skipped = missing = 0
    for gid in ids:
        dest = cache / f"{gid}.json"
        if dest.exists() and not refresh:
            try:
                json.loads(dest.read_text())  # corrupt-cache guard
                skipped += 1
                continue
            except Exception:  # noqa: BLE001 — corrupt cache: re-fetch once
                pass
        try:
            resp = dl(final_url(gid))
            if getattr(resp, "status_code", 200) != 200 or not getattr(resp, "text", ""):
                missing += 1
                continue
            json.loads(resp.text)  # validate
            dest.write_text(resp.text, encoding="utf-8")
            fetched += 1
        except Exception:  # noqa: BLE001 — one bad game cannot abort the batch
            missing += 1
    return {"fetched": fetched, "skipped": skipped, "missing": missing, "total": len(ids)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/cfb_data_ingest/test_fetch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/cfb_data_ingest/fetch.py python/tests/cfb_data_ingest/test_fetch.py
git commit -m "feat(ingest): URL fetch + cache of final.json with fail-soft + corrupt-cache guard"
```

---

### Task 10: `cfb_data_ingest` CLI (`python -m cfb_data_ingest`)

**Files:**
- Create: `python/cfb_data_ingest/cli.py`
- Create: `python/cfb_data_ingest/__main__.py`
- Test: `python/tests/cfb_data_ingest/test_cli.py`

**Interfaces:**
- Produces: `python -m cfb_data_ingest --seasons A B --cache-dir DIR [--schedule PATH] [--refresh]` → warms the cache; prints the completeness counts. `build_parser()` + `main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

```python
from cfb_data_ingest.cli import build_parser


def test_parser_has_seasons_and_cache():
    ns = build_parser().parse_args(["--seasons", "2023", "2024", "--cache-dir", "/tmp/c"])
    assert ns.seasons == [2023, 2024] and ns.cache_dir == "/tmp/c"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cfb_data_ingest/test_cli.py -v`
Expected: FAIL — `cannot import name 'build_parser'`.

- [ ] **Step 3: Write `cli.py` + `__main__.py`**

`python/cfb_data_ingest/cli.py`:
```python
from __future__ import annotations

import argparse

from .fetch import fetch_final


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_data_ingest")
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    ap.add_argument("--cache-dir", default=".cache/cfb_final")
    ap.add_argument("--schedule", default=None, help="local schedule master override; default RAW_BASE URL")
    ap.add_argument("--refresh", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stats = fetch_final(args.seasons, args.cache_dir, schedule=args.schedule, refresh=args.refresh)
    print(f"ingest: fetched={stats['fetched']} skipped={stats['skipped']} "
          f"missing={stats['missing']} total={stats['total']} -> {args.cache_dir}")
    return 0
```

`python/cfb_data_ingest/__main__.py`:
```python
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/cfb_data_ingest/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/cfb_data_ingest/cli.py python/cfb_data_ingest/__main__.py python/tests/cfb_data_ingest/test_cli.py
git commit -m "feat(ingest): cfb_data_ingest CLI to warm the final.json cache"
```

---

### Task 11: Re-point `model_training` ingest at the cache dir

**Files:**
- Modify: `python/model_training/cli.py` (the `--final-dir` default)
- Modify: `python/model_training/fourth_down/cli.py` (the `--final-dir` default + fold its duplicate read loop)

**Interfaces:**
- Consumes: `cfb_data_ingest.fetch.fetch_final` (Task 9). `build_training_frame(final_dir, seasons)` signature is UNCHANGED — only the directory it reads changes (a directory swap, per spec §5.2).

- [ ] **Step 1: Write the failing test** (CLI default now points at the cache, not `cfb/json/final`)

`python/tests/model_training/test_cli_default_cache.py`:
```python
from model_training.cli import build_parser


def test_train_ep_final_dir_defaults_to_cache():
    ns = build_parser().parse_args(["train-ep", "--seasons", "2024"])
    assert ".cache/cfb_final" in ns.final_dir.replace("\\", "/")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/model_training/test_cli_default_cache.py -v`
Expected: FAIL — default is still `cfb/json/final`.

- [ ] **Step 3: Change the two `--final-dir` defaults**

In `model_training/cli.py` and `model_training/fourth_down/cli.py`, change the argparse default from `"cfb/json/final"` to `".cache/cfb_final"`. In `fourth_down/cli.py`, replace its inline `for fpath in sorted(final_dir.glob("*.json"))` read loop with a call to the shared reader so both paths use one code path:

```python
# fourth_down/cli.py — replace the duplicate read loop with:
from model_training.ingest import _read_final_plays  # shared final.json reader
all_plays = _read_final_plays(final_dir, args.seasons)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/model_training/test_cli_default_cache.py tests/model_training -m "not integration" -q`
Expected: PASS (default points at the cache; the rest of the model_training unit suite stays green — `build_training_frame`/`_read_final_plays` unchanged).

- [ ] **Step 5: Commit**

```bash
git add python/model_training/cli.py python/model_training/fourth_down/cli.py python/tests/model_training/test_cli_default_cache.py
git commit -m "feat(model_training): default ingest to the cfb_data_ingest cache dir; unify fourth_down reader"
```

---

### Task 12: Re-point `rb_eval` ingest at the cache dir

**Files:**
- Modify: `python/rb_eval/cli.py` (the `features --final-dir` default)

**Interfaces:**
- Consumes: the cache dir. `load_rush_plays(final_dir, seasons)` signature UNCHANGED.

- [ ] **Step 1: Write the failing test**

`python/tests/rb_eval/test_cli_default_cache.py`:
```python
from rb_eval.cli import build_parser


def test_features_final_dir_defaults_to_cache():
    ns = build_parser().parse_args(["features"])
    assert ".cache/cfb_final" in ns.final_dir.replace("\\", "/")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/rb_eval/test_cli_default_cache.py -v`
Expected: FAIL — default is still `cfb/json/final`.

- [ ] **Step 3: Change the `--final-dir` default** in `rb_eval/cli.py` from `"cfb/json/final"` to `".cache/cfb_final"`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/rb_eval/test_cli_default_cache.py tests/rb_eval -m "not integration" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/rb_eval/cli.py python/tests/rb_eval/test_cli_default_cache.py
git commit -m "feat(rb_eval): default features ingest to the cfb_data_ingest cache dir"
```

---

### Task 13: Rework `cpoe` to consume final.json (R3)

**Files:**
- Modify: `python/cpoe/features.py` (the `_play_type_col` candidate tuple)
- Modify: `python/cpoe/constants.py` (`PASS_PLAY_TYPES`)
- Modify: `python/cpoe/ingest.py` (swap the per-game parquet walk for the final.json glob reader)

**Interfaces:**
- Consumes: the cache dir of final.json. Produces: `cpoe` trains off the same final.json plays as `model_training` (all 8 Approach-A features + the `completion` target are present on final.json plays; air-yards intentionally unused).

- [ ] **Step 1: Write the failing test** (final.json-style play dicts flow through `extract_pass_features`)

`python/tests/cpoe/test_final_json_features.py`:
```python
import polars as pl
from cpoe.features import extract_pass_features


def test_extract_pass_features_reads_final_json_play_types():
    plays = pl.DataFrame([
        {"type.text": "Pass Completion", "completion": True, "start.down": 1, "start.distance": 10,
         "start.yardsToEndzone": 75, "pos_score_diff_start": 0, "start.TimeSecsRem": 1800,
         "start.is_home": True, "period": 1, "passing_down": False},
        {"type.text": "Rush", "completion": False, "start.down": 2, "start.distance": 8,
         "start.yardsToEndzone": 60, "pos_score_diff_start": 0, "start.TimeSecsRem": 1700,
         "start.is_home": True, "period": 1, "passing_down": False},
    ], infer_schema_length=None)
    feats = extract_pass_features(plays)
    assert len(feats) == 1  # only the pass play survives
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cpoe/test_final_json_features.py -v`
Expected: FAIL — `_play_type_col` doesn't recognize `type.text`, so `extract_pass_features` returns empty (`len == 0`).

- [ ] **Step 3: Make the three R3 edits**

(a) `cpoe/features.py` — add `"type.text"` to the `_play_type_col` candidate names:
```python
# OLD: for c in ("playType", "play_type", "type"):
for c in ("type.text", "playType", "play_type", "type"):
```
(b) `cpoe/constants.py` — add the final.json pass-play values to `PASS_PLAY_TYPES`:
```python
PASS_PLAY_TYPES = PASS_PLAY_TYPES | {"Pass Completion", "Interception Return"}
```
(c) `cpoe/ingest.py` — replace the `<season>/<game_id>/plays.parquet` walk with the final.json glob reader (mirror `model_training.ingest._read_final_plays`):
```python
def load_season_pass_plays(final_dir, seasons=None):
    import json
    from pathlib import Path
    import polars as pl
    frames = []
    for f in sorted(Path(final_dir).glob("*.json")):
        obj = json.loads(f.read_text())
        if seasons is not None and obj.get("season") not in seasons:
            continue
        plays = obj.get("plays") or []
        if plays:
            frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    return extract_pass_features(pl.concat(frames, how="diagonal_relaxed"))
```
Also update `cpoe/cli.py`'s stale docstring/help that references the `<raw-dir>/<season>/<season_type>/<game_id>/plays.parquet` layout, and change its `--final-dir`/raw-dir default to `".cache/cfb_final"`.

- [ ] **Step 4: Run to verify it passes** (new test + the whole cpoe unit suite)

Run: `cd python && uv run pytest tests/cpoe/test_final_json_features.py tests/cpoe -m "not integration" -q`
Expected: PASS (train_cp/loso/validate untouched; `_COL_MAP` needs no change).

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/features.py python/cpoe/constants.py python/cpoe/ingest.py python/cpoe/cli.py python/tests/cpoe/test_final_json_features.py
git commit -m "feat(cpoe): consume final.json plays (type.text + pass-type values + glob reader)"
```

---

### Task 14: Bring `pregame_wp` CFBD ingest under the managed/gated framework (R4)

**Files:**
- Modify: `python/pregame_wp/data_ingest.py` (cache the CFBD fetch under the same cache root; clear error on missing key)
- Modify: `python/tests/pregame_wp/test_data_ingest.py` (mark CFBD-touching tests `integration`)
- Create: `python/cfb_data_ingest/README.md` (document `CFB_DATA_API_KEY`)

**Interfaces:**
- pregame_wp keeps reading CFBD `/games`,`/plays`,`/drives` (R4: `team_summaries` cannot supply the five factors). SP1 only *manages* it: documented key, gated tests, optional disk cache. Full URL-unification (re-point at the published PBP/drives releases) is deferred.

- [ ] **Step 1: Write the failing test** — a CFBD-touching test must be marked integration (so the default suite never needs a key)

`python/tests/pregame_wp/test_ingest_gated.py`:
```python
import pytest


def test_cfbd_tests_are_integration_marked():
    # Sentinel: the live-CFBD ingest test must carry the integration marker so the
    # default `-m "not integration"` run never requires CFB_DATA_API_KEY.
    import tests.pregame_wp.test_data_ingest as t  # noqa: F401
    marks = getattr(t, "pytestmark", [])
    names = {m.name for m in (marks if isinstance(marks, list) else [marks])}
    assert "integration" in names
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/pregame_wp/test_ingest_gated.py -v`
Expected: FAIL — `test_data_ingest.py` has no module-level `integration` mark.

- [ ] **Step 3: Mark the CFBD tests + harden the key error**

In `python/tests/pregame_wp/test_data_ingest.py`, add a module-level mark and keep the existing `CFB_DATA_API_KEY` skip:
```python
import pytest
pytestmark = pytest.mark.integration  # CFBD-network tests; deselected by default
```
In `pregame_wp/data_ingest.py`, ensure a missing key raises a clear, actionable error (keep behavior; just confirm the message names `CFB_DATA_API_KEY`), and write fetched CFBD payloads under `<cache_dir>/cfbd/` so repeat runs are cheap.

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/pregame_wp/test_ingest_gated.py tests/pregame_wp -m "not integration" -q`
Expected: PASS — the gated sentinel passes; the rest of pregame_wp's unit suite (five-factor math, ep_curve, box_score with inline fixtures) stays green with no key.

- [ ] **Step 5: Commit**

```bash
git add python/pregame_wp/data_ingest.py python/tests/pregame_wp/test_data_ingest.py python/tests/pregame_wp/test_ingest_gated.py python/cfb_data_ingest/README.md
git commit -m "feat(pregame_wp): manage CFBD ingest (key docs, integration-gate tests, cache)"
```

---

### Task 15: Phase-2 green gate + offline integration smoke

**Files:**
- Test: full suite + a tiny warmed-cache integration run

- [ ] **Step 1: Default suite stays green**

Run: `cd python && uv run pytest -m "not integration" -q`
Expected: PASS — all unit suites + the new `cfb_data_ingest` tests green; integration deselected.

- [ ] **Step 2: Warm a tiny cache from a fixture schedule (offline) and run one integration test**

Create `python/tests/cfb_data_ingest/fixtures/` with a 1-row schedule parquet + one `{game_id}.json` already placed in a temp cache, then:
```bash
cd python && CFB_FINAL_CACHE="$(pwd)/.cache/cfb_final" uv run pytest -m integration tests/model_training/test_ingest_build.py -q
```
Expected: with the cache warmed, the integration test runs (not skipped) and passes on the small frame; with an empty cache it skips with the warm-the-cache reason.

- [ ] **Step 3: Commit checkpoint**

```bash
git commit --allow-empty -m "chore(python): Phase 2 complete — all four packages on managed URL/CFBD ingest"
```

---

# PHASE 3 — `cfb_model_pbp` (carry EP/WP/EPA/WPA + net-new CPOE)

### Task 16: Freeze the model-PBP schema (R2)

**Files:**
- Create: `python/cfb_model_pbp/__init__.py`
- Create: `python/cfb_model_pbp/schema.py`
- Test: `python/tests/cfb_model_pbp/test_schema.py`
- Create: `python/tests/cfb_model_pbp/__init__.py`

**Interfaces:**
- Produces: `IDENTITY_COLS`, `DESCRIPTOR_COLS`, `PREDICTION_COLS`, `MODEL_PBP_COLUMNS` (ordered), `JOIN_KEYS = ("game_id", "id")`, and `CARRY_RENAME` (final.json EP/WP names → snake before/after names). This is the SP2 publish contract.

- [ ] **Step 1: Write the failing test**

```python
from cfb_model_pbp.schema import MODEL_PBP_COLUMNS, JOIN_KEYS, CARRY_RENAME


def test_schema_contract():
    assert JOIN_KEYS == ("game_id", "id")
    for c in ("game_id", "id", "epa", "wpa", "cpoe", "completion_prob", "ep_before", "wp_after"):
        assert c in MODEL_PBP_COLUMNS
    # EP/WP are carried (renamed) from final.json, not re-scored in SP1
    assert CARRY_RENAME["EPA"] == "epa" and CARRY_RENAME["wp_before"] == "wp_before"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: cfb_model_pbp`.

- [ ] **Step 3: Write `__init__.py` + `schema.py`**

`python/cfb_model_pbp/__init__.py`:
```python
__version__ = "0.1.0"
```

`python/cfb_model_pbp/schema.py`:
```python
from __future__ import annotations

JOIN_KEYS = ("game_id", "id")

IDENTITY_COLS = ["game_id", "id", "sequenceNumber", "game_play_number", "drive.id", "season", "week", "period"]

DESCRIPTOR_COLS = [
    "pos_team", "def_pos_team", "start.pos_team.name", "homeTeamId", "awayTeamId",
    "homeTeamName", "awayTeamName", "type.text", "text", "start.down", "start.distance",
    "start.yardsToEndzone", "pos_score_diff_start", "start.TimeSecsRem", "start.is_home",
    "passing_down", "pass", "rush", "completion", "scoring_play", "statYardage", "passer_player_name",
]

PREDICTION_COLS = [
    "ep_before", "ep_after", "epa", "def_epa",
    "wp_before", "wp_after", "wpa", "def_wp_before", "def_wp_after",
    "home_wp_before", "away_wp_before", "home_wp_after", "away_wp_after",
    "completion_prob", "cpoe",
    "model_pbp_version", "ep_model_version", "wp_model_version", "cp_model_version", "scored_date",
]

MODEL_PBP_COLUMNS = IDENTITY_COLS + DESCRIPTOR_COLS + PREDICTION_COLS

# EP/WP/EPA/WPA are CARRIED from final.json (they already embed CFBPlayProcess differencing);
# only CPOE is net-new scored in SP1. Map final.json source names -> frozen snake names.
CARRY_RENAME = {
    "EP_start": "ep_before", "EP_end": "ep_after", "EPA": "epa",
    "wp_before": "wp_before", "wp_after": "wp_after", "wpa": "wpa",
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_schema.py -v`
Expected: PASS. (Create `python/tests/cfb_model_pbp/__init__.py` empty first.)

- [ ] **Step 5: Commit**

```bash
git add python/cfb_model_pbp/__init__.py python/cfb_model_pbp/schema.py python/tests/cfb_model_pbp/
git commit -m "feat(model_pbp): freeze the model-PBP column contract (carry EP/WP + net-new CPOE)"
```

---

### Task 17: Build the model-PBP frame (carry EP/WP/EPA/WPA + identity/descriptors)

**Files:**
- Create: `python/cfb_model_pbp/build.py`
- Test: `python/tests/cfb_model_pbp/test_build.py`

**Interfaces:**
- Consumes: the final.json plays frame (via `model_training.ingest._read_final_plays`) + `schema.py`.
- Produces: `build_carry_frame(final_dir, seasons=None) -> pl.DataFrame` — identity + descriptor cols + renamed `ep_before/ep_after/epa/wp_before/wp_after/wpa` carried from final.json; drops rows missing required carry cols and returns the frame; `last_completeness()` exposes kept/dropped counts.

- [ ] **Step 1: Write the failing test**

```python
import polars as pl
from cfb_model_pbp.build import build_carry_frame


def test_build_carry_renames_and_keeps_keys(tmp_path):
    import json
    game = {"season": 2024, "plays": [{
        "game_id": 1, "id": 100, "sequenceNumber": 1, "game_play_number": 1, "drive.id": "d1",
        "week": 1, "period": 1, "EP_start": 2.0, "EP_end": 2.5, "EPA": 0.5,
        "wp_before": 0.5, "wp_after": 0.55, "wpa": 0.05, "type.text": "Rush", "completion": False,
    }]}
    (tmp_path / "1.json").write_text(json.dumps(game))
    df = build_carry_frame(tmp_path, seasons=[2024])
    assert {"game_id", "id", "epa", "wp_after", "ep_before"} <= set(df.columns)
    row = df.row(0, named=True)
    assert row["epa"] == 0.5 and row["ep_before"] == 2.0 and row["wp_after"] == 0.55
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_build.py -v`
Expected: FAIL — `cannot import name 'build_carry_frame'`.

- [ ] **Step 3: Write `build.py` (carry half)**

```python
from __future__ import annotations

from pathlib import Path

import polars as pl

from model_training.ingest import _read_final_plays
from .schema import CARRY_RENAME, DESCRIPTOR_COLS, IDENTITY_COLS

_REQUIRED_CARRY = list(CARRY_RENAME.keys())
_LAST = {"kept": 0, "dropped": 0}


def build_carry_frame(final_dir, seasons=None) -> pl.DataFrame:
    df = _read_final_plays(final_dir, seasons)
    if df.is_empty():
        return df
    # keep only rows that carry the EP/WP source columns (raw/pre-enrichment games lack them)
    present_required = [c for c in _REQUIRED_CARRY if c in df.columns]
    before = df.height
    if present_required:
        df = df.drop_nulls(subset=present_required)
    _LAST["kept"], _LAST["dropped"] = df.height, before - df.height
    df = df.rename({k: v for k, v in CARRY_RENAME.items() if k in df.columns})
    carry = [c for c in (IDENTITY_COLS + DESCRIPTOR_COLS + list(CARRY_RENAME.values())) if c in df.columns]
    return df.select(carry)


def last_completeness() -> dict:
    return dict(_LAST)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_build.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/cfb_model_pbp/build.py python/tests/cfb_model_pbp/test_build.py
git commit -m "feat(model_pbp): build carry frame (identity/descriptors + renamed EP/WP/EPA/WPA)"
```

---

### Task 18: Score net-new CPOE onto the model-PBP frame

**Files:**
- Modify: `python/cfb_model_pbp/build.py` (add `score_cpoe`)
- Test: `python/tests/cfb_model_pbp/test_cpoe_scoring.py`

**Interfaces:**
- Consumes: a trained CP booster (`cpoe.train_cp` saved `.ubj`), `cpoe.features.extract_pass_features`.
- Produces: `score_cpoe(carry_df, plays_df, cp_model_path) -> pl.DataFrame` — appends `completion_prob` (CP booster predict on pass rows) and `cpoe = completion - completion_prob`; NaN on non-pass rows; joined on `JOIN_KEYS`.

- [ ] **Step 1: Write the failing test** (inject a stub predictor so no real model is needed)

```python
import polars as pl
from cfb_model_pbp.build import score_cpoe


def test_score_cpoe_appends_completion_prob_and_cpoe():
    carry = pl.DataFrame({"game_id": [1, 1], "id": [100, 101], "completion": [True, None], "pass": [True, False]})
    plays = pl.DataFrame({"game_id": [1, 1], "id": [100, 101], "completion": [True, None],
                          "type.text": ["Pass Completion", "Rush"], "start.down": [1, 2],
                          "start.distance": [10, 8], "start.yardsToEndzone": [75, 60],
                          "pos_score_diff_start": [0, 0], "start.TimeSecsRem": [1800, 1700],
                          "start.is_home": [True, True], "period": [1, 1], "passing_down": [False, False]},
                         infer_schema_length=None)
    out = score_cpoe(carry, plays, cp_model_path=None, _predict=lambda X: [0.6])  # 1 pass row -> cp 0.6
    pass_row = out.filter(pl.col("id") == 100).row(0, named=True)
    assert abs(pass_row["completion_prob"] - 0.6) < 1e-9
    assert abs(pass_row["cpoe"] - (1.0 - 0.6)) < 1e-9
    assert out.filter(pl.col("id") == 101).row(0, named=True)["completion_prob"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_cpoe_scoring.py -v`
Expected: FAIL — `cannot import name 'score_cpoe'`.

- [ ] **Step 3: Add `score_cpoe` to `build.py`**

```python
def score_cpoe(carry_df: pl.DataFrame, plays_df: pl.DataFrame, cp_model_path, _predict=None) -> pl.DataFrame:
    """Append completion_prob + cpoe (pass plays only) to carry_df, joined on (game_id, id)."""
    from cpoe.features import extract_pass_features
    feats = extract_pass_features(plays_df)  # pass rows only, with id retained
    if feats.is_empty():
        return carry_df.with_columns(completion_prob=pl.lit(None, dtype=pl.Float64),
                                     cpoe=pl.lit(None, dtype=pl.Float64))
    if _predict is None:
        import numpy as np
        import xgboost as xgb
        from cpoe.constants import FEATURE_COLS
        booster = xgb.Booster(); booster.load_model(str(cp_model_path))
        preds = booster.predict(xgb.DMatrix(feats.select(FEATURE_COLS).to_pandas()))
        preds = np.asarray(preds).tolist()
    else:
        preds = _predict(feats)
    scored = feats.select("game_id", "id", "completion").with_columns(
        completion_prob=pl.Series("completion_prob", preds, dtype=pl.Float64),
    ).with_columns(cpoe=(pl.col("completion").cast(pl.Float64) - pl.col("completion_prob")))
    return carry_df.join(scored.select("game_id", "id", "completion_prob", "cpoe"),
                         on=["game_id", "id"], how="left")
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_cpoe_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/cfb_model_pbp/build.py python/tests/cfb_model_pbp/test_cpoe_scoring.py
git commit -m "feat(model_pbp): net-new CPOE scoring (completion_prob + cpoe) joined on (game_id,id)"
```

---

### Task 19: model-PBP CLI — assemble, stamp provenance, write parquet

**Files:**
- Create: `python/cfb_model_pbp/cli.py`
- Create: `python/cfb_model_pbp/__main__.py`
- Test: `python/tests/cfb_model_pbp/test_cli.py`

**Interfaces:**
- Produces: `python -m cfb_model_pbp --final-dir DIR --cp-model PATH --out PATH [--seasons ...]` → writes the model-PBP parquet (carry frame + CPOE + provenance stamps), prints kept/dropped completeness.

- [ ] **Step 1: Write the failing test**

```python
from cfb_model_pbp.cli import build_parser


def test_parser_requires_out_and_cp_model():
    ns = build_parser().parse_args(["--final-dir", ".cache/cfb_final", "--cp-model", "m.ubj", "--out", "o.parquet"])
    assert ns.out == "o.parquet" and ns.cp_model == "m.ubj"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_cli.py -v`
Expected: FAIL — `cannot import name 'build_parser'`.

- [ ] **Step 3: Write `cli.py` + `__main__.py`**

`python/cfb_model_pbp/cli.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from . import __version__
from .build import build_carry_frame, last_completeness, score_cpoe
from .schema import PREDICTION_COLS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_model_pbp")
    ap.add_argument("--final-dir", default=".cache/cfb_final")
    ap.add_argument("--cp-model", required=True, help="trained CP booster (.ubj)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from model_training.ingest import _read_final_plays
    plays = _read_final_plays(args.final_dir, args.seasons)
    carry = build_carry_frame(args.final_dir, args.seasons)
    scored = score_cpoe(carry, plays, args.cp_model)
    scored = scored.with_columns(
        model_pbp_version=pl.lit(__version__),
        cp_model_version=pl.lit(Path(args.cp_model).name),
        ep_model_version=pl.lit("carried:final_json"),
        wp_model_version=pl.lit("carried:final_json"),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(args.out)
    comp = last_completeness()
    print(f"model_pbp: rows={scored.height} kept={comp['kept']} dropped={comp['dropped']} -> {args.out}")
    return 0
```

`python/cfb_model_pbp/__main__.py`:
```python
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && uv run pytest tests/cfb_model_pbp/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/cfb_model_pbp/cli.py python/cfb_model_pbp/__main__.py python/tests/cfb_model_pbp/test_cli.py
git commit -m "feat(model_pbp): CLI to assemble + stamp + write the model-PBP parquet"
```

---

### Task 20: Phase-3 green gate + full SP1 acceptance

**Files:**
- Test: full suite + a manual one-season end-to-end

- [ ] **Step 1: Full default suite green**

Run: `cd python && uv run pytest -m "not integration" -q`
Expected: PASS — all packages + `cfb_data_ingest` + `cfb_model_pbp` unit tests green; integration deselected.

- [ ] **Step 2: Manual one-season end-to-end (small season, offline-capable once cached)**

```bash
cd python
uv run python -m cfb_data_ingest --seasons 2024 --cache-dir .cache/cfb_final
uv run python -m model_training train-ep --seasons 2024 --out artifacts/ep.ubj
uv run python -m cpoe --final-dir .cache/cfb_final --out artifacts/cp.ubj   # train CP (per cpoe CLI)
uv run python -m cfb_model_pbp --final-dir .cache/cfb_final --cp-model artifacts/cp.ubj --out artifacts/model_pbp_2024.parquet --seasons 2024
```
Expected: the cache warms; `artifacts/ep.ubj` + `model_card.json`, `artifacts/cp.ubj`, and `artifacts/model_pbp_2024.parquet` are produced locally; the model-PBP print shows a nonzero `rows` and a `kept/dropped` completeness count. (All under the gitignored `artifacts/`.)

- [ ] **Step 3: Confirm artifacts are gitignored** (nothing to commit from the run)

Run: `git status --short python/artifacts python/.cache`
Expected: empty (both ignored).

- [ ] **Step 4: Final SP1 checkpoint commit**

```bash
git commit --allow-empty -m "chore(python): SP1 complete — modeling subsystem + URL ingest + model-PBP build, green offline"
```

---

## Self-Review

**1. Spec coverage** (spec §2/§5/§6):
- D3/D7 python/ subsystem + `python/tests/` → Tasks 1–4. ✓
- D1/D8 unify ingest: model_training (T11), rb_eval (T12), cpoe rework R3 (T13), pregame_wp CFBD-managed R4 (T14). ✓
- D6 `sportsdataverse` hard dep → T1 deps + T3 import check. ✓
- D2/D9 model-PBP Phase 3 (carry EP/WP + net-new CPOE, R2 schema) → T16–T19. ✓
- Integration-marker registration + sweep + cache re-point → T1, T6, T14. ✓
- `test_qbr_scrape` excluded, missing `__init__` added, fixtures moved → T4. ✓
- Acceptance #1 (default green) T7/T15/T20; #2 (hermetic ingest) T9/T11; #3 (integration sweep+offline) T6/T15; #4 (one-season e2e) T20; #5 (cfb-raw untouched) — all moves `cp` from cfb-raw, never edit it; #6 (R suite isolation) T7. ✓

**2. Placeholder scan:** every code/test step carries real code or an exact command. No TBD/TODO. ✓

**3. Type consistency:** `build_training_frame(final_dir, seasons)` / `_read_final_plays(final_dir, seasons)` used unchanged in T11/T13/T17/T19; `fetch_final(...)` signature consistent T9→T10; `score_cpoe`/`build_carry_frame`/`last_completeness` consistent T17→T18→T19; `JOIN_KEYS=("game_id","id")` consistent T16→T18. ✓

## Open follow-ups (documented, not SP1 blockers)
- **EP/WP carried vs re-scored:** SP1 carries final.json EP/WP/EPA/WPA (they embed CFBPlayProcess differencing). Re-scoring with the suite's own boosters requires porting `__process_epa`/`__process_wpa` (~466 lines) — deferred (own effort).
- **pregame_wp full URL-unification:** R4 ruled out `team_summaries`; viable path is re-pointing at the published PBP/drives releases or publishing the 5FR boxes — deferred.
- **`highlight_yards`:** rb_eval's aggregate consumes a `highlight_yards` per-play column it does NOT compute; confirm final.json plays carry it (else the rb_eval `aggregate` integration path errors).
