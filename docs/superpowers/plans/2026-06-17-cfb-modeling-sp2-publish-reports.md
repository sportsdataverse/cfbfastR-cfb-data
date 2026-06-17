# CFB Modeling SP2 — Publish + Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn SP1's local model outputs into published products — per-model Markdown reports committed under `docs/models/`, the model-PBP dataset published to `espn_cfb_model_pbp` (rds/csv via the R writer), and all trained model artifacts published to `espn_cfb_model_artifacts`.

**Architecture:** Two new Python packages — `cfb_model_reports` (pure Markdown renderer + per-family metric runners + a discovery CLI) and `cfb_model_publish` (a `gh release` artifact uploader with `--dry-run`) — plus one thin R script (`espn_cfb_16_model_pbp.R`) that feeds the Python-built model-PBP parquet through the existing `write_dataset`/`publish_dataset`. Reports/publish discover models by globbing `python/artifacts/**/*.json` cards and reading `model_type`. Run-locally-then-publish; CI deferred.

**Tech Stack:** Python ≥3.11 (polars 1.x, pandas, xgboost, scikit-learn for inline log-loss/Brier, joblib for the rb_eval GAM), the existing per-track `validate`/`loso`/`figures` modules, R (`_data_utils.R` publish helpers + piggyback), `gh` CLI.

## Global Constraints

- Everything Python under `python/`; tests under `python/tests/`. Run pytest from `python/`.
- `integration` marker registered + default-deselected (`addopts='-m "not integration"'`, from SP1). Model/network/`gh`-dependent tests are `integration`-marked; the default suite is hermetic.
- **Discovery rule:** report + publish code finds models by globbing `python/artifacts/**/*.json` (the model-card sidecars). A `write_xgb_model_card` card has a `model_type` key (`ep`/`wp_spread`/`wp_naive`/`qbr`/`cpoe`/`fourth_down`); the rb_eval card lacks `model_type` (it has `model_formula`/`features`/`target`) and is identified by its sibling `.pkl` → treated as `model_type="rb_eval"`. The model file is the card's sibling (same stem, `.ubj` or `.pkl`).
- Reports are Markdown committed under `docs/models/`; figures committed under `docs/models/figures/` and embedded with repo-relative links (`figures/<name>.png`).
- Publish targets: `sportsdataverse/sportsdataverse-data`, tags `espn_cfb_model_pbp` (dataset) + `espn_cfb_model_artifacts` (models). Both registered in `R/releases_init.R`.
- polars 1.x API only. Conventional Commits. **NO AI co-author trailers.**
- Do NOT modify the SP1 packages' training/ingest logic; SP2 only READS their outputs + calls their existing `validate`/`figures` functions.

## File Structure

**New (Python):**
- `python/cfb_model_reports/__init__.py` — `__version__`.
- `python/cfb_model_reports/discovery.py` — `discover_models(artifacts_dir) -> list[ModelArtifact]` (glob cards, read model_type, locate model file).
- `python/cfb_model_reports/report.py` — pure Markdown renderer: `ModelReport` dataclass + `render_model_report(r) -> str` + `render_index(reports) -> str`.
- `python/cfb_model_reports/metrics.py` — per-family metric runners returning a `dict[str, float|str]`.
- `python/cfb_model_reports/cli.py` + `__main__.py` — `python -m cfb_model_reports --artifacts <dir> --cache <dir> --out docs/models`.
- `python/cfb_model_publish/__init__.py`, `artifacts.py`, `cli.py`, `__main__.py` — `gh release upload` of artifacts (+ `--dry-run`).
- `python/tests/cfb_model_reports/` + `python/tests/cfb_model_publish/`.

**New (R):** `R/espn_cfb_16_model_pbp.R`.
**Modified (R):** `R/releases_init.R` (register 2 tags).
**New (committed docs):** `docs/models/README.md`, `docs/models/<model>.md`, `docs/models/figures/` (produced by the CLI; committed as part of the acceptance run, not by a task's unit test).

---

## PHASE A — Reports

### Task 1: `report.py` — pure Markdown renderer

**Files:**
- Create: `python/cfb_model_reports/__init__.py`, `python/cfb_model_reports/report.py`
- Create: `python/tests/cfb_model_reports/__init__.py`, `python/tests/cfb_model_reports/test_report.py`

**Interfaces:**
- Produces: `@dataclass ModelReport(model_type:str, title:str, metrics:dict[str,object], figures:list[str], provenance:dict[str,object], notes:list[str])`; `render_model_report(r: ModelReport) -> str`; `render_index(reports: list[ModelReport]) -> str`. Pure — no I/O, no model loading.

- [ ] **Step 1: Write the failing test**
```python
from cfb_model_reports.report import ModelReport, render_model_report, render_index


def test_render_model_report_has_metrics_figures_provenance():
    r = ModelReport(
        model_type="cpoe", title="CPOE Model",
        metrics={"log_loss": 0.512, "brier_score": 0.187, "n": 3112},
        figures=["figures/cpoe_calibration.png"],
        provenance={"trained_date": "2026-06-17", "features": ["down", "distance"], "training_seasons": [2014, 2024]},
        notes=["QBR correlation requires ESPN QBR (integration-only)."],
    )
    md = render_model_report(r)
    assert "# CPOE Model" in md
    assert "log_loss" in md and "0.512" in md          # metrics table
    assert "![](figures/cpoe_calibration.png)" in md    # embedded figure (repo-relative)
    assert "2026-06-17" in md and "down" in md          # provenance
    assert "QBR correlation requires" in md             # notes


def test_render_index_links_each_report():
    rs = [ModelReport("ep", "EP", {}, [], {}, []), ModelReport("cpoe", "CPOE", {}, [], {}, [])]
    idx = render_index(rs)
    assert "[EP](ep.md)" in idx and "[CPOE](cpoe.md)" in idx
```

- [ ] **Step 2: Run to verify it fails**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: cfb_model_reports`.

- [ ] **Step 3: Implement**
`python/cfb_model_reports/__init__.py`:
```python
__version__ = "0.1.0"
```
`python/cfb_model_reports/report.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelReport:
    model_type: str
    title: str
    metrics: dict
    figures: list  # repo-relative paths, e.g. "figures/cpoe_calibration.png"
    provenance: dict
    notes: list = field(default_factory=list)


def _metrics_table(metrics: dict) -> str:
    if not metrics:
        return "_No metrics available._\n"
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in metrics.items())
    return f"| metric | value |\n|---|---|\n{rows}\n"


def render_model_report(r: ModelReport) -> str:
    parts = [f"# {r.title}\n", "## Metrics\n", _metrics_table(r.metrics)]
    if r.figures:
        parts.append("\n## Figures\n")
        parts += [f"![]({p})\n" for p in r.figures]
    parts.append("\n## Provenance\n")
    parts.append(_metrics_table(r.provenance) if r.provenance else "_n/a_\n")
    if r.notes:
        parts.append("\n## Notes\n")
        parts += [f"- {n}\n" for n in r.notes]
    return "\n".join(parts)


def render_index(reports: list) -> str:
    lines = ["# CFB Model Reports\n", "Per-model metrics, calibration, and provenance.\n", "## Models\n"]
    lines += [f"- [{r.title}]({r.model_type}.md)" for r in reports]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run to verify it passes**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git -C "<cfb-data>" add python/cfb_model_reports/__init__.py python/cfb_model_reports/report.py python/tests/cfb_model_reports/
git -C "<cfb-data>" commit -m "feat(model_reports): pure Markdown report renderer + index"
```

---

### Task 2: `discovery.py` — find models from artifact cards

**Files:**
- Create: `python/cfb_model_reports/discovery.py`
- Create: `python/tests/cfb_model_reports/test_discovery.py`

**Interfaces:**
- Consumes: nothing. Produces: `@dataclass ModelArtifact(model_type:str, model_path:Path, card_path:Path, card:dict)`; `discover_models(artifacts_dir: str|Path) -> list[ModelArtifact]`. Globs `*.json` cards recursively; `model_type` from the card, else `"rb_eval"` if the sibling is `.pkl`; the model file is the same-stem `.ubj`/`.pkl` sibling. Skips a card with no model sibling.

- [ ] **Step 1: Write the failing test**
```python
import json
from pathlib import Path
from cfb_model_reports.discovery import discover_models


def test_discover_reads_model_type_and_locates_sibling(tmp_path):
    (tmp_path / "ep.ubj").write_bytes(b"x")
    (tmp_path / "ep.json").write_text(json.dumps({"model_type": "ep", "features": ["down"]}))
    (tmp_path / "xrepa_final.pkl").write_bytes(b"y")
    (tmp_path / "xrepa_final.json").write_text(json.dumps({"model_formula": "s(0)+s(1)", "target": "unadjusted_epa"}))
    (tmp_path / "orphan.json").write_text(json.dumps({"model_type": "ghost"}))  # no sibling model -> skipped
    found = {m.model_type: m for m in discover_models(tmp_path)}
    assert set(found) == {"ep", "rb_eval"}
    assert found["ep"].model_path.name == "ep.ubj" and found["ep"].card["features"] == ["down"]
    assert found["rb_eval"].model_path.suffix == ".pkl"
```

- [ ] **Step 2: Run to verify it fails**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_discovery.py -v`
Expected: FAIL — `cannot import name 'discover_models'`.

- [ ] **Step 3: Implement**
`python/cfb_model_reports/discovery.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelArtifact:
    model_type: str
    model_path: Path
    card_path: Path
    card: dict


def discover_models(artifacts_dir) -> list:
    out = []
    for card_path in sorted(Path(artifacts_dir).rglob("*.json")):
        try:
            card = json.loads(card_path.read_text())
        except Exception:  # noqa: BLE001 — skip unreadable cards
            continue
        ubj = card_path.with_suffix(".ubj")
        pkl = card_path.with_suffix(".pkl")
        if ubj.exists():
            model_path = ubj
        elif pkl.exists():
            model_path = pkl
        else:
            continue  # card with no model sibling -> skip
        model_type = card.get("model_type") or ("rb_eval" if model_path.suffix == ".pkl" else model_path.stem)
        out.append(ModelArtifact(model_type=model_type, model_path=model_path, card_path=card_path, card=card))
    return out
```

- [ ] **Step 4: Run to verify it passes**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_discovery.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git -C "<cfb-data>" add python/cfb_model_reports/discovery.py python/tests/cfb_model_reports/test_discovery.py
git -C "<cfb-data>" commit -m "feat(model_reports): discover models from artifact cards"
```

---

### Task 3: `metrics.py` — per-family metric runners + provenance extraction

**Files:**
- Create: `python/cfb_model_reports/metrics.py`
- Create: `python/tests/cfb_model_reports/test_metrics.py`

**Interfaces:**
- Consumes: `cpoe.validate.calibration_metrics`, `cpoe.validate.feature_importance`, `rb_eval.validate.{calibration_table,weighted_cal_error,weighted_r2}`, `model_training.model_card` card keys. Produces: `provenance_from_card(card: dict) -> dict` (features / hyperparams / training_seasons / trained_date / xgboost_version, tolerant of missing keys); `rb_eval_metrics(loso_parquet: str|Path) -> dict` (reads `xrepa_loso.parquet` → `weighted_r2` + `weighted_cal_error`); `xgb_importance(model_path, top_n=15) -> dict` (booster gain importance). Model-and-data-dependent metric runners (EP/WP/CPOE/fourth-down predictions) are `integration`-only and live behind `compute_classification_metrics(y_true, y_pred) -> dict` (pure: sklearn log_loss + brier).

- [ ] **Step 1: Write the failing test** (pure pieces only — hermetic)
```python
import numpy as np
import polars as pl
from cfb_model_reports.metrics import provenance_from_card, compute_classification_metrics, rb_eval_metrics


def test_provenance_from_card_tolerates_missing():
    p = provenance_from_card({"features": ["a", "b"], "training_seasons": [2014, 2024], "trained_date": "2026-06-17"})
    assert p["features"] == ["a", "b"] and p["training_seasons"] == [2014, 2024] and p["trained_date"] == "2026-06-17"
    assert provenance_from_card({}) == {"features": [], "hyperparameters": {}, "training_seasons": None,
                                        "trained_date": None, "xgboost_version": None}


def test_compute_classification_metrics_binary():
    y = np.array([1, 0, 1, 0]); p = np.array([0.9, 0.1, 0.8, 0.2])
    m = compute_classification_metrics(y, p)
    assert m["n"] == 4 and 0.0 <= m["brier_score"] <= 1.0 and m["log_loss"] > 0.0


def test_rb_eval_metrics_from_loso(tmp_path):
    lp = tmp_path / "xrepa_loso.parquet"
    pl.DataFrame({"exp_rb_epa": [0.1, 0.2, 0.3, 0.4], "target": [0.12, 0.18, 0.31, 0.39]}).write_parquet(lp)
    m = rb_eval_metrics(lp)
    assert "weighted_r2" in m and "weighted_cal_error" in m
```

- [ ] **Step 2: Run to verify it fails**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_metrics.py -v`
Expected: FAIL — `cannot import name 'provenance_from_card'`.

- [ ] **Step 3: Implement**
`python/cfb_model_reports/metrics.py`:
```python
from __future__ import annotations

from pathlib import Path


def provenance_from_card(card: dict) -> dict:
    return {
        "features": card.get("features") or [],
        "hyperparameters": card.get("hyperparameters") or {},
        "training_seasons": card.get("training_seasons"),
        "trained_date": card.get("trained_date"),
        "xgboost_version": card.get("xgboost_version"),
    }


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Binary/multiclass log-loss + (binary) Brier, computed inline (sklearn)."""
    import numpy as np
    from sklearn.metrics import log_loss
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    out = {"n": int(yt.shape[0]), "log_loss": float(log_loss(yt, yp))}
    if yp.ndim == 1:  # binary
        out["brier_score"] = float(np.mean((yp - yt) ** 2))
    return out


def xgb_importance(model_path, top_n: int = 15) -> dict:
    import xgboost as xgb
    b = xgb.Booster(); b.load_model(str(model_path))
    score = b.get_score(importance_type="gain")
    top = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {k: round(float(v), 4) for k, v in top}


def rb_eval_metrics(loso_parquet) -> dict:
    import polars as pl
    from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2
    cv = pl.read_parquet(loso_parquet)
    tbl = calibration_table(cv)
    return {"weighted_r2": round(float(weighted_r2(tbl)), 4),
            "weighted_cal_error": round(float(weighted_cal_error(tbl)), 4),
            "n": int(cv.height)}
```

- [ ] **Step 4: Run to verify it passes**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git -C "<cfb-data>" add python/cfb_model_reports/metrics.py python/tests/cfb_model_reports/test_metrics.py
git -C "<cfb-data>" commit -m "feat(model_reports): metric runners (provenance, classification, importance, rb_eval LOSO)"
```

---

### Task 4: `cli.py` — orchestrate discovery → metrics/figures → write reports

**Files:**
- Create: `python/cfb_model_reports/cli.py`, `python/cfb_model_reports/__main__.py`
- Create: `python/tests/cfb_model_reports/test_cli.py`

**Interfaces:**
- Consumes: `discovery.discover_models`, `report.*`, `metrics.*`, `cpoe/loso`, the per-track `figures`. Produces: `python -m cfb_model_reports --artifacts <dir> --cache <dir> --out docs/models [--rb-loso <parquet>]`. `build_parser()` + `main(argv)->int`. For each discovered model it builds a `ModelReport` (offline-safe metrics: provenance always; rb_eval → `rb_eval_metrics` if its loso parquet is present; xgb models → `xgb_importance`; classification log-loss/Brier + calibration figures require predictions → attempted only when `--cache` data is available, else added as a `note`). Writes `<out>/<model_type>.md` + `<out>/README.md`. Emits a `reported/skipped` completeness line. QBR correlation-vs-ESPN is recorded as a `note` (integration-only).

- [ ] **Step 1: Write the failing test** (parser + offline report write on synthetic artifacts — no real models needed for the importance-free path)
```python
import json
from cfb_model_reports.cli import build_parser, main


def test_parser_defaults():
    ns = build_parser().parse_args(["--artifacts", "a", "--out", "docs/models"])
    assert ns.artifacts == "a" and ns.out == "docs/models"


def test_main_writes_reports_and_index(tmp_path):
    art = tmp_path / "artifacts"; art.mkdir()
    # rb_eval: card + pkl + a loso parquet (rb_eval_metrics path, no xgboost needed)
    (art / "xrepa_final.pkl").write_bytes(b"y")
    (art / "xrepa_final.json").write_text(json.dumps({"target": "unadjusted_epa", "features": ["epa_per_play", "success"], "trained_date": "2026-06-17"}))
    import polars as pl
    (art / "xrepa_loso.parquet").write_text("") if False else pl.DataFrame(
        {"exp_rb_epa": [0.1, 0.2, 0.3, 0.4], "target": [0.1, 0.2, 0.3, 0.4]}).write_parquet(art / "xrepa_loso.parquet")
    out = tmp_path / "docs" / "models"
    rc = main(["--artifacts", str(art), "--out", str(out), "--rb-loso", str(art / "xrepa_loso.parquet")])
    assert rc == 0
    assert (out / "rb_eval.md").exists() and (out / "README.md").exists()
    assert "weighted_r2" in (out / "rb_eval.md").read_text()
```

- [ ] **Step 2: Run to verify it fails**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_cli.py -v`
Expected: FAIL — `cannot import name 'build_parser'`.

- [ ] **Step 3: Implement**
`python/cfb_model_reports/cli.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from .discovery import discover_models
from .metrics import provenance_from_card, rb_eval_metrics, xgb_importance
from .report import ModelReport, render_index, render_model_report

_TITLES = {"ep": "Expected Points (EP)", "wp_spread": "Win Probability (spread)",
           "wp_naive": "Win Probability (naive)", "qbr": "QBR", "cpoe": "CPOE",
           "fourth_down": "Fourth-Down Yards", "rb_eval": "RB Evaluation (xREPA)"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_model_reports")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--cache", default=None, help="warmed final.json cache for prediction-based metrics (optional)")
    ap.add_argument("--out", default="docs/models")
    ap.add_argument("--rb-loso", default=None, help="path to xrepa_loso.parquet (rb_eval metrics)")
    return ap


def _build_report(m, args) -> ModelReport:
    metrics, figures, notes = {}, [], []
    prov = provenance_from_card(m.card)
    if m.model_type == "rb_eval":
        loso = args.rb_loso or str(Path(args.artifacts) / "xrepa_loso.parquet")
        if Path(loso).exists():
            metrics.update(rb_eval_metrics(loso))
        else:
            notes.append("rb_eval LOSO metrics require xrepa_loso.parquet (run `python -m rb_eval train`).")
    elif m.model_path.suffix == ".ubj":
        try:
            metrics["importance_top"] = ", ".join(list(xgb_importance(m.model_path, top_n=8)))
        except Exception as e:  # noqa: BLE001
            notes.append(f"feature importance unavailable: {e}")
        if m.model_type == "qbr":
            notes.append("QBR correlation/RMSE vs ESPN QBR is integration-only (needs the ESPN QBR frame).")
        if args.cache is None:
            notes.append("Calibration + log-loss/Brier require a warmed --cache; run the integration report for those.")
    return ModelReport(model_type=m.model_type, title=_TITLES.get(m.model_type, m.model_type),
                       metrics=metrics, figures=figures, provenance=prov, notes=notes)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    found = discover_models(args.artifacts)
    reports = []
    for m in found:
        r = _build_report(m, args)
        (out / f"{m.model_type}.md").write_text(render_model_report(r), encoding="utf-8")
        reports.append(r)
    (out / "README.md").write_text(render_index(reports), encoding="utf-8")
    print(f"model_reports: wrote {len(reports)} report(s) -> {out} (models: {', '.join(r.model_type for r in reports) or 'none'})")
    return 0
```
`python/cfb_model_reports/__main__.py`:
```python
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**
Run: `cd python && uv run pytest tests/cfb_model_reports/test_cli.py tests/cfb_model_reports -m "not integration" -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git -C "<cfb-data>" add python/cfb_model_reports/cli.py python/cfb_model_reports/__main__.py python/tests/cfb_model_reports/test_cli.py
git -C "<cfb-data>" commit -m "feat(model_reports): CLI to discover models + write per-model Markdown reports"
```

---

## PHASE B — Publish artifacts

### Task 5: `cfb_model_publish` — `gh release upload` artifacts (+ `--dry-run`)

**Files:**
- Create: `python/cfb_model_publish/__init__.py`, `artifacts.py`, `cli.py`, `__main__.py`
- Create: `python/tests/cfb_model_publish/__init__.py`, `python/tests/cfb_model_publish/test_artifacts.py`

**Interfaces:**
- Consumes: `cfb_model_reports.discovery.discover_models`. Produces: `plan_uploads(artifacts_dir) -> list[Path]` (every `.ubj`/`.pkl` + its `.json` card); `upload_artifacts(artifacts_dir, tag, repo, *, dry_run=False, runner=None) -> dict` (`{uploaded, files, tag}`; `dry_run` prints + uploads nothing; `runner` is an injectable `gh` invoker for tests). CLI: `python -m cfb_model_publish artifacts --artifacts <dir> [--dry-run] [--tag espn_cfb_model_artifacts] [--repo sportsdataverse/sportsdataverse-data]`.

- [ ] **Step 1: Write the failing test**
```python
import json
from cfb_model_publish.artifacts import plan_uploads, upload_artifacts


def _seed(tmp_path):
    (tmp_path / "ep.ubj").write_bytes(b"x"); (tmp_path / "ep.json").write_text(json.dumps({"model_type": "ep"}))
    (tmp_path / "xrepa_final.pkl").write_bytes(b"y"); (tmp_path / "xrepa_final.json").write_text(json.dumps({"target": "x"}))
    return tmp_path


def test_plan_uploads_lists_models_and_cards(tmp_path):
    files = {p.name for p in plan_uploads(_seed(tmp_path))}
    assert files == {"ep.ubj", "ep.json", "xrepa_final.pkl", "xrepa_final.json"}


def test_dry_run_uploads_nothing(tmp_path):
    calls = []
    res = upload_artifacts(_seed(tmp_path), "espn_cfb_model_artifacts", "owner/repo",
                           dry_run=True, runner=lambda args: calls.append(args))
    assert res["uploaded"] == 0 and len(res["files"]) == 4 and calls == []


def test_upload_invokes_runner_per_file(tmp_path):
    calls = []
    res = upload_artifacts(_seed(tmp_path), "espn_cfb_model_artifacts", "owner/repo",
                           dry_run=False, runner=lambda args: calls.append(args))
    assert res["uploaded"] == 4 and len(calls) == 4
```

- [ ] **Step 2: Run to verify it fails**
Run: `cd python && uv run pytest tests/cfb_model_publish/test_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError: cfb_model_publish`.

- [ ] **Step 3: Implement**
`python/cfb_model_publish/__init__.py`:
```python
__version__ = "0.1.0"
```
`python/cfb_model_publish/artifacts.py`:
```python
from __future__ import annotations

import subprocess
from pathlib import Path

from cfb_model_reports.discovery import discover_models


def plan_uploads(artifacts_dir) -> list:
    files: list = []
    for m in discover_models(artifacts_dir):
        files.append(m.model_path)
        files.append(m.card_path)
    # de-dup, stable order
    seen, out = set(), []
    for p in files:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def _gh_runner(args: list) -> None:
    subprocess.run(["gh", *args], check=True)


def upload_artifacts(artifacts_dir, tag: str, repo: str, *, dry_run: bool = False, runner=None) -> dict:
    run = runner or _gh_runner
    files = plan_uploads(artifacts_dir)
    uploaded = 0
    for f in files:
        if dry_run:
            print(f"[dry-run] would upload {f} -> {repo}:{tag}")
            continue
        run(["release", "upload", tag, str(f), "--repo", repo, "--clobber"])
        uploaded += 1
    return {"uploaded": uploaded, "files": [str(f) for f in files], "tag": tag}
```
`python/cfb_model_publish/cli.py`:
```python
from __future__ import annotations

import argparse

from .artifacts import upload_artifacts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_model_publish")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("artifacts")
    a.add_argument("--artifacts", required=True)
    a.add_argument("--tag", default="espn_cfb_model_artifacts")
    a.add_argument("--repo", default="sportsdataverse/sportsdataverse-data")
    a.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "artifacts":
        res = upload_artifacts(args.artifacts, args.tag, args.repo, dry_run=args.dry_run)
        print(f"publish: uploaded={res['uploaded']} files={len(res['files'])} -> {args.repo}:{res['tag']}"
              + (" (dry-run)" if args.dry_run else ""))
    return 0
```
`python/cfb_model_publish/__main__.py`:
```python
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**
Run: `cd python && uv run pytest tests/cfb_model_publish -m "not integration" -v`
Expected: PASS (create `python/tests/cfb_model_publish/__init__.py` empty first).

- [ ] **Step 5: Commit**
```bash
git -C "<cfb-data>" add python/cfb_model_publish python/tests/cfb_model_publish
git -C "<cfb-data>" commit -m "feat(model_publish): gh-release artifact uploader with --dry-run"
```

---

## PHASE C — Dataset publish (R) + tag registration

### Task 6: `espn_cfb_16_model_pbp.R` + register tags

**Files:**
- Create: `R/espn_cfb_16_model_pbp.R`
- Modify: `R/releases_init.R` (append two tags)

**Interfaces:**
- Consumes: `R/_data_utils.R` `write_dataset(df, dataset, season, stem)` + `publish_dataset(dataset, season, stem, tag)`. Produces: a script that, given `--parquet <path> --season <year>`, reads the Python model-PBP parquet, calls `write_dataset(df, "model_pbp", season, "model_pbp")` then `publish_dataset("model_pbp", season, "model_pbp", "espn_cfb_model_pbp")` (publish gated behind `CFB_PUBLISH=1`).

- [ ] **Step 1: Write `R/espn_cfb_16_model_pbp.R`** (read the existing `_data_utils.R` first to match arg/style conventions)
```r
#!/usr/bin/env Rscript
# Publish the Python-built model-PBP parquet as the espn_cfb_model_pbp dataset.
# Converts to parquet/rds/gzipped-csv parity via the shared write_dataset() writer.
# Does NOT re-run any Python model code; reads the pre-built parquet from --parquet.
suppressPackageStartupMessages({
  library(arrow); library(optparse); library(cli)
})
if (!exists("write_dataset")) source("R/_data_utils.R")

opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
  optparse::make_option(c("-p", "--parquet"), type = "character",
                        help = "Path to the Python-built model-PBP parquet file"),
  optparse::make_option(c("-s", "--season"),  type = "integer",
                        help = "CFB season year (e.g. 2024)"))))

if (is.null(opt$parquet) || is.null(opt$season)) {
  cli::cli_abort("Both --parquet <path> and --season <year> are required.")
}

cli::cli_alert_info("Reading model-PBP parquet: {opt$parquet}")
df <- as.data.frame(arrow::read_parquet(opt$parquet))
cli::cli_alert_info("model_pbp {opt$season}: {nrow(df)} rows, {ncol(df)} cols")

write_dataset(df, "model_pbp", opt$season, "model_pbp")

if (identical(Sys.getenv("CFB_PUBLISH"), "1")) {
  publish_dataset("model_pbp", opt$season, "model_pbp", "espn_cfb_model_pbp")
  cli::cli_alert_success("Published model_pbp {opt$season} -> espn_cfb_model_pbp")
} else {
  cli::cli_alert_info("Wrote model_pbp {opt$season} locally (set CFB_PUBLISH=1 to upload)")
}
```
(Snippet matches the shipped `R/espn_cfb_16_model_pbp.R` exactly — uses `if (!exists("write_dataset")) source("R/_data_utils.R")` instead of the `sys.frame`/`%||%` pattern.)

- [ ] **Step 2: Register the tags** — in `R/releases_init.R`, append `"espn_cfb_model_pbp"` and `"espn_cfb_model_artifacts"` to the tags vector that the script iterates (match the existing list's exact formatting).

- [ ] **Step 3: Verify the R script parses + tags registered**
Run: `Rscript -e "invisible(parse('R/espn_cfb_16_model_pbp.R')); invisible(parse('R/releases_init.R')); cat('R parse ok\n')"`
Expected: `R parse ok`. (If `Rscript` is unavailable in the environment, instead `grep -q espn_cfb_model_pbp R/releases_init.R && grep -q espn_cfb_model_artifacts R/releases_init.R && echo tags-ok` → `tags-ok`, and note R wasn't run.)

- [ ] **Step 4: Commit**
```bash
git -C "<cfb-data>" add R/espn_cfb_16_model_pbp.R R/releases_init.R
git -C "<cfb-data>" commit -m "feat(R): publish model-PBP dataset (espn_cfb_model_pbp) + register model release tags"
```

---

## PHASE D — Gate

### Task 7: SP2 green gate + acceptance run

**Files:** Test the whole subsystem + a manual report/publish run.

- [ ] **Step 1: Default suite green**
Run: `cd python && uv run pytest -m "not integration" -q`
Expected: PASS — SP1 suite + the new `cfb_model_reports` + `cfb_model_publish` tests; integration deselected; 0 errors. If any failure, STOP + report (do not weaken tests).

- [ ] **Step 2: Manual report generation over a synthetic artifacts dir (offline)**
```bash
cd python
mkdir -p /tmp/sp2art && python -c "import json,polars as pl; open('/tmp/sp2art/xrepa_final.pkl','wb').write(b'x'); open('/tmp/sp2art/xrepa_final.json','w').write(json.dumps({'target':'unadjusted_epa','features':['epa_per_play','success'],'trained_date':'2026-06-17'})); pl.DataFrame({'exp_rb_epa':[0.1,0.2,0.3,0.4],'target':[0.1,0.2,0.3,0.4]}).write_parquet('/tmp/sp2art/xrepa_loso.parquet')"
uv run python -m cfb_model_reports --artifacts /tmp/sp2art --out /tmp/sp2docs
uv run python -m cfb_model_publish artifacts --artifacts /tmp/sp2art --dry-run
```
Expected: `/tmp/sp2docs/rb_eval.md` + `README.md` written with `weighted_r2`; the publish prints `[dry-run] would upload …` for the `.pkl` + `.json` and `uploaded=0 … (dry-run)`. (A real-models run over SP1's `python/artifacts/` is the integration path.)

- [ ] **Step 3: Commit checkpoint**
```bash
git -C "<cfb-data>" commit --allow-empty -m "chore(model_reports): SP2 complete — reports + artifact/dataset publish capability, green offline"
```

---

## Self-Review

**1. Spec coverage** (spec §2/§4/§5):
- D1 publish dataset+artifacts+reports → Phase A (reports) + Phase B (artifacts) + Phase C (dataset). ✓
- D2 Markdown under docs/models/ → Task 1 renderer + Task 4 CLI writes docs/models/*.md. ✓
- D3 all-models one tag → Task 5 globs all models, uploads to espn_cfb_model_artifacts. ✓
- D4 Python owns reports/artifacts; R for dataset → Phases A/B Python, Phase C R. ✓
- D5 R writer for model-PBP parity → Task 6. ✓
- D6 run-locally + CI deferred → Task 7 acceptance is a local run; no workflow added. ✓
- Discovery-by-card rule (Global Constraints) → Task 2. ✓
- Error handling (missing artifact skip, --dry-run, insufficient-data notes) → Task 2 skip, Task 5 dry-run, Task 4 notes. ✓
- R1 (validation source) — resolved: rb_eval reads its loso parquet; xgb importance offline; classification/calibration metrics are integration-only (noted in the report). R2 (artifact layout) — resolved via card-glob discovery (name-agnostic). R3 (QBR) — QBR correlation is an integration note. R4 (figure paths) — figures dir + repo-relative embeds in Task 1/4.

**2. Placeholder scan:** every code/test step has real code/commands. The R `source()`/`write_dataset` signature carries a "read `_data_utils.R` first to match exactly" instruction (the implementer confirms the exact arg names against the file) — this is a deliberate adapt-to-existing-API step, not a placeholder; the surrounding logic + the publish call are concrete.

**3. Type consistency:** `ModelReport`/`render_model_report`/`render_index` (Task 1) used by Task 4; `discover_models`/`ModelArtifact` (Task 2) used by Tasks 4 + 5; `provenance_from_card`/`rb_eval_metrics`/`xgb_importance` (Task 3) used by Task 4; `plan_uploads`/`upload_artifacts` (Task 5) consistent across its test + CLI. ✓

## Notes for execution
- Figures: the offline report path (Task 4) embeds figures only when a `--cache` prediction pass produces them; the default acceptance run is metrics+provenance+notes. Wiring the per-track `figures.py` (write_calibration / write_fd_figures / write_xrepa_calibration) into `docs/models/figures/` is done in the integration report path (a follow-up within SP2's integration runs), since it needs real models + the warmed cache — keep that work behind `-m integration`.
- The committed `docs/models/*.md` from a real-models run is produced during the acceptance/integration run, not by a unit test.
