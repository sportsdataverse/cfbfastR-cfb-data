from __future__ import annotations

import argparse
import re
from pathlib import Path

from .discovery import discover_models
from .metrics import provenance_from_card, rb_eval_metrics, xgb_importance
from .report import ModelReport, render_index, render_model_report

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_model_stem(model_type: str) -> str:
    stem = _SAFE_STEM.sub("_", model_type).strip("._")
    return stem or "model"

_TITLES = {
    "ep": "Expected Points (EP)",
    "wp_spread": "Win Probability (spread)",
    "wp_naive": "Win Probability (naive)",
    "qbr": "QBR",
    "cpoe": "CPOE",
    "fourth_down": "Fourth-Down Yards",
    "rb_eval": "RB Evaluation (xREPA)",
}


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
            imp = xgb_importance(m.model_path, top_n=8)
            metrics["importance_top"] = ", ".join(f"{k}:{v}" for k, v in imp.items())
        except Exception as e:  # noqa: BLE001
            notes.append(f"feature importance unavailable: {e}")
        if m.model_type == "qbr":
            notes.append("QBR correlation/RMSE vs ESPN QBR is integration-only (needs the ESPN QBR frame).")
        if args.cache is None:
            notes.append("Calibration + log-loss/Brier require a warmed --cache; run the integration report for those.")
    return ModelReport(
        model_type=m.model_type,
        title=_TITLES.get(m.model_type, m.model_type),
        metrics=metrics,
        figures=figures,
        provenance=prov,
        notes=notes,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    found = discover_models(args.artifacts)
    reports = []
    for m in found:
        r = _build_report(m, args)
        safe_stem = _safe_model_stem(m.model_type)
        report_path = (out / f"{safe_stem}.md").resolve()
        if out.resolve() != report_path.parent:
            raise ValueError(f"Unsafe model_type for output path: {m.model_type}")
        r = r.__class__(
            model_type=safe_stem,
            title=r.title,
            metrics=r.metrics,
            figures=r.figures,
            provenance=r.provenance,
            notes=r.notes,
        )
        report_path.write_text(render_model_report(r), encoding="utf-8")
        reports.append(r)
    (out / "README.md").write_text(render_index(reports), encoding="utf-8")
    print(f"model_reports: wrote {len(reports)} report(s) -> {out} (models: {', '.join(r.model_type for r in reports) or 'none'})")
    return 0
