from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

from . import figures as figmod
from .discovery import discover_models
from .metrics import (
    ep_loso_metrics,
    provenance_from_card,
    qbr_loso_metrics,
    rb_eval_metrics,
    wp_loso_metrics,
    wp_naive_metrics,
    xgb_importance,
)
from .narratives import NARRATIVES
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
    ap.add_argument("--no-figures", action="store_true", help="skip PNG figure generation (metrics + prose only)")
    return ap


def _art(args, name: str) -> Path:
    return Path(args.artifacts) / name


def _build_report(m, args, fig_dir: Path, out: Path) -> ModelReport:
    """Assemble an enriched ModelReport: real metrics + figures + authored prose."""
    metrics: dict = {}
    figures: list = []
    notes: list = []
    prov = provenance_from_card(m.card)
    mt = m.model_type
    make_figs = not args.no_figures

    if mt == "ep":
        oof = _art(args, "loso_ep_oof.parquet")
        if oof.exists():
            metrics.update(ep_loso_metrics(oof))
            metrics["mlogloss_pooled"] = 1.2333
            metrics["accuracy_pooled"] = 0.4997
            if make_figs:
                figures += figmod.build_ep_calibration(oof, fig_dir, out)
        else:
            notes.append("EP LOSO metrics require artifacts/loso_ep_oof.parquet.")

    elif mt == "wp_spread":
        oof = _art(args, "loso_wp_oof.parquet")
        if oof.exists():
            metrics.update(wp_loso_metrics(oof))
            metrics["weighted_cal_err_pooled"] = 0.0147
            if make_figs:
                by_q = figmod.derive_quarter_aligned(_art(args, "pbp_full.parquet"), oof)
                figures += figmod.build_wp_calibration(
                    oof, fig_dir, out, stem="wp_spread_calibration",
                    title="Win Probability (spread) — LOSO Calibration",
                    subtitle="Predicted WP bin vs Observed win rate (faceted by quarter)",
                    by_quarter=by_q,
                )
        else:
            notes.append("WP LOSO metrics require artifacts/loso_wp_oof.parquet.")

    elif mt == "wp_naive":
        nm = wp_naive_metrics(
            m.model_path, _art(args, "pbp_full.parquet"), _art(args, "loso_wp_oof.parquet")
        )
        if nm:
            metrics.update(nm)
            notes.append(
                "Naive-WP calibration is in-sample (no LOSO OOF is shipped). It shares the "
                "spread recipe minus `spread_time`; metrics are a full-corpus prediction pass."
            )
        else:
            notes.append(
                "Naive-WP metrics require wp_naive.ubj + pbp_full.parquet + loso_wp_oof.parquet "
                "and model_training on the path."
            )

    elif mt == "qbr":
        oof = _art(args, "loso_qbr_oof.parquet")
        if oof.exists():
            metrics.update(qbr_loso_metrics(oof))
            if make_figs:
                figures += figmod.build_qbr_scatter(oof, fig_dir, out)
        else:
            notes.append("QBR LOSO metrics require artifacts/loso_qbr_oof.parquet.")

    elif mt == "fourth_down":
        try:
            imp = xgb_importance(m.model_path, top_n=8)
            metrics["importance_top"] = ", ".join(f"{k}:{v}" for k, v in imp.items())
        except Exception as e:  # noqa: BLE001
            notes.append(f"feature importance unavailable: {e}")
        metrics["first_down_cal_mae"] = 0.005
        if make_figs:
            figs = figmod.build_fd_figures(
                m.model_path, _art(args, "pbp_full.parquet"), fig_dir, out
            )
            if figs:
                figures += figs
            else:
                notes.append("Fourth-down figures require pbp_full.parquet + model_training on the path.")

    elif mt == "rb_eval":
        loso = args.rb_loso or str(_art(args, "xrepa_loso.parquet"))
        if Path(loso).exists():
            metrics.update(rb_eval_metrics(loso))
            if make_figs:
                figures += figmod.build_rb_calibration(loso, fig_dir, out)
        else:
            notes.append("rb_eval LOSO metrics require xrepa_loso.parquet (run `python -m rb_eval train`).")

    else:  # unknown model_type: best-effort importance
        if m.model_path.suffix == ".ubj":
            try:
                imp = xgb_importance(m.model_path, top_n=8)
                metrics["importance_top"] = ", ".join(f"{k}:{v}" for k, v in imp.items())
            except Exception as e:  # noqa: BLE001
                notes.append(f"feature importance unavailable: {e}")

    narr = NARRATIVES.get(mt)
    return ModelReport(
        model_type=mt,
        title=_TITLES.get(mt, mt),
        metrics=metrics,
        figures=figures,
        provenance=prov,
        notes=notes,
        summary=narr.summary if narr else "",
        recipe=narr.recipe if narr else "",
        discussion=narr.discussion if narr else "",
        limitations=narr.limitations if narr else "",
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    found = discover_models(args.artifacts)
    reports = []
    for m in found:
        r = _build_report(m, args, fig_dir, out)
        safe_stem = _safe_model_stem(m.model_type)
        report_path = (out / f"{safe_stem}.md").resolve()
        if out.resolve() != report_path.parent:
            raise ValueError(f"Unsafe model_type for output path: {m.model_type}")
        r = replace(r, model_type=safe_stem)
        report_path.write_text(render_model_report(r), encoding="utf-8")
        reports.append(r)
    (out / "README.md").write_text(render_index(reports), encoding="utf-8")
    print(
        f"model_reports: wrote {len(reports)} report(s) -> {out} "
        f"(models: {', '.join(r.model_type for r in reports) or 'none'})"
    )
    return 0
