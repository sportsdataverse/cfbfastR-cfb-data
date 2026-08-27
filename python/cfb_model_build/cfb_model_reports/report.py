from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ModelReport:
    model_type: str
    title: str
    metrics: dict
    figures: list  # repo-relative paths, e.g. "figures/cpoe_calibration.png"
    provenance: dict
    notes: list = field(default_factory=list)
    # Authored prose sections (Markdown). Empty string => section omitted.
    summary: str = ""          # -> "## Overview"
    recipe: str = ""           # -> "## Recipe & lineage" (kept for back-compat)
    discussion: str = ""       # -> "## Discussion" (calibration prose)
    limitations: str = ""      # -> "## Limitations"
    # Enriched methodology sections (nflfastR-post structure). Optional.
    features: str = ""         # -> "## Model features"
    model: str = ""            # -> "## The model"
    calibration_caption: str = ""  # prose under the calibration figure
    importance: str = ""       # -> "## Feature importance"
    # Figure routing: calibration figures render under "Calibration Results";
    # importance figures under "Feature importance"; the rest under "Figures".
    calibration_figures: list = field(default_factory=list)
    importance_figures: list = field(default_factory=list)


def _fmt_cell(v) -> str:
    """Format a metric/provenance value for a Markdown table cell."""
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)):
        result = ", ".join(str(x) for x in v)
    elif isinstance(v, dict):
        result = json.dumps(v, separators=(",", ":"))
    else:
        result = str(v)
    return result.replace("|", "\\|")


def _metrics_table(metrics: dict) -> str:
    if not metrics:
        return "_No metrics available._\n"
    rows = "\n".join(f"| `{k}` | {_fmt_cell(v)} |" for k, v in metrics.items())
    return f"| metric | value |\n|---|---|\n{rows}\n"


def render_model_report(r: ModelReport) -> str:
    """Render an enriched per-model report as Markdown.

    Section order mirrors the canonical nflfastR EP/WP/CP methodology post:
    Overview -> Model features -> The model -> Calibration Results -> Feature
    importance -> Limitations -> Provenance -> Notes. ``Recipe & lineage``,
    ``Metrics`` and ``Discussion`` are retained (folded into "The model" /
    "Calibration Results") so older minimal reports and tests still render.
    Prose sections are omitted when empty.

    Args:
        r: the :class:`ModelReport` to render.

    Returns:
        A Markdown document string.
    """
    parts = [f"# {r.title}\n"]

    # --- Overview / About -----------------------------------------------------
    if r.summary:
        parts += ["## Overview\n", r.summary + "\n"]

    # --- Model features -------------------------------------------------------
    if r.features:
        parts += ["\n## Model features\n", r.features + "\n"]

    # --- The model (algorithm + recipe + pooled metrics) ----------------------
    if r.recipe:
        parts += ["\n## Recipe & lineage\n", r.recipe + "\n"]
    if r.model:
        parts += ["\n## The model\n", r.model + "\n"]
    parts += ["\n## Metrics\n", _metrics_table(r.metrics)]

    # --- Calibration Results (the centerpiece) --------------------------------
    cal_figs = r.calibration_figures or []
    if cal_figs or r.discussion or r.calibration_caption:
        parts.append("\n## Calibration Results\n")
        for p in cal_figs:
            parts.append(f"![]({p})\n")
        if r.discussion:
            parts.append("\n## Discussion\n")
            parts.append(r.discussion + "\n")
        if r.calibration_caption:
            parts.append(r.calibration_caption + "\n")

    # --- Feature importance ---------------------------------------------------
    if r.importance or r.importance_figures:
        parts.append("\n## Feature importance\n")
        if r.importance:
            parts.append(r.importance + "\n")
        for p in r.importance_figures:
            parts.append(f"![]({p})\n")

    # --- Any remaining (unrouted) figures -------------------------------------
    routed = set(cal_figs) | set(r.importance_figures or [])
    other = [p for p in r.figures if p not in routed]
    if other:
        parts.append("\n## Figures\n")
        parts += [f"![]({p})\n" for p in other]

    # --- Limitations ----------------------------------------------------------
    if r.limitations:
        parts += ["\n## Limitations\n", r.limitations + "\n"]

    # --- Provenance -----------------------------------------------------------
    parts.append("\n## Provenance\n")
    parts.append(_metrics_table(r.provenance) if r.provenance else "_n/a_\n")

    # --- Notes ----------------------------------------------------------------
    if r.notes:
        parts.append("\n## Notes\n")
        parts += [f"- {n}\n" for n in r.notes]
    return "\n".join(parts)


def render_index(reports: list) -> str:
    lines = ["# CFB Model Reports\n", "Per-model metrics, calibration, and provenance.\n", "## Models\n"]
    for r in reports:
        blurb = ""
        summary = getattr(r, "summary", "")
        if summary:
            first = summary.split(". ")[0].rstrip(".")
            blurb = f" — {first}."
        lines.append(f"- [{r.title}]({r.model_type}.md){blurb}")
    return "\n".join(lines) + "\n"
