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
    summary: str = ""
    recipe: str = ""
    discussion: str = ""
    limitations: str = ""


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

    Section order: Overview, Recipe & lineage, Metrics, Figures, Discussion,
    Limitations, Provenance, Notes. Prose sections are omitted when empty so a
    minimal (metrics-only) report still renders.

    Args:
        r: the :class:`ModelReport` to render.

    Returns:
        A Markdown document string.
    """
    parts = [f"# {r.title}\n"]
    if r.summary:
        parts += ["## Overview\n", r.summary + "\n"]
    if r.recipe:
        parts += ["\n## Recipe & lineage\n", r.recipe + "\n"]
    parts += ["\n## Metrics\n", _metrics_table(r.metrics)]
    if r.figures:
        parts.append("\n## Figures\n")
        parts += [f"![]({p})\n" for p in r.figures]
    if r.discussion:
        parts += ["\n## Discussion\n", r.discussion + "\n"]
    if r.limitations:
        parts += ["\n## Limitations\n", r.limitations + "\n"]
    parts.append("\n## Provenance\n")
    parts.append(_metrics_table(r.provenance) if r.provenance else "_n/a_\n")
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
