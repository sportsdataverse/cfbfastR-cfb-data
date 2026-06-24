"""Summarize the era experiment: canonical vs spread-backfilled, keep/drop verdicts.

Reads the two results JSONs written by ``era_experiment`` and emits a single
markdown report:

* ``artifacts/era_results.json``            — baseline-vs-era on the canonical frame
  (spread-independent models are authoritative here: ep, fg, xpass, wp_naive,
  cpoe, rb_eval).
* ``artifacts/era_results_backfilled.json`` — baseline-vs-era on the
  spread-backfilled frame (authoritative for the spread-dependent models:
  wp_spread, two_pt, fourth_down, qbr).

For the spread-dependent models the report also surfaces the *backfill* effect
(canonical-baseline vs backfilled-baseline) alongside the *era* effect, so the two
interventions are not conflated.

The keep rule: era is KEPT for a model only when its authoritative-frame headline
metric improves out-of-fold (``era_wins == True``), with the magnitude reported so
marginal/noise-level wins are visible rather than rubber-stamped.

Run::

    python -m model_training.era_report --artifacts artifacts --out artifacts/era_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Which frame is authoritative for each model's final verdict.
SPREAD_DEPENDENT = {"wp_spread", "two_pt", "fourth_down", "qbr"}
SPREAD_INDEPENDENT = {"ep", "fg", "xpass", "wp_naive", "cpoe", "rb_eval"}
_ORDER = ["ep", "wp_spread", "wp_naive", "qbr", "cpoe", "fg", "xpass", "two_pt",
          "fourth_down", "rb_eval", "pregame_wp"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))


def _row(model: str, payload: dict, frame: str) -> dict | None:
    if not payload or payload.get("status") != "done":
        return None
    v = payload.get("verdict")
    if not v:
        return {"model": model, "frame": frame, "status": payload.get("status", "?"),
                "headline": "-", "baseline": "", "era": "", "delta": "", "wins": payload.get("status")}
    return {
        "model": model, "frame": frame, "headline": v["headline"],
        "baseline": _fmt(v["baseline"]), "era": _fmt(v["era"]), "delta": _fmt(v["delta"]),
        "delta_raw": v["delta"], "wins": v["era_wins"], "secondary": payload.get("secondary", {}),
    }


# Minimum out-of-fold delta (per headline metric) below which a "win" is fold noise,
# not a real effect. Calibrated to the spread of per-season fold variation.
_MATERIAL = {"logloss": 0.001, "mlogloss": 0.005, "rmse": 0.10, "ep_cal_mae": 0.0,
             "first_down_cal_mae": 0.0005, "weighted_cal_err": 0.001, "weighted_r2": 0.01}


def _classify(headline: str | None, delta_raw, wins) -> str:
    """material | neutral | no | pending."""
    if wins is None:
        return "pending"
    if not wins or delta_raw is None:
        return "no"
    return "material" if abs(delta_raw) >= _MATERIAL.get(headline, 0.0) else "neutral"


def build_report(canon: dict, back: dict) -> str:
    lines: list[str] = ["# CFB era-variable experiment — results", ""]
    lines += [
        "Honest leave-one-season-out (LOSO) out-of-fold comparison of each model's",
        "shipped recipe **with vs without** the one-hot rule-era dummies (era0..era3,",
        "cuts 2006/2013/2020 → 2004-06 / 2007-13 / 2014-20 / 2021+). Spread-dependent",
        "models are evaluated on the odds-backfilled frame; the rest on the canonical frame.",
        "Era is kept only where the authoritative-frame headline metric improves out-of-fold.",
        "",
        "| Model | Auth. frame | Headline | Baseline | +era | Δ | era kept? |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    decisions: dict[str, dict] = {}
    for m in _ORDER:
        auth = "backfilled" if m in SPREAD_DEPENDENT else "canonical"
        src = back if m in SPREAD_DEPENDENT else canon
        r = _row(m, src.get(m, {}), auth)
        if r is None:
            # fall back to whichever frame has it
            other = canon if m in SPREAD_DEPENDENT else back
            r = _row(m, other.get(m, {}), "canonical" if src is back else "backfilled") or \
                {"model": m, "frame": "-", "headline": "(pending)", "baseline": "", "era": "",
                 "delta": "", "wins": None}
        cls = _classify(r.get("headline"), r.get("delta_raw"), r.get("wins"))
        kept_str = {"material": "✅ keep (material)", "neutral": "~ neutral (noise)",
                    "no": "— no", "pending": "(pending)"}[cls]
        lines.append(f"| {m} | {r['frame']} | {r.get('headline','-')} | {r['baseline']} | "
                     f"{r['era']} | {r['delta']} | {kept_str} |")
        decisions[m] = {"frame": r["frame"], "headline": r.get("headline"), "delta": r.get("delta"),
                        "cls": cls}

    # Backfill effect on the spread-dependent models (baseline canonical vs backfilled).
    lines += ["", "## Spread-backfill effect (baseline, canonical vs backfilled frame)", "",
              "Isolates the odds-backfill from the era change for the spread-dependent models.", "",
              "| Model | Headline | Canonical baseline | Backfilled baseline | Δ |",
              "|---|---|---:|---:|---:|"]
    for m in sorted(SPREAD_DEPENDENT):
        cb = canon.get(m, {}).get("verdict", {}) or {}
        bb = back.get(m, {}).get("verdict", {}) or {}
        if cb.get("baseline") is None or bb.get("baseline") is None:
            lines.append(f"| {m} | {cb.get('headline','-')} | {_fmt(cb.get('baseline'))} | "
                         f"{_fmt(bb.get('baseline'))} | (pending) |")
            continue
        d = bb["baseline"] - cb["baseline"]
        lines.append(f"| {m} | {cb['headline']} | {cb['baseline']:.4f} | {bb['baseline']:.4f} | {d:+.4f} |")

    material = [m for m, d in decisions.items() if d["cls"] == "material"]
    neutral = [m for m, d in decisions.items() if d["cls"] == "neutral"]
    nope = [m for m, d in decisions.items() if d["cls"] == "no"]
    pend = [m for m, d in decisions.items() if d["cls"] == "pending"]
    # Derive the headline from the computed verdict so it can't drift from the table.
    wp_cb = (canon.get("wp_spread", {}).get("verdict", {}) or {}).get("baseline")
    wp_bb = (back.get("wp_spread", {}).get("verdict", {}) or {}).get("baseline")
    head = (
        f"era is a *material* OOF win for **{', '.join(material)}**"
        if material
        else "era is not a *material* OOF win for any model"
    ) + "; elsewhere it is noise-level or a calibration regression."
    if wp_cb is not None and wp_bb is not None:
        head += (
            f" The **spread backfill** moves **wp_spread** baseline by "
            f"{wp_bb - wp_cb:+.4f} — independent of era."
        )
    lines += ["", "## Verdict", "",
              f"- **Keep era — material OOF gain:** {', '.join(material) or 'none'}",
              f"- **Neutral — era wins only at noise level (keep shipped recipe, era optional):** "
              f"{', '.join(neutral) or 'none'}",
              f"- **Drop era — no OOF gain (or calibration regression):** {', '.join(nope) or 'none'}",
              f"- **Not evaluated:** {', '.join(pend) or 'none'} "
              "(rb_eval: local cache too small; pregame_wp: single-feature pregame model)", "",
              f"**Headline:** {head}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="model_training.era_report")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--canonical", default="era_results.json")
    ap.add_argument("--backfilled", default="era_results_backfilled.json")
    ap.add_argument("--out", default="artifacts/era_report.md")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)
    report = build_report(_load(art / args.canonical), _load(art / args.backfilled))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
