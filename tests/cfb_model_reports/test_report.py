from cfb_model_reports.report import ModelReport, render_model_report, render_index, _fmt_cell


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


def test_render_model_report_renders_prose_sections():
    """Overview / Recipe & lineage / Discussion / Limitations render when set."""
    r = ModelReport(
        model_type="ep", title="Expected Points (EP)",
        metrics={"ep_cal_mae": 0.014}, figures=["figures/ep_calibration.png"],
        provenance={"trained_date": "2026-06-17"}, notes=[],
        summary="EP estimates next-score value.",
        recipe="8-feature multiclass softprob.",
        discussion="LOSO pooled mlogloss 1.2333.",
        limitations="EP is a start-of-play quantity.",
    )
    md = render_model_report(r)
    assert "## Overview" in md and "EP estimates next-score value." in md
    assert "## Recipe & lineage" in md and "softprob" in md
    assert "## Discussion" in md and "1.2333" in md
    assert "## Limitations" in md and "start-of-play" in md
    # Section ordering: Overview before Metrics before Discussion before Provenance.
    assert md.index("## Overview") < md.index("## Metrics") < md.index("## Discussion") < md.index("## Provenance")


def test_render_model_report_omits_empty_prose():
    """A report with no prose still renders metrics/provenance (back-compat)."""
    r = ModelReport("x", "X", {"a": 1}, [], {})
    md = render_model_report(r)
    assert "## Overview" not in md and "## Discussion" not in md
    assert "## Metrics" in md and "## Provenance" in md


def test_render_index_links_each_report():
    rs = [ModelReport("ep", "EP", {}, [], {}, []), ModelReport("cpoe", "CPOE", {}, [], {}, [])]
    idx = render_index(rs)
    assert "[EP](ep.md)" in idx and "[CPOE](cpoe.md)" in idx


def test_render_methodology_sections_and_figure_routing():
    """nflfastR-post structure: Model features / The model / Calibration Results
    (with the calibration figure) / Feature importance (with its figure)."""
    r = ModelReport(
        model_type="ep", title="Expected Points (EP)",
        metrics={"ep_cal_mae": 0.014},
        figures=["figures/ep_class_calibration.png", "figures/ep_importance.png"],
        provenance={"trained_date": "2026-06-17"}, notes=[],
        summary="EP estimates next-score value.",
        recipe="8-feature multiclass softprob.",
        discussion="Binned predicted class prob vs empirical rate.",
        limitations="Start-of-play quantity.",
        features="| Feature | Type | What it encodes |\n|---|---|---|\n| `x` | n | y |",
        model="XGBoost multi:softprob, 525 rounds, leave-one-season-out CV.",
        importance="yards_to_goal dominates by gain.",
        calibration_figures=["figures/ep_class_calibration.png"],
        importance_figures=["figures/ep_importance.png"],
    )
    md = render_model_report(r)
    for h in ("## Model features", "## The model", "## Calibration Results",
              "## Feature importance"):
        assert h in md, f"missing section {h}"
    # The calibration figure renders under Calibration Results, the importance
    # figure under Feature importance.
    assert "![](figures/ep_class_calibration.png)" in md
    assert "![](figures/ep_importance.png)" in md
    # Ordering: features -> the model -> metrics -> calibration -> importance -> limitations.
    order = [
        "## Model features", "## The model", "## Metrics", "## Calibration Results",
        "## Feature importance", "## Limitations", "## Provenance",
    ]
    idxs = [md.index(h) for h in order]
    assert idxs == sorted(idxs), f"section order wrong: {idxs}"
    # A routed figure is NOT duplicated into a generic Figures section.
    assert "## Figures" not in md


def test_fmt_cell_list():
    assert _fmt_cell(["a", "b"]) == "a, b"


def test_fmt_cell_none():
    assert _fmt_cell(None) == "n/a"


def test_fmt_cell_pipe_escaped():
    assert _fmt_cell("has|pipe") == "has\\|pipe"
