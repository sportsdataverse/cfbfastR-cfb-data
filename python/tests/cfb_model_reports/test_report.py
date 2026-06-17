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


def test_render_index_links_each_report():
    rs = [ModelReport("ep", "EP", {}, [], {}, []), ModelReport("cpoe", "CPOE", {}, [], {}, [])]
    idx = render_index(rs)
    assert "[EP](ep.md)" in idx and "[CPOE](cpoe.md)" in idx


def test_fmt_cell_list():
    assert _fmt_cell(["a", "b"]) == "a, b"


def test_fmt_cell_none():
    assert _fmt_cell(None) == "n/a"


def test_fmt_cell_pipe_escaped():
    assert _fmt_cell("has|pipe") == "has\\|pipe"
