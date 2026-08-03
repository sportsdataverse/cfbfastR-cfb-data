import json

import polars as pl

from cfb_model_reports.cli import build_parser, main


def test_parser_defaults():
    ns = build_parser().parse_args(["--artifacts", "a", "--out", "docs/models"])
    assert ns.artifacts == "a" and ns.out == "docs/models"


def test_main_writes_reports_and_index(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    # rb_eval: card + pkl + a loso parquet (rb_eval_metrics path, no xgboost needed)
    (art / "xrepa_final.pkl").write_bytes(b"y")
    (art / "xrepa_final.json").write_text(
        json.dumps(
            {
                "target": "unadjusted_epa",
                "features": ["epa_per_play", "success"],
                "trained_date": "2026-06-17",
            }
        )
    )
    pl.DataFrame(
        {"exp_rb_epa": [0.1, 0.2, 0.3, 0.4], "target": [0.1, 0.2, 0.3, 0.4]}
    ).write_parquet(art / "xrepa_loso.parquet")
    out = tmp_path / "docs" / "models"
    rc = main(["--artifacts", str(art), "--out", str(out), "--no-figures",
               "--rb-loso", str(art / "xrepa_loso.parquet")])
    assert rc == 0
    assert (out / "rb_eval.md").exists() and (out / "README.md").exists()
    body = (out / "rb_eval.md").read_text()
    assert "weighted_r2" in body
    # Enriched prose sections injected from narratives for rb_eval.
    assert "## Overview" in body and "## Recipe & lineage" in body
    assert "## Discussion" in body and "## Limitations" in body
    assert "xREPA" in body


def test_no_figures_flag_parses():
    ns = build_parser().parse_args(["--artifacts", "a", "--no-figures"])
    assert ns.no_figures is True
