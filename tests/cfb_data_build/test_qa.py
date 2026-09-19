"""Offline tests for the report-only QA stage (``cfb_data_build.qa``).

Real inputs only: the vendored ``final_401628455.json`` (a complete 2024 game)
for the per-game gate, and the real pbp frame cut from it for the drift gate.
No network -- the published comparison frame is passed in, never downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cfb_data_build import qa
from cfb_data_build.build import build_dataset_frame, qa_sidecar
from cfb_data_build.config import REGISTRY

FIX = Path(__file__).parent / "fixtures"
GID = 401628455


def _game() -> dict:
    return json.loads((FIX / f"final_{GID}.json").read_text(encoding="utf-8"))


def test_qa_row_shape_and_identity():
    df = build_dataset_frame(REGISTRY["qa"], _game())
    assert df.height == 1
    assert set(qa.QA_SCHEMA) <= set(df.columns)
    row = df.row(0, named=True)
    assert row["game_id"] == GID and row["season"] == 2024
    assert row["league"] == "cfb" and row["source"] == "espn"
    assert row["processing_version"] == _game().get("processing_version")
    assert row["n_rows"] > 100
    assert row["ok"] == (row["n_errors"] == 0)
    # flat strings, so parquet / csv / rds share one schema
    assert df.schema["failed_rule_ids"] == pl.Utf8
    assert df.schema["warned_rule_ids"] == pl.Utf8


def test_qa_row_agrees_with_validate_game():
    from sportsdataverse.validation import validate_game

    g = _game()
    report = validate_game(
        pl.from_dicts(g["plays"], infer_schema_length=None),
        "cfb",
        header=g.get("header"),
        summary=g,
        box=g.get("advBoxScore"),
    )
    row = build_dataset_frame(REGISTRY["qa"], g).row(0, named=True)
    assert row["ok"] is report.ok
    assert row["n_errors"] == len(report.errors)
    assert row["failed_rule_ids"] == ",".join(sorted(f.rule_id for f in report.errors))


def test_season_summary_and_sidecar(tmp_path):
    df = build_dataset_frame(REGISTRY["qa"], _game())
    path = qa_sidecar(df, 2024, base=tmp_path)
    s = json.loads(path.read_text())
    assert s["games"] == 1 and s["season"] == 2024 and s["league"] == "cfb"
    assert s["max_error_share"] == qa.MAX_ERROR_SHARE
    assert s["blocking"] is False, "V2 ships report-only"
    assert s["threshold_exceeded"] is (s["error_share"] > qa.MAX_ERROR_SHARE)
    assert s["drift"] == []  # no season pbp parquet under tmp_path
    for rule in filter(None, df.row(0, named=True)["failed_rule_ids"].split(",")):
        assert s["counts_by_rule"][rule] == 1


def test_drift_gate_reports_schema_null_constant_and_mean_shift():
    new = build_dataset_frame(REGISTRY["pbp"], _game())
    assert new.height > 100 and "EPA" in new.columns
    prev = (
        new.drop("period")
        .with_columns(
            pl.col("game_id").cast(pl.Utf8),
            (pl.col("EPA") * 0.5).alias("EPA"),
        )
        .with_columns(
            pl.Series("varies", [str(i % 7) for i in range(new.height)], dtype=pl.Utf8)
        )
    )
    new = new.with_columns(pl.lit("one", dtype=pl.Utf8).alias("varies"))
    found = qa.drift_findings(new, prev)
    by = {(f["check"], f["locator"].get("column")) for f in found}
    assert ("schema_contract", "period") in by
    assert ("schema_contract", "game_id") in by
    assert ("constant_column", "varies") in by
    assert ("rate_anomaly", "EPA") in by
    assert qa.drift_findings(new, None) == []


def test_published_frame_is_best_effort(tmp_path):
    assert qa.published_frame(str(tmp_path / "no-such-release.parquet")) is None
