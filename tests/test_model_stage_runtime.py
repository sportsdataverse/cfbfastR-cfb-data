"""Unit tests for the fingerprint/ledger stage runtime (_model_stage)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import _model_stage as M  # noqa: E402


def test_compute_is_stable_and_argv_sensitive(tmp_path, monkeypatch):
    pkg = tmp_path / "python" / "fake_pkg"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(M, "REPO", tmp_path)
    base = M._compute("fake_pkg", ["train-ep", "--out", "a.ubj"])
    assert base == M._compute("fake_pkg", ["train-ep", "--out", "a.ubj"])
    assert base != M._compute("fake_pkg", ["train-wp", "--out", "a.ubj"])  # argv
    (pkg / "a.py").write_text("X = 2\n", encoding="utf-8")
    assert base != M._compute("fake_pkg", ["train-ep", "--out", "a.ubj"])  # code


def test_out_paths_parse_only_known_flags():
    argv = [
        "train-ep",
        "--pbp",
        "in.parquet",
        "--out",
        "a.ubj",
        "--oof-out",
        "b.parquet",
        "--verbose",
    ]
    assert M._out_paths(argv) == [Path("a.ubj"), Path("b.parquet")]
    assert M._out_paths(["loso", "--model", "ep"]) == []


def test_run_skips_only_with_match_and_existing_artifacts(tmp_path, monkeypatch):
    pkg = tmp_path / "python" / "fake_pkg"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_text("X = 1\n", encoding="utf-8")
    out = tmp_path / "m.ubj"
    calls = []
    monkeypatch.setattr(M, "REPO", tmp_path)
    monkeypatch.setattr(M, "STORE", tmp_path / ".fingerprints.json")
    monkeypatch.setattr(M, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(
        M.runpy, "run_module", lambda *a, **k: (calls.append(a), out.write_bytes(b"x"))
    )

    monkeypatch.setattr(sys, "argv", ["stage", "train-ep", "--out", str(out)])
    assert M.run("fake_pkg") == 0
    assert len(calls) == 1
    ledger = [
        json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert ledger[0]["in_published_data"] is False

    # unchanged fingerprint + artifact present -> skip (no second run)
    monkeypatch.setattr(sys, "argv", ["stage", "train-ep", "--out", str(out)])
    assert M.run("fake_pkg") == 0
    assert len(calls) == 1

    # --force retrains
    monkeypatch.setattr(
        sys, "argv", ["stage", "train-ep", "--out", str(out), "--force"]
    )
    assert M.run("fake_pkg") == 0
    assert len(calls) == 2

    # no --out flags in argv -> never skips
    monkeypatch.setattr(sys, "argv", ["stage", "loso", "--model", "ep"])
    assert M.run("fake_pkg") == 0
    monkeypatch.setattr(sys, "argv", ["stage", "loso", "--model", "ep"])
    assert M.run("fake_pkg") == 0
    assert len(calls) == 4
