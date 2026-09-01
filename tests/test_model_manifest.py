"""models/manifest.yaml — the per-ROW registry lockstep (Track C step 2).

`tests/test_model_registry.py` remains the package-level floor; its own header
documents that deleting one row (e.g. `ep`) still passes because sibling rows
cite the same package. THIS guard closes that gap: every registry row's model
cell must have a manifest entry naming it verbatim, and vice versa. Deleting a
row from either file goes red here.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "models" / "manifest.yaml"
REGISTRY = REPO / "models" / "REGISTRY.md"

PACKAGE_RE = re.compile(r'^PACKAGE = "(?P<pkg>[a-z_.]+)"$', re.M)


def _doc() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _registry_model_cells() -> set[str]:
    rows = [
        ln
        for ln in REGISTRY.read_text(encoding="utf-8").splitlines()
        if ln.startswith("|") and not set(ln) <= set("|- ")
    ][1:]  # drop header
    return {row.split("|")[1].strip() for row in rows}


def test_manifest_parses_and_driver_exists():
    doc = _doc()
    assert (REPO / doc["driver"]).is_file()
    assert doc["models"] and doc["stages"]


def test_registry_rows_and_manifest_models_lockstep():
    registry = _registry_model_cells()
    manifest = {m["registry_model"] for m in _doc()["models"].values()}
    assert registry == manifest, (
        f"registry-only rows (add a manifest entry): {sorted(registry - manifest)}; "
        f"manifest-only entries (row deleted?): {sorted(manifest - registry)}"
    )


def test_stages_exist_and_declare_their_package():
    for num, s in _doc()["stages"].items():
        f = REPO / s["file"]
        assert f.is_file(), f"stage {num} file missing: {s['file']}"
        m = PACKAGE_RE.search(f.read_text(encoding="utf-8"))
        assert m and m.group("pkg") == s["package"], (
            f"stage {num}: manifest package {s['package']!r} != file's PACKAGE"
        )


def test_numbered_stage_files_and_manifest_agree_bidirectionally():
    on_disk = {p.name for p in (REPO / "python").glob("cfb_model_*_creation.py")}
    in_manifest = {Path(s["file"]).name for s in _doc()["stages"].values()}
    assert on_disk == in_manifest, (
        f"disk-only={on_disk - in_manifest}, manifest-only={in_manifest - on_disk}"
    )


def test_every_model_points_at_a_declared_stage():
    stages = set(_doc()["stages"])
    for name, m in _doc()["models"].items():
        assert m["stage"] in stages, f"{name} cites undeclared stage {m['stage']!r}"


def test_fitted_stages_route_through_the_fingerprint_runtime():
    for num in ("30", "31", "32", "33", "34"):
        src = (REPO / _doc()["stages"][num]["file"]).read_text(encoding="utf-8")
        assert "_model_stage" in src, (
            f"stage {num} bypasses the fingerprint/ledger runtime"
        )
