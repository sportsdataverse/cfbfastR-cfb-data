import json
from pathlib import Path
from cfb_model_reports.discovery import discover_models


def test_discover_reads_model_type_and_locates_sibling(tmp_path):
    (tmp_path / "ep.ubj").write_bytes(b"x")
    (tmp_path / "ep.json").write_text(json.dumps({"model_type": "ep", "features": ["down"]}))
    (tmp_path / "xrepa_final.pkl").write_bytes(b"y")
    (tmp_path / "xrepa_final.json").write_text(json.dumps({"model_formula": "s(0)+s(1)", "target": "unadjusted_epa"}))
    (tmp_path / "orphan.json").write_text(json.dumps({"model_type": "ghost"}))  # no sibling model -> skipped
    found = {m.model_type: m for m in discover_models(tmp_path)}
    assert set(found) == {"ep", "rb_eval"}
    assert found["ep"].model_path.name == "ep.ubj" and found["ep"].card["features"] == ["down"]
    assert found["rb_eval"].model_path.suffix == ".pkl"
