import json
from cfb_model_publish.artifacts import plan_uploads, upload_artifacts


def _seed(tmp_path):
    (tmp_path / "ep.ubj").write_bytes(b"x")
    (tmp_path / "ep.json").write_text(json.dumps({"model_type": "ep"}))
    (tmp_path / "xrepa_final.pkl").write_bytes(b"y")
    (tmp_path / "xrepa_final.json").write_text(json.dumps({"target": "x"}))
    return tmp_path


def test_plan_uploads_lists_models_and_cards(tmp_path):
    files = {p.name for p in plan_uploads(_seed(tmp_path))}
    assert files == {"ep.ubj", "ep.json", "xrepa_final.pkl", "xrepa_final.json"}


def _boom(*_a, **_k):
    raise AssertionError("should not be called")


def test_dry_run_uploads_nothing(tmp_path):
    calls = []
    # dry-run must be network-free: it must NOT probe for release existence.
    res = upload_artifacts(_seed(tmp_path), "espn_cfb_model_artifacts", "owner/repo",
                           dry_run=True, runner=lambda args: calls.append(args),
                           exists_check=_boom)
    assert res["uploaded"] == 0 and len(res["files"]) == 4 and calls == []
    assert res["created_release"] is False


def test_upload_invokes_runner_per_file(tmp_path):
    calls = []
    res = upload_artifacts(_seed(tmp_path), "espn_cfb_model_artifacts", "owner/repo",
                           dry_run=False, runner=lambda args: calls.append(args),
                           exists_check=lambda tag, repo: True)
    assert res["uploaded"] == 4 and len(calls) == 4
    assert res["created_release"] is False
    assert not any(c[:2] == ["release", "create"] for c in calls)


def test_creates_release_when_missing(tmp_path):
    calls = []
    res = upload_artifacts(_seed(tmp_path), "espn_cfb_model_artifacts", "owner/repo",
                           dry_run=False, runner=lambda args: calls.append(args),
                           exists_check=lambda tag, repo: False)
    # first call creates the release, then one upload per file
    assert calls[0][:4] == ["release", "create", "espn_cfb_model_artifacts", "--repo"]
    assert res["created_release"] is True
    assert res["uploaded"] == 4 and len(calls) == 5


def test_skips_create_when_present(tmp_path):
    calls = []
    res = upload_artifacts(_seed(tmp_path), "espn_cfb_model_artifacts", "owner/repo",
                           dry_run=False, runner=lambda args: calls.append(args),
                           exists_check=lambda tag, repo: True)
    assert res["created_release"] is False
    assert all(c[0:2] != ["release", "create"] for c in calls)
    assert res["uploaded"] == 4 and len(calls) == 4
