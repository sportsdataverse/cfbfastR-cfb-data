"""build_cfb_crosswalk --upload sends only changed assets (one shared 5,000/h API quota)."""

import hashlib
from pathlib import Path

import build_cfb_crosswalk as bx

TAG, REPO = "cfb_crosswalk", "sportsdataverse/sportsdataverse-data"


def _files(tmp_path: Path) -> list[Path]:
    out = []
    for name, body in [
        ("cfb_teams_crosswalk_2026.parquet", b"T"),
        ("cfb_schedule_crosswalk_2026.parquet", b"S"),
        ("cfb_crosswalk_in_data_repo.csv", b"dataset,generated_at_utc\nteams,now\n"),
    ]:
        p = tmp_path / name
        p.write_bytes(body)
        out.append(p)
    return out


def _sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def _uploads(calls):
    return [Path(c[3]).name for c in calls if c[:2] == ["release", "upload"]]


def test_unchanged_parquets_skip_everything_including_the_restamped_manifest(tmp_path):
    files = _files(tmp_path)
    remote = {p.name: _sha(p) for p in files}
    remote["cfb_crosswalk_in_data_repo.csv"] = "sha256:yesterday"
    calls: list[list[str]] = []
    bx.upload_assets(files, TAG, REPO, run=calls.append, digests=lambda t, r: remote)
    assert calls == []


def test_changed_parquet_uploads_it_then_the_manifest_then_sidecars(tmp_path):
    files = _files(tmp_path)
    remote = {p.name: _sha(p) for p in files}
    remote["cfb_schedule_crosswalk_2026.parquet"] = "sha256:old"
    calls: list[list[str]] = []
    bx.upload_assets(files, TAG, REPO, run=calls.append, digests=lambda t, r: remote)
    up = _uploads(calls)
    assert up[:2] == ["cfb_schedule_crosswalk_2026.parquet", "cfb_crosswalk_in_data_repo.csv"]
    assert "timestamp.json" in up[2:]


def test_manifest_missing_from_release_uploads_alone(tmp_path):
    files = _files(tmp_path)
    remote = {p.name: _sha(p) for p in files if p.suffix == ".parquet"}
    calls: list[list[str]] = []
    bx.upload_assets(files, TAG, REPO, run=calls.append, digests=lambda t, r: remote)
    assert _uploads(calls)[0] == "cfb_crosswalk_in_data_repo.csv"


def test_no_release_creates_it_and_uploads_all(tmp_path, monkeypatch):
    files = _files(tmp_path)
    created = []
    monkeypatch.setattr(bx, "ensure_release", lambda t, r, **k: created.append(t))
    calls: list[list[str]] = []
    bx.upload_assets(files, TAG, REPO, run=calls.append, digests=lambda t, r: None)
    assert created == [TAG]
    assert _uploads(calls)[:3] == [p.name for p in files]


# --- bounded retry: a 502 on the delete half of --clobber dropped an asset (2026-10-12) ---


class _Proc:
    def __init__(self, rc, err=""):
        self.returncode, self.stdout, self.stderr = rc, "", err


def _seq(monkeypatch, results):
    from cfb_data_build import publish

    seen = []
    it = iter(results)

    def _run(argv, **kw):
        seen.append(argv)
        return _Proc(*next(it))

    monkeypatch.setattr(publish, "GH_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(publish.subprocess, "run", _run)
    return publish, seen


def test_gh_retry_transient_then_success(monkeypatch):
    publish, seen = _seq(monkeypatch, [(1, "HTTP 502: Server Error"), (0,)])
    publish.gh_retry(["release", "upload", TAG, "f", "--clobber"])
    assert len(seen) == 2


def test_gh_retry_never_repeats_a_4xx(monkeypatch):
    import subprocess

    import pytest

    publish, seen = _seq(monkeypatch, [(1, "HTTP 403: Forbidden")] * 3)
    with pytest.raises(subprocess.CalledProcessError):
        publish.gh_retry(["release", "upload", TAG, "f", "--clobber"])
    assert len(seen) == 1


def test_failed_upload_names_the_asset_clobber_deleted(tmp_path, caplog):
    import subprocess

    import pytest

    files = _files(tmp_path)
    listings = iter([{}, {}])  # before, and the re-list after the failure: nothing there

    def run(argv):
        raise subprocess.CalledProcessError(1, argv, "", "HTTP 502")

    with pytest.raises(subprocess.CalledProcessError):
        bx.upload_assets(files, TAG, REPO, run=run, digests=lambda t, r: next(listings))
    assert f"ASSET MISSING: {TAG}:{files[0].name}" in caplog.text


def test_create_that_landed_despite_failing_is_not_fatal(tmp_path):
    import subprocess

    def run(argv):
        raise subprocess.CalledProcessError(1, argv, "", "HTTP 422: already_exists")

    bx.ensure_release(TAG, REPO, run=run, digests=lambda t, r: {})  # re-list finds it
