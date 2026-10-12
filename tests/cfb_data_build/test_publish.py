"""publish_dataset uploads only what changed (the droplet's shared 5,000/h API quota)."""

import hashlib
from pathlib import Path

from cfb_data_build import publish
from cfb_data_build.config import REGISTRY

SPEC = REGISTRY["team_box"]
REPO = "sportsdataverse/sportsdataverse-data"


def _write(base: Path, season: int = 2026) -> dict[str, Path]:
    out = {}
    for sub, ext, body in [
        ("parquet", "parquet", b"PQ"),
        ("rds", "rds", b"RDS-with-a-write-time"),
        ("csv", "csv", b"a,b\n1,2\n"),
    ]:
        d = base / SPEC.dataset / sub
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{SPEC.stem}_{season}.{ext}"
        f.write_bytes(body)
        out[ext] = f
    return out


def _digest(f: Path) -> str:
    return "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()


def _uploads(calls):
    return [Path(c[3]).name for c in calls if c[:2] == ["release", "upload"]]


def test_unchanged_parquet_skips_the_stem_and_the_stamp(tmp_path):
    files = _write(tmp_path)
    remote = {f.name: _digest(f) for f in files.values()}
    remote[files["rds"].name] = "sha256:stale-write-time"  # rds re-stamped
    calls: list[list[str]] = []

    res = publish.publish_dataset(
        SPEC, 2026, base=tmp_path, runner=calls.append, digests=lambda t, r: remote
    )

    assert calls == []  # no upload, no sidecar re-stamp
    assert res["uploaded"] == {REPO: 0}
    assert sorted(res["skipped"][REPO]) == sorted(f.name for f in files.values())


def test_changed_parquet_uploads_every_format_parquet_last_then_stamps(tmp_path):
    files = _write(tmp_path)
    remote = {f.name: _digest(f) for f in files.values()}
    remote[files["parquet"].name] = "sha256:old"
    remote[files["rds"].name] = "sha256:old"
    calls: list[list[str]] = []

    publish.publish_dataset(
        SPEC, 2026, base=tmp_path, runner=calls.append, digests=lambda t, r: remote
    )

    up = _uploads(calls)
    # csv matched its digest so it stays; rds + parquet go, parquet LAST before the sidecars
    assert up[:2] == [files["rds"].name, files["parquet"].name]
    assert "timestamp.json" in up[2:]


def test_missing_sibling_forces_the_stem_even_when_parquet_matches(tmp_path):
    files = _write(tmp_path)
    remote = {files["parquet"].name: _digest(files["parquet"])}
    calls: list[list[str]] = []

    publish.publish_dataset(
        SPEC, 2026, base=tmp_path, runner=calls.append, digests=lambda t, r: remote
    )

    up = _uploads(calls)
    assert set(up[:3]) == {f.name for f in files.values()}
    assert up[2] == files["parquet"].name


def test_no_release_creates_it_then_uploads_everything(tmp_path):
    _write(tmp_path)
    calls: list[list[str]] = []

    publish.publish_dataset(
        SPEC,
        2026,
        base=tmp_path,
        runner=calls.append,
        exists_check=lambda t, r: False,
        digests=lambda t, r: None,
    )

    assert calls[0][:3] == ["release", "create", SPEC.tag]
    assert len(_uploads(calls)) >= 3


def test_failed_sibling_holds_the_parquet_back(tmp_path):
    import subprocess

    import pytest

    files = _write(tmp_path)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        if argv[3].endswith(".rds"):
            raise subprocess.CalledProcessError(1, argv)

    with pytest.raises(subprocess.CalledProcessError):
        publish.publish_dataset(SPEC, 2026, base=tmp_path, runner=runner, digests=lambda t, r: {})
    assert files["parquet"].name not in _uploads(calls)


def test_release_digests_404_is_none_other_errors_raise(monkeypatch):
    import subprocess

    import pytest

    class R:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    monkeypatch.setattr(publish, "GH_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R(1, err="HTTP 404: Not Found"))
    assert publish._release_digests("t", REPO) is None

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R(0, out="a.parquet\tsha256:ab\n"))
    assert publish._release_digests("t", REPO) == {"a.parquet": "sha256:ab"}

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R(1, err="HTTP 403: rate limit"))
    with pytest.raises(RuntimeError):
        publish._release_digests("t", REPO)
