"""The publish-time sdv-py pin guard, on this repo's real uv.lock.

The failure it guards (2026-11-26): a daily build pulled 5 min before the sdv-py pin bump
merged and published 2026 on the old engine. Its venv matched its own lock, so only the
comparison with MAIN's lock catches it.
"""

from __future__ import annotations

import pytest

from cfb_data_build import pin_guard
from cfb_data_build.config import SUMMARIES_REGISTRY

LOCK = pin_guard.LOCK.read_text(encoding="utf-8")
PINNED = pin_guard.locked_commit(LOCK)
OTHER = "12555e0354c92a7a6e48f546a0a09486310a4200"


def test_reads_the_pin_from_the_real_lock():
    assert len(PINNED) == 40
    assert f"sportsdataverse-py?branch=main#{PINNED}" in LOCK


def test_current_pin_passes():
    assert pin_guard.check(installed=PINNED, local_lock=LOCK, main_lock=LOCK) == PINNED


def test_venv_behind_its_own_lock_is_refused():
    with pytest.raises(pin_guard.StalePinError, match="this checkout's uv.lock pins"):
        pin_guard.check(installed=OTHER, local_lock=LOCK, main_lock=LOCK)


def test_pin_moved_on_main_mid_build_is_refused():
    """The 2026-11-26 case: venv == the pulled lock, main has moved on."""
    old_lock = LOCK.replace(PINNED, OTHER)
    with pytest.raises(pin_guard.StalePinError, match="main's uv.lock now pins"):
        pin_guard.check(installed=OTHER, local_lock=old_lock, main_lock=LOCK)


def test_non_git_install_is_refused():
    with pytest.raises(pin_guard.StalePinError, match="not a git install"):
        pin_guard.check(installed="", local_lock=LOCK, main_lock=LOCK)


def test_publish_dataset_refuses_before_any_upload(tmp_path, monkeypatch):
    """Through the real transport, a stale pin stops the publish before gh is called."""
    from cfb_data_build import publish

    calls: list[list[str]] = []
    monkeypatch.setattr(publish, "gh_retry", lambda args: calls.append(args))
    # were the guard removed, the listing would go to the network: keep it hermetic
    monkeypatch.setattr(
        publish, "_release_digests", lambda t, r: calls.append(["list", t]) or {}
    )
    monkeypatch.setattr(pin_guard, "installed_commit", lambda: OTHER)
    monkeypatch.setattr(pin_guard, "remote_lock_text", lambda: LOCK)
    monkeypatch.delenv("CFB_SKIP_PIN_CHECK", raising=False)
    spec = SUMMARIES_REGISTRY["team_summaries"]
    with pytest.raises(pin_guard.StalePinError):
        publish.publish_dataset(spec, 2026, base=tmp_path, repos=["x/y"])
    assert calls == []


def test_skip_env_is_explicit(monkeypatch, capsys):
    monkeypatch.setattr(pin_guard, "installed_commit", lambda: OTHER)
    monkeypatch.setenv("CFB_SKIP_PIN_CHECK", "1")
    pin_guard.require_current_pin()
    assert "NOT checked" in capsys.readouterr().out


def test_upload_artifacts_refuses_before_any_upload(tmp_path, monkeypatch):
    from cfb_model_build.cfb_model_publish import artifacts

    calls: list[list[str]] = []
    monkeypatch.setattr(artifacts, "_gh_runner", lambda args: calls.append(args))
    monkeypatch.setattr(artifacts, "_gh_release_exists", lambda t, r: True)
    monkeypatch.setattr(pin_guard, "installed_commit", lambda: OTHER)
    monkeypatch.setattr(pin_guard, "remote_lock_text", lambda: LOCK)
    monkeypatch.delenv("CFB_SKIP_PIN_CHECK", raising=False)
    (tmp_path / "x.parquet").write_bytes(b"x")
    with pytest.raises(pin_guard.StalePinError):
        artifacts.upload_artifacts(
            tmp_path, "cfb_ratings", "x/y", files=[tmp_path / "x.parquet"]
        )
    assert calls == []


def test_crosswalk_upload_refuses_before_any_upload(tmp_path, monkeypatch):
    import build_cfb_crosswalk as xw

    calls: list[list[str]] = []
    monkeypatch.setattr(xw, "gh_retry", lambda args: calls.append(args))
    monkeypatch.setattr(pin_guard, "installed_commit", lambda: OTHER)
    monkeypatch.setattr(pin_guard, "remote_lock_text", lambda: LOCK)
    monkeypatch.delenv("CFB_SKIP_PIN_CHECK", raising=False)
    with pytest.raises(pin_guard.StalePinError):
        xw.upload_assets(
            [tmp_path / "a.parquet"], "cfb_crosswalk", "x/y", digests=lambda t, r: {}
        )
    assert calls == []


def test_lock_with_two_sportsdataverse_entries_is_refused():
    i = LOCK.index('[[package]]\nname = "sportsdataverse"\n')
    j = LOCK.index("[[package]]", i + 1)
    with pytest.raises(pin_guard.StalePinError, match="2 sportsdataverse packages"):
        pin_guard.locked_commit(LOCK + "\n" + LOCK[i:j])


def test_cli_honours_the_override(monkeypatch):
    monkeypatch.setattr(pin_guard, "installed_commit", lambda: OTHER)
    monkeypatch.setenv("CFB_SKIP_PIN_CHECK", "1")
    assert pin_guard.main() == 0
    monkeypatch.delenv("CFB_SKIP_PIN_CHECK")
    monkeypatch.setattr(pin_guard, "remote_lock_text", lambda: LOCK)
    assert pin_guard.main() == 1
