"""``_gh_runner`` retries a transient ``gh`` failure and still fails loudly.

cfbfastR-cfb-data run 34161449645 died on ``HTTP 502`` uploading the
``espn_cfb_team_box`` timestamp sidecar, twenty minutes after the season's
parquet had published. One un-retried blip reddened a five-season publish.
"""

from __future__ import annotations

import subprocess

import pytest

from cfb_model_build.cfb_model_publish import artifacts


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Keep the retry timing out of the test's wall clock."""
    monkeypatch.setattr(artifacts.time, "sleep", lambda _s: None)


def _boom(returncode=1):
    return subprocess.CalledProcessError(returncode, ["gh", "release", "upload"])


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise _boom()
        return None

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)
    artifacts._gh_runner(["release", "upload", "tag", "f.json", "--clobber"])
    assert len(calls) == 3, "should have retried twice before succeeding"
    assert calls[-1][0] == "gh"


def test_persistent_failure_raises_and_never_returns_none(monkeypatch):
    """The whole point: a lost asset must NOT read as a green run.

    ``derived.py::_retry`` returns None when the attempts run out, which is
    right for one week of a season sweep and wrong here -- a swallowed publish
    failure is the green-run-that-published-nothing mode.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        raise _boom(returncode=2)

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        artifacts._gh_runner(["release", "upload", "tag", "f.json", "--clobber"])
    assert len(calls) == artifacts.GH_RETRY_ATTEMPTS


def test_success_runs_exactly_once(monkeypatch):
    calls = []
    monkeypatch.setattr(artifacts.subprocess, "run", lambda args, **kw: calls.append(args))
    artifacts._gh_runner(["release", "view", "tag"])
    assert len(calls) == 1


def test_timeout_is_not_retried(monkeypatch):
    """A 30-minute timeout retried three times stalls a publish for 90.

    GH_TIMEOUT_SECONDS is already sized for the largest artifact on a slow
    link, so a timeout is a real failure, not a blip to paper over.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(["gh"], artifacts.GH_TIMEOUT_SECONDS)

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        artifacts._gh_runner(["release", "upload", "tag", "big.parquet", "--clobber"])
    assert len(calls) == 1
