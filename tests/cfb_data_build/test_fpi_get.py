"""``fpi._get`` retries transient core-v2 failures with backoff; a client error is final.

Shared by the FPI and polls captures. Polls propagates a failure (one bad fetch
fails the season), so without this ~120 sequential requests made one stray 403
fail a backfill season."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from cfb_data_build import fpi

URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/x"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "msg", {}, None)  # type: ignore[arg-type]


@pytest.fixture
def opener(monkeypatch):
    """Serve scripted outcomes (an exception, raw bytes, or a JSON body); record the waits."""
    state = {"outcomes": [], "calls": 0, "waits": []}

    def urlopen(req, timeout):
        state["calls"] += 1
        outcome = state["outcomes"].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Resp(
            outcome if isinstance(outcome, bytes) else json.dumps(outcome).encode()
        )

    monkeypatch.setattr(fpi.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fpi.time, "sleep", state["waits"].append)
    return state


def test_fails_twice_then_succeeds(opener):
    opener["outcomes"] = [
        _http(503),
        urllib.error.URLError("connection reset"),
        {"ok": 1},
    ]
    assert fpi._get(URL) == {"ok": 1}
    assert opener["calls"] == 3
    assert opener["waits"] == [2, 6]


@pytest.mark.parametrize(
    "exc",
    [
        _http(403),
        _http(429),
        _http(500),
        _http(502),
        urllib.error.URLError("dns"),
        TimeoutError(),
    ],
    ids=["403", "429", "500", "502", "URLError", "TimeoutError"],
)
def test_transient_failures_retry(opener, exc):
    opener["outcomes"] = [exc, {"ok": 1}]
    assert fpi._get(URL) == {"ok": 1}
    assert opener["calls"] == 2 and opener["waits"] == [2]


@pytest.mark.parametrize(
    "exc", [_http(404), _http(400), _http(401)], ids=["404", "400", "401"]
)
def test_a_client_error_is_final(opener, exc):
    opener["outcomes"] = [exc, {"never": 1}]
    with pytest.raises(urllib.error.HTTPError):
        fpi._get(URL)
    assert opener["calls"] == 1 and opener["waits"] == []


def test_gives_up_after_the_last_wait(opener):
    opener["outcomes"] = [_http(503)] * (len(fpi._RETRY_WAITS) + 1)
    with pytest.raises(urllib.error.HTTPError):
        fpi._get(URL)
    assert opener["calls"] == len(fpi._RETRY_WAITS) + 1
    assert opener["waits"] == list(fpi._RETRY_WAITS) == [2, 6, 18]


def test_a_bad_body_is_not_retried(opener):
    """Garbage on a 200 is a parse error, not an outage."""
    opener["outcomes"] = [b"<html>", {"never": 1}]
    with pytest.raises(json.JSONDecodeError):
        fpi._get(URL)
    assert opener["calls"] == 1 and opener["waits"] == []
