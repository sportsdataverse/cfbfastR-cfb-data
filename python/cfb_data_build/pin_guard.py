"""Refuse to publish from a sportsdataverse-py that is not the one main's uv.lock pins.

Incident, 2026-11-26: the droplet's daily build pulled at 09:19Z, five minutes before the
sdv-py pin bump (#163) merged. It built and published the 2026 team summaries family on the
old engine at 12:35Z, overwriting a correct republish. Its venv matched the lock it had
pulled, so a venv-vs-local-lock check alone would not have caught it. The check that does
compares the RUNNING sdv-py with the pin on origin's main at publish time.

    python -m cfb_data_build.pin_guard     # exit 1 with the reason when stale

``CFB_SKIP_PIN_CHECK=1`` skips it, for a deliberate publish from a branch whose lock is
ahead of main, such as a pin-bump PR's own republish. Say so in that run's log.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = "sportsdataverse/cfbfastR-cfb-data"
LOCK = Path(__file__).resolve().parents[2] / "uv.lock"
_PIN = re.compile(r"sportsdataverse-py\?branch=main#([0-9a-f]{40})")
_TTL_S = 600  # re-read main's lock at most every 10 min, so a mid-build merge is caught
_cache: dict[str, tuple[float, str]] = {}


class StalePinError(RuntimeError):
    """The running sportsdataverse-py is not the one the lock pins."""


def locked_commit(lock_text: str) -> str:
    m = _PIN.search(lock_text)
    if not m:
        raise StalePinError("no sportsdataverse-py git pin found in uv.lock")
    return m.group(1)


def installed_commit() -> str | None:
    """Commit of the importable sportsdataverse; None when it is not a git install."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        raw = distribution("sportsdataverse").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    return (
        json.loads(raw).get("vcs_info", {}).get("commit_id") if raw else None
    ) or None


def remote_lock_text() -> str:
    """main's uv.lock, read through gh (the transport the uploads already use)."""
    now = time.monotonic()
    hit = _cache.get("main")
    if hit and now - hit[0] < _TTL_S:
        return hit[1]
    text = subprocess.run(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github.raw",
            f"repos/{REPO}/contents/uv.lock?ref=main",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    _cache["main"] = (now, text)
    return text


def check(
    *,
    installed: str | None = None,
    local_lock: str | None = None,
    main_lock: str | None = None,
) -> str:
    """Return the pinned commit, or raise StalePinError naming what differs.

    Every input is injectable for tests; by default the running package, this checkout's
    uv.lock and main's uv.lock are read.
    """
    have = installed if installed is not None else installed_commit()
    if not have:
        raise StalePinError(
            "sportsdataverse is not a git install; cannot tell which commit is running"
        )
    local = locked_commit(
        local_lock if local_lock is not None else LOCK.read_text(encoding="utf-8")
    )
    if have != local:
        raise StalePinError(
            f"the venv runs sportsdataverse-py {have[:10]} but this checkout's uv.lock pins {local[:10]}: "
            "run `uv sync --frozen --inexact`, then rebuild"
        )
    main = locked_commit(main_lock if main_lock is not None else remote_lock_text())
    if have != main:
        raise StalePinError(
            f"this build runs sportsdataverse-py {have[:10]} but main's uv.lock now pins {main[:10]} "
            "(the pin moved after this build pulled): pull, sync, and rerun the season"
        )
    return have


def require_current_pin() -> None:
    """Raise unless publishing is allowed; honours CFB_SKIP_PIN_CHECK=1."""
    if os.environ.get("CFB_SKIP_PIN_CHECK") == "1":
        print(
            "pin_guard: CFB_SKIP_PIN_CHECK=1, sportsdataverse-py pin NOT checked",
            flush=True,
        )
        return
    check()


if __name__ == "__main__":
    try:
        sha = check()
    except (StalePinError, subprocess.SubprocessError, OSError) as e:
        print(f"::error ::pin_guard: {e}", flush=True)
        sys.exit(1)
    print(
        f"pin_guard: sportsdataverse-py {sha[:10]} matches this checkout's and main's uv.lock",
        flush=True,
    )
