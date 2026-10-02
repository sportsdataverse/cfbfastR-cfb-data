"""The nightly driver's recruiting block: which datasets run for which season.

Runs the block itself out of ``scripts/daily_cfb_processor.sh`` under bash with
``run_py`` stubbed to echo, so the assertion is on what the driver would build,
not on a re-implementation of its rules.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from cfb_data_build.recruiting import TEAM_PORTAL_FIRST_SEASON

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "daily_cfb_processor.sh"


def _run_block(season: int, raw_root: Path) -> list[str]:
    sh = SCRIPT.read_text()
    lists = "\n".join(re.findall(r'^PY_RECRUITING(?:_NO_STORE)?="[^"]*"$', sh, re.M))
    block = re.search(
        r"^    # team_portal starts at .*?^      for ds in \$REC_NO_STORE.*?^    fi$",
        sh,
        re.M | re.S,
    ).group(0)
    prog = f"""set -uo pipefail
{lists}
i={season}; CFB_RAW_ROOT={raw_root}
run_py() {{ echo "RUN $1"; }}
{block}
"""
    out = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, check=True)
    return [ln.split()[1] for ln in out.stdout.splitlines() if ln.startswith("RUN ")]


@pytest.fixture()
def with_store(tmp_path):
    (tmp_path / "cfb" / "recruits" / "json").mkdir(parents=True)
    return tmp_path


def test_team_portal_runs_nightly_after_returning_production(with_store):
    ran = _run_block(2026, with_store)
    assert ran == ["recruits", "team_talent", "returning_production", "team_portal"]


def test_team_portal_is_left_out_below_its_floor(with_store):
    """The builder raises below 2015; running it would turn a backfill season red."""
    assert "team_portal" not in _run_block(TEAM_PORTAL_FIRST_SEASON - 1, with_store)
    assert "team_portal" in _run_block(TEAM_PORTAL_FIRST_SEASON, with_store)


def test_store_free_datasets_still_run_without_the_247_store(tmp_path):
    assert _run_block(2026, tmp_path) == ["returning_production", "team_portal"]
    assert _run_block(2014, tmp_path) == ["returning_production"]
