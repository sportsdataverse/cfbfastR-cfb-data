"""Capture the CFBD side-input payloads the matchup line's Phase 6 joins read.

Writes small parquet fixtures under tests/cfb_data_build/fixtures/matchup/ for
one season: /talent, /games/weather, /teams and the head-coach history
(/coaches, tidied to one row per coach-school-season). Needs CFBD_API_KEY.

    PYTHONPATH=python uv run python ops/oneoff/20260915_matchup_oracle_capture/capture_side_inputs.py 2025
"""

from __future__ import annotations

import sys
from pathlib import Path

from cfb_data_build.matchup_side import (
    fetch_cfbd_coaches,
    fetch_cfbd_talent,
    fetch_cfbd_teams,
    fetch_cfbd_weather,
    tidy_cfbd_coaches,
    tidy_cfbd_talent,
    tidy_cfbd_teams,
    tidy_cfbd_weather,
)

FIX = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "cfb_data_build"
    / "fixtures"
    / "matchup"
)


def main(season: int) -> None:
    out = {
        f"cfbd_talent_{season}": tidy_cfbd_talent(fetch_cfbd_talent(season)),
        f"cfbd_weather_{season}": tidy_cfbd_weather(fetch_cfbd_weather(season)),
        f"cfbd_teams_{season}": tidy_cfbd_teams(fetch_cfbd_teams(season)),
        f"cfbd_coaches_thru_{season}": tidy_cfbd_coaches(fetch_cfbd_coaches(season)),
    }
    for stem, frame in out.items():
        path = FIX / f"{stem}.parquet"
        frame.write_parquet(path)
        print(f"{path.name}: {frame.shape}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2025)
