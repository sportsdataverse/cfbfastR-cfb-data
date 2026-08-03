"""Follow-up dataset parity tests -- the remaining 17 datasets vs the R oracle.

Each oracle ``fixtures/oracle_<dataset>_401628455.parquet`` was produced by the
matching R ``reshape_*`` on the committed game fixture (see
``capture_all_oracle`` provenance). The expected ``(rows, cols)`` shape is
asserted first as a guard that the oracle hasn't silently drifted.

``sort=True`` only for ``rosters`` (a dedup whose row order is not positional).
``injuries`` is empty for this fixture, so it is checked separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.build import build_dataset_frame
from cfb_data_build.config import REGISTRY
from tests.cfb_data_build._parity_helpers import assert_frame_parity

FIX = Path(__file__).parent / "fixtures"
GID = 401628455

# (dataset, expected_shape, sort_before_compare)
CASES: list[tuple[str, tuple[int, int], bool]] = [
    ("team_box", (2, 21), False),
    ("player_box", (92, 56), False),
    ("drives", (24, 20), False),
    ("game_rosters", (230, 71), False),
    ("rosters", (230, 64), True),
    ("betting", (1, 9), False),
    ("schedules", (1, 20), False),
    ("linescores", (8, 5), False),
    ("power_index", (2, 4), False),
    ("adv_team", (2, 77), False),
    ("adv_passing", (6, 23), False),
    ("adv_rushing", (12, 15), False),
    ("adv_receiving", (17, 16), False),
    ("adv_defensive", (2, 21), False),
    ("adv_turnover", (2, 14), False),
    ("adv_drives", (2, 11), False),
    ("adv_situational", (2, 73), False),
]


def _load_game() -> dict:
    return json.loads((FIX / f"final_{GID}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("dataset,shape,sort", CASES, ids=[c[0] for c in CASES])
def test_followup_parity(dataset: str, shape: tuple[int, int], sort: bool) -> None:
    g = _load_game()
    py = build_dataset_frame(REGISTRY[dataset], g)
    oracle = pl.read_parquet(FIX / f"oracle_{dataset}_{GID}.parquet")
    assert oracle.shape == shape, (
        f"{dataset}: oracle drifted to {oracle.shape}, expected {shape}"
    )
    assert_frame_parity(py, oracle, name=dataset, sort=sort)


def test_injuries_empty() -> None:
    # injuries is empty for this fixture (CFB injuries usually absent); the
    # builder must degrade to a 0-row frame rather than raise.
    g = _load_game()
    assert build_dataset_frame(REGISTRY["injuries"], g).height == 0
