"""team_summaries parity (integration) -- the season-aggregation port vs R oracle.

Marked ``integration`` (deselected by default) because it consumes the cached
season ``plays_input`` parquet captured from R's ``cfbfastR::load_cfb_pbp`` (too
large to commit; regenerate with the capture script). The 5 R output frames ARE
committed (small). Run with: ``pytest -m integration tests/cfb_data_build``.

Parity bar: deterministic aggregation columns exact (value, order-agnostic since
the team_summaries column order is a join artifact); the ridge-adjusted EPA
columns (glmnet in R vs sklearn here) held to a Pearson-correlation bar.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.team_summaries import build_team_summaries
from tests.cfb_data_build._parity_helpers import assert_frame_parity

pytestmark = pytest.mark.integration

FIX = Path(__file__).parent / "fixtures"
CACHE = Path(__file__).parents[2] / ".cache" / "team_summaries"

# ridge-adjusted EPA columns -- correlation bar, not exact (glmnet vs sklearn)
CORR_COLS = {
    "adj_off_epa",
    "adj_def_epa",
    "net_adj_epa",
    "off_strength_faced",
    "def_strength_faced",
    "adj_off_epa_rank",
    "adj_def_epa_rank",
    "net_adj_epa_rank",
}

# dataset -> (sort keys, correlation columns)
CASES = [
    ("percentiles", ["pctile"], set()),
    ("passing", ["team_id", "player_id"], set()),
    ("rushing", ["team_id", "player_id"], set()),
    ("receiving", ["team_id", "player_id"], set()),
    ("team_summaries", ["team_id"], CORR_COLS),
]


def _season() -> int:
    f = FIX / "team_summaries_oracle_season.txt"
    if not f.exists():
        pytest.skip("team_summaries oracle not captured")
    return int(f.read_text().strip())


@pytest.fixture(scope="module")
def built() -> tuple[int, dict[str, pl.DataFrame]]:
    yr = _season()
    plays_path = CACHE / f"plays_input_{yr}.parquet"
    if not plays_path.exists():
        pytest.skip(f"cached plays_input missing: {plays_path}")
    return yr, build_team_summaries(pl.read_parquet(plays_path), yr)


@pytest.mark.parametrize("ds,keys,corr", CASES, ids=[c[0] for c in CASES])
def test_team_summaries_parity(
    built: tuple[int, dict[str, pl.DataFrame]],
    ds: str,
    keys: list[str],
    corr: set[str],
) -> None:
    yr, out = built
    oracle = pl.read_parquet(FIX / f"oracle_ts_{ds}_{yr}.parquet")
    assert_frame_parity(
        out[ds],
        oracle,
        name=ds,
        match_order=False,
        sort_keys=keys,
        corr_cols=corr,
        corr_threshold=0.9,
    )
