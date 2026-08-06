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
CACHE = Path(__file__).parents[2] / "python" / ".cache" / "team_summaries"

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

# Columns we DELIBERATELY diverge from the R oracle on, excluded from parity.
#
# R computes these as `mean(epa_success * pass)` over every play. That product is
# 1 only on a successful pass and 0 on every other play (including every rush),
# so the mean is (# successful passes) / (# ALL plays) -- the share of all plays
# that were successful passes, not the pass success rate. The published data
# shows it plainly: the pass and rush values SUM to the overall metric
# (2024 medians 0.2153 + 0.2190 = 0.4343 vs success 0.4426) rather than each
# sitting near it.
#
# Not put on a correlation bar instead: across the 99 quantile rows both the old
# and new columns increase monotonically, so they would correlate ~1.0 and the
# check would pass while measuring nothing. Excluded outright, with
# test_percentile_pass_rush_are_rates_not_shares below asserting the new
# semantics directly.
DIVERGENT_FROM_R = {
    "percentiles": {"pass_success", "rush_success", "pass_explosive", "rush_explosive"},
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
    got = out[ds]
    drop = DIVERGENT_FROM_R.get(ds, set())
    if drop:
        got = got.drop([c for c in drop if c in got.columns])
        oracle = oracle.drop([c for c in drop if c in oracle.columns])
    assert_frame_parity(
        got,
        oracle,
        name=ds,
        match_order=False,
        sort_keys=keys,
        corr_cols=corr,
        corr_threshold=0.9,
    )


def test_percentile_pass_rush_are_rates_not_shares(
    built: tuple[int, dict[str, pl.DataFrame]],
) -> None:
    """The four columns excluded from R parity must be RATES, not shares.

    The failure signature this guards is specific and was visible in every
    published season: because R divides successful passes by ALL plays, the pass
    and rush values SUM to the overall metric instead of each sitting near it.
    A real pass success rate is close to the overall success rate (~0.44), not
    half of it.
    """
    _yr, out = built
    p = out["percentiles"]
    mid = p.filter((pl.col("pctile") - 0.5).abs() < 1e-9)
    assert mid.height == 1, "expected a median row"
    row = mid.row(0, named=True)
    for whole, part in (
        ("success", "pass_success"),
        ("success", "rush_success"),
        ("explosive", "pass_explosive"),
        ("explosive", "rush_explosive"),
    ):
        assert row[part] > 0.6 * row[whole], (
            f"{part}={row[part]:.4f} is far below {whole}={row[whole]:.4f} at the median -- "
            "this is the share-of-all-plays bug, not a rate"
        )
    # and the decisive one: the two halves must NOT sum to the whole
    assert row["pass_success"] + row["rush_success"] > 1.4 * row["success"], (
        "pass_success + rush_success still sums to overall success, so the "
        "denominator is still all plays rather than pass / rush plays"
    )


def test_sack_and_int_only_passers_are_kept(
    built: tuple[int, dict[str, pl.DataFrame]],
) -> None:
    """A passer whose whole season is sacks/picks must still appear (#33).

    `qb_data` used to be seeded only from completion/incompletion-derived ids
    with sacks and interceptions LEFT-joined on, so a passer with neither an
    attempt nor a completion had no seed row and vanished -- 13 sack-only and
    20 int-only keys in 2025. Same family as #30: the plays that went missing
    were the negative-only ones.
    """
    _yr, out = built
    p = out["passing"]
    seeded = p.filter(pl.col("att") == 0)

    # every seeded row must be there for a reason and be self-consistent
    assert (seeded["sacked"] + seeded["pass_int"] > 0).all(), (
        "a zero-attempt passer only belongs here if he took a sack or threw a pick"
    )
    assert (seeded["dropbacks"] > 0).all(), "seeded rows must have a real denominator"
    assert seeded["games"].is_not_null().all(), "seeded rows need games for EPAgame"
    # comppct is undefined without attempts -- null, never 0/0
    assert seeded.filter(pl.col("pass_int") == 0)["comppct"].is_null().all()


def test_no_division_artifacts_in_passing(
    built: tuple[int, dict[str, pl.DataFrame]],
) -> None:
    """Seeding zero-attempt passers (#33) added new zero denominators.

    `comppct` divides by attempts and `yardsplay` by plays, both of which are
    now legitimately 0 for a seeded row. A NaN reaching the release would also
    poison the derived `*_rank` columns.
    """
    _yr, out = built
    p = out["passing"]
    for col, dtype in p.schema.items():
        if dtype not in (pl.Float64, pl.Float32):
            continue
        s = p[col]
        bad = int((s.is_nan() | s.is_infinite()).sum())
        assert bad == 0, f"{col} has {bad} inf/NaN values"
