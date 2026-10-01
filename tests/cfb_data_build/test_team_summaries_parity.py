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

# ridge-adjusted EPA columns. Historically a 0.9 correlation bar vs R (glmnet vs
# sklearn); since the R fit was found to be a no-op (see DIVERGENT_FROM_R) every
# member is also listed as divergent, so the R comparison skips them and the
# producer's own gate (corr(adj, raw) < 0.95 in team_summaries.py) is what
# guards the adjustment. Kept so the intent is visible if an R oracle with a
# real fit is ever captured.
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
# Two later producer changes moved the Python output away from the 2023 R
# oracle ON PURPOSE (triaged 2026-09-15; the oracle cannot be re-captured into
# agreement because the R producer is retired and would reproduce the same
# numbers):
#
# * #31 returns sack and interception EPA to the passer and counts
#   interceptions as dropbacks / attempts-for-completion-percentage
#   (team_summaries.py, the ``dropbacks = att + sacked + pass_int`` block), so
#   the passer EPA family, ``dropbacks``, ``comppct``, ``yardsdropback`` and
#   every rank derived from them differ; it also KEEPS the sack- and
#   interception-only passers R dropped (12 extra rows in 2023), which
#   ``test_sack_and_int_only_passers_are_kept`` asserts directly. The row
#   alignment below removes exactly those rows (att == 0) before comparing.
# * The R opponent adjustment ran glmnet at ``cv$lambda[[1]] = 325``, a no-op
#   (spearman(raw, adj) = 0.99998; strengths spanning 0.0008 EPA/play), while
#   the Python producer fits the real ridge at lambda 0.035 and gates itself on
#   corr(adj, raw) < 0.95 -- so ``adj_*`` / ``*_strength_faced`` / their ranks
#   cannot correlate with the oracle, and ``valid_games`` is +1 for the
#   reference team's opponents (R's model.matrix drops that level; Python
#   keeps it at the intercept). The deterministic columns stay on exact parity.
DIVERGENT_FROM_R = {
    "percentiles": {"pass_success", "rush_success", "pass_explosive", "rush_explosive"},
    "passing": {
        "TEPA",
        "EPAplay",
        "EPAgame",
        "comppct",
        "dropbacks",
        "yardsdropback",
        "TEPA_rank",
        "EPAgame_rank",
        "EPAplay_rank",
        "success_rank",
        "comppct_rank",
        "yards_rank",
        "yardsplay_rank",
        "yardsgame_rank",
        "sack_adj_yards_rank",
        "yardsdropback_rank",
        "detmer_rank",
        "detmergame_rank",
    },
    "team_summaries": {
        # per drive since 2026-09-29; R averages the drive start over plays
        "start_position_off",
        "start_position_def",
        "start_position_margin",
        "start_position_off_rank",
        "start_position_def_rank",
        "start_position_margin_rank",
        "valid_games",
        "adj_off_epa",
        "adj_def_epa",
        "def_strength_faced",
        "off_strength_faced",
        "net_adj_epa",
        "adj_off_epa_rank",
        "adj_def_epa_rank",
        "net_adj_epa_rank",
    },
}

# Python-only ADDITIVE columns: the leader percentiles / ranks added in #50
# (feat(summaries): player percentiles) have no R counterpart, so the oracle can
# never carry them. They are asserted PRESENT (a silent drop still fails) and
# then removed before the column-set comparison. Any other extra column is a
# real divergence and still fails the set check.
_LEADER_PCT = {
    "EPAgame_pct",
    "EPAplay_pct",
    "TEPA_pct",
    "success_pct",
    "yards_pct",
    "yardsgame_pct",
    "yardsplay_pct",
}
#: IF-2 Five Factors columns (test_five_factors_columns.py): R never built them.
#: They also need inputs an R ``load_cfb_pbp`` capture does not carry --
#: ``drive.result``, and ``pos_team_game_giveaways`` from
#: ``summaries_input.game_giveaways`` (over sdv-py's ``is_pos_team_turnover`` /
#: ``is_def_pos_team_turnover``). A re-captured R ``plays_input`` must add them,
#: or ``build_team_summaries`` raises ColumnNotFoundError before any comparison.
_FIVE_FACTORS = {
    "explosive_margin",
    "pts_per_opp_off",
    "pts_per_opp_def",
    "pts_per_opp_margin",
    "turnovers_off",
    "turnovers_def",
    "turnover_margin",
}
#: CFBE-1d drive-efficiency columns, also Python-only (they read ``drive.result``
#: too). The ``pts_per_drive_{off,def}_n`` counts are not listed: the test drops
#: every ``_n`` column before this check.
_DRIVE_EFFICIENCY = {"pts_per_drive_off", "pts_per_drive_def", "pts_per_drive_margin"}
#: Team-factor extensions (test_factor_extension_columns.py), Python-only: starting EP
#: per drive, havoc EPA per game, expected turnover margin and turnover luck.
_FACTOR_EXTENSIONS = {
    "drive_start_ep_off",
    "drive_start_ep_def",
    "drive_start_ep_margin",
    "havoc_EPAgame_off",
    "havoc_EPAgame_def",
    "havoc_EPAgame_margin",
    "expected_turnovers_off",
    "expected_turnovers_def",
    "expected_turnover_margin",
    "turnover_luck_off",
    "turnover_luck_def",
    "turnover_luck",
    "havoc_margin",
}
PYTHON_ONLY = {
    "team_summaries": _FIVE_FACTORS
    | _DRIVE_EFFICIENCY
    | _FACTOR_EXTENSIONS
    | {f"{c}_rank" for c in _FIVE_FACTORS | _DRIVE_EFFICIENCY | _FACTOR_EXTENSIONS}
    | {"off_strength_faced_rank", "def_strength_faced_rank"},
    "passing": _LEADER_PCT
    | {
        "comppct_pct",
        "detmer_pct",
        "detmergame_pct",
        "int_epa",
        "pass_int_pct",
        "pass_int_rank",
        "passing_td_pct",
        "passing_td_rank",
        "sack_adj_yards_pct",
        "sack_epa",
        "sacked_pct",
        "sacked_rank",
        "yardsdropback_pct",
    },
    "rushing": _LEADER_PCT
    | {
        "fumbles_pct",
        "fumbles_rank",
        "plays_pct",
        "plays_rank",
        "rushing_td_pct",
        "rushing_td_rank",
    },
    "receiving": _LEADER_PCT
    | {
        "catchpct_pct",
        "comp_pct",
        "comp_rank",
        "fumbles_pct",
        "fumbles_rank",
        "passing_td_pct",
        "passing_td_rank",
        "targets_pct",
        "targets_rank",
    },
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
    # _n sample sizes and the cohort percentiles are Python-only, unit-tested in
    # test_sample_sizes.py and test_cohort_percentiles.py
    got = got.drop(
        [
            c
            for c in got.columns
            if c.endswith(("_n", "_pos_pct", "_conf_pct")) or c == "position_group"
        ]
    )
    if ds == "passing":
        # #31 keeps sack- and interception-only passers the oracle never had
        only_py = got.join(oracle.select(keys), on=keys, how="anti")
        # the 2023 fixture has exactly 12 such passers; fewer means #31 regressed
        assert only_py.height == 12 and (only_py["att"] == 0).all(), (
            "rows absent from the oracle must be the 12 att == 0 passers #31 keeps"
        )
        got = got.join(oracle.select(keys), on=keys, how="semi")
    extra = PYTHON_ONLY.get(ds, set())
    lost = sorted(extra - set(got.columns))
    assert not lost, f"{ds}: Python-only columns missing from the build: {lost}"
    got = got.drop(list(extra))
    drop = DIVERGENT_FROM_R.get(ds, set())
    corr = {c for c in corr if c not in drop}
    if drop:
        # a divergent column must still be EMITTED -- only its values differ
        gone = sorted(drop - set(got.columns))
        assert not gone, f"{ds}: divergent columns no longer produced: {gone}"
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
