"""Leaderboard rank + percentile contract (``_attach_leader_ranks`` / ``_pct``).

These use a small constructed frame on purpose. The parity suite next door is the
real-data gate for the aggregation itself; what is under test here is a
deterministic arithmetic contract -- direction, the qualifier gate, and above all
what happens to a NULL metric -- and a null at a known position is exactly what a
real season will not hand you on demand.

The null case is the one that matters. ``_rank`` reproduces R's ``na.last = TRUE``
and gives a null metric a TRAILING rank, which is correct for a leaderboard; if
the percentile were derived from that rank alone, a missing stat would render as
the 0th percentile -- an unknown displayed to a reader as the worst value on the
board -- and would also inflate the denominator for everyone else.
"""

from __future__ import annotations

import polars as pl

from cfb_data_build.team_summaries import _attach_leader_ranks

KEYS = ["pos_team_id", "player_id"]


def _frame(values: list[float | None], *, plays: list[int] | None = None) -> pl.DataFrame:
    n = len(values)
    return pl.DataFrame(
        {
            "pos_team_id": list(range(1, n + 1)),
            "player_id": list(range(101, 101 + n)),
            "plays": plays if plays is not None else [10] * n,
            "epa": values,
        }
    )


def _attach(df: pl.DataFrame, **kw) -> pl.DataFrame:
    return _attach_leader_ranks(df, keys=KEYS, min_expr=pl.col("plays") >= 5, rank_cols=["epa"], **kw)


def test_a_null_metric_yields_a_null_percentile_not_a_zero():
    out = _attach(_frame([3.0, 1.0, 2.0, None])).sort("player_id")
    pct = out["epa_pct"].to_list()

    # the null row is null in the percentile, NOT 0.0 and NOT the worst placing
    assert pct[3] is None
    # ...and it still carries a trailing RANK, which is the leaderboard behaviour
    assert out["epa_rank"].to_list()[3] == 4.0


def test_a_null_metric_does_not_depress_everyone_elses_percentile():
    without_null = _attach(_frame([3.0, 1.0, 2.0])).sort("player_id")["epa_pct"].to_list()
    with_null = _attach(_frame([3.0, 1.0, 2.0, None])).sort("player_id")["epa_pct"].to_list()

    # n counts qualifiers that HAVE the metric, so adding a null row changes nothing
    assert with_null[:3] == without_null


def test_the_scale_is_symmetric_and_pins_nobody_to_zero_or_one_hundred():
    out = _attach(_frame([3.0, 1.0, 2.0])).sort("player_id")
    pct = out["epa_pct"].to_list()
    n = 3

    assert pct[0] == 100 * n / (n + 1)  # best  -> 75.0, not 100
    assert pct[1] == 100 * 1 / (n + 1)  # worst -> 25.0, not 0
    assert all(0 < p < 100 for p in pct)


def test_an_ascending_column_ranks_low_as_good():
    df = pl.DataFrame(
        {
            "pos_team_id": [1, 2, 3],
            "player_id": [101, 102, 103],
            "plays": [10, 10, 10],
            "fumbles": [0, 2, 5],
        }
    )
    out = _attach_leader_ranks(
        df,
        keys=KEYS,
        min_expr=pl.col("plays") >= 5,
        rank_cols=["fumbles"],
        asc_cols=["fumbles"],
    ).sort("player_id")

    # fewest fumbles is the best placing
    assert out["fumbles_rank"].to_list() == [1.0, 2.0, 3.0]
    assert out["fumbles_pct"].to_list()[0] > out["fumbles_pct"].to_list()[2]


def test_non_qualifiers_get_no_rank_and_no_percentile():
    out = _attach(_frame([3.0, 1.0, 2.0], plays=[10, 10, 1])).sort("player_id")

    assert out["epa_pct"].to_list()[2] is None
    assert out["epa_rank"].to_list()[2] is None
    # the two qualifiers are ranked among THEMSELVES, not among all three
    assert out["epa_rank"].to_list()[:2] == [1.0, 2.0]
