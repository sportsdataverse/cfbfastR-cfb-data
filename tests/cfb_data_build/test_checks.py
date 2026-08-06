"""Regression tests for the build-time gates in ``cfb_data_build.checks``.

These exist because a gate that has never been run against the bug it targets
is decoration.  ``assert_passer_epa_includes_sacks`` shipped wired into
``build_team_summaries`` but *unimported* -- it raised ``NameError`` rather than
checking anything -- and its first threshold (0.45 EPA/dropback) was set by
assumption.  Rebuilding 2004-2025 showed the real post-fix maximum is 0.7666
(Jameis Winston, 2013), ABOVE the 0.7641 the broken build produced, so the
threshold was replaced with exact identities.  Both facts were found only by
running the gate against real seasons.
"""

from __future__ import annotations

import polars as pl
import pytest

from cfb_data_build.checks import assert_passer_epa_includes_sacks

#: Dylan Raiola, Nebraska, 2025 -- the reference case from cfbfastR-cfb-data#30,
#: post-fix, taken from the rebuilt release.
RAIOLA = {
    "passer_player_name": "Dylan Raiola",
    "TEPA": 32.07396871224046,
    "EPAplay": 0.12778473590534048,
    "dropbacks": 251.0,
    "sacked": 27,
    "pass_int": 5,
    "sack_epa": -44.13125030696392,
    "int_epa": -28.94,
}


def _passers(**over) -> pl.DataFrame:
    row = dict(RAIOLA)
    row.update(over)
    return pl.DataFrame({k: [v] for k, v in row.items()})


def test_accepts_the_real_post_fix_row() -> None:
    assert assert_passer_epa_includes_sacks(_passers()) is None


def test_accepts_a_legitimate_outlier_season() -> None:
    """Jameis Winston 2013: 0.7666 EPA/dropback is real, and must pass.

    The gate's first version rejected this, which would have blocked the
    2004-2025 republish on a Heisman season.
    """
    winston = _passers(
        passer_player_name="Jameis Winston",
        dropbacks=321.0,
        EPAplay=0.7666,
        TEPA=0.7666 * 321.0,
    )
    assert assert_passer_epa_includes_sacks(winston) is None


def test_raises_when_epaplay_spans_a_different_play_set() -> None:
    """The #30 shape: EPAplay was TEPA/plays while dropbacks was att+sacked.

    Raiola's published row -- TEPA 105.142 over 219 attempts, EPAplay 0.4801,
    but 251 dropbacks -- is exactly this, and the identity catches it.
    """
    pre_fix = _passers(TEPA=105.142, EPAplay=0.4801, dropbacks=251.0)
    with pytest.raises(ValueError, match="TEPA != EPAplay"):
        assert_passer_epa_includes_sacks(pre_fix)


def test_raises_when_sack_or_int_epa_is_not_negative() -> None:
    """A non-negative aggregate means the name->id join stopped delivering."""
    for col in ("sack_epa", "int_epa"):
        with pytest.raises(ValueError, match=col):
            assert_passer_epa_includes_sacks(_passers(**{col: 0.0}))


def test_raises_on_empty_and_on_missing_columns() -> None:
    with pytest.raises(ValueError, match="EMPTY"):
        assert_passer_epa_includes_sacks(_passers().clear())
    with pytest.raises(ValueError, match="missing"):
        assert_passer_epa_includes_sacks(_passers().drop("sack_epa"))
