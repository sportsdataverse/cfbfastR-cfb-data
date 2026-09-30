"""A new team_summaries column must not silently join the lean model inputs.

``family_of`` files a column matching no fragment under "other", and "other" is
in ``LEAN_FAMILIES`` -- it is meant for the ``rt_*`` cfb_ratings block and the
roster context. A summaries column that lands there joins the recommended
model's inputs at the next weekly republish + retrain with nobody deciding it
should (IF-2's Five Factors columns would have added ten).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from cfb_data_build.team_summaries import (
    _add_turnover_luck,
    _havoc_and_expected_turnovers,
)
from cfb_model_build.cfb_higher_models.data import feature_columns
from cfb_model_build.cfb_higher_models.train_game import (
    LEAN_FAMILIES,
    family_of,
    lean_features,
)

from tests.cfb_data_build.test_factor_extension_columns import _team_off
from tests.cfb_data_build.test_five_factors_columns import _drives, _overall, _plays

#: the team_summaries columns in the lean set. Adding one is a model change:
#: re-run the walk-forward in train_game.py before editing this.
LEAN_SUMMARY_COLUMNS = {
    "adj_off_epa",
    "adj_def_epa",
    "net_adj_epa",
    "off_strength_faced",
    "def_strength_faced",
    "total_available_yards_off",
    "total_available_yards_def",
    "total_available_yards_margin",
    "total_gained_yards_off",
    "total_gained_yards_def",
    "total_gained_yards_margin",
    "available_yards_pct_off",
    "available_yards_pct_def",
    "available_yards_pct_margin",
}
_PUBLISHED = Path(__file__).parents[1] / "cfb" / "team_summaries" / "parquet"


def _summary_columns() -> set[str]:
    """What the builder emits now, plus the newest published table's columns."""
    built = _plays()
    cols = set(feature_columns(_overall(built))) | set(feature_columns(_drives(built)))
    # the whole-team extras built outside _summarize_team / _drives: havoc EPA, expected
    # turnovers and turnover luck (the luck step also needs the actual counts)
    extras = _havoc_and_expected_turnovers(_team_off()).with_columns(
        turnovers_off=pl.lit(1.0), turnovers_def=pl.lit(1.0), turnover_margin=pl.lit(0.0)
    )
    cols |= set(feature_columns(_add_turnover_luck(extras)))
    newest = sorted(_PUBLISHED.glob("cfb_team_summaries_*.parquet"))[-1:]
    for path in newest:
        cols |= set(feature_columns(pl.DataFrame(schema=pl.read_parquet_schema(path))))
    return cols


def test_every_summary_column_has_a_named_family():
    stray = sorted(c for c in _summary_columns() if family_of(f"{c}_diff") == "other")
    assert not stray, f"give these a family in train_game.FAMILIES: {stray}"


def test_the_lean_summary_columns_are_pinned():
    diffs = [f"{c}_diff" for c in sorted(_summary_columns())]
    lean = {c.removesuffix("_diff") for c in lean_features(diffs)}
    assert lean <= LEAN_SUMMARY_COLUMNS, sorted(lean - LEAN_SUMMARY_COLUMNS)


def test_five_factors_columns_stay_out_of_the_lean_set():
    for c in (
        "explosive_margin",
        "pts_per_opp_off",
        "pts_per_opp_def_n",
        "pts_per_opp_margin",
        "turnovers_off",
        "turnovers_def_n",
        "turnover_margin",
    ):
        assert family_of(f"{c}_diff") == "five_factors"
    assert "five_factors" not in LEAN_FAMILIES


def test_drive_efficiency_columns_stay_out_of_the_lean_set():
    for c in (
        "pts_per_drive_off",
        "pts_per_drive_def",
        "pts_per_drive_margin",
        "pts_per_drive_off_n",
        "pts_per_drive_def_n",
    ):
        assert family_of(f"{c}_diff") == "drive_efficiency"
    assert "drive_efficiency" not in LEAN_FAMILIES
