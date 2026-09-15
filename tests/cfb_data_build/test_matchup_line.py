"""Unit J -- the assembled matchup line -- vs the delivered 2025 rows.

Oracle: ``fixtures/matchup/matchup_line_2025.csv`` (773 rows x 268 columns).
Inputs are the other committed oracles, so this is a pure assembly check:
the R as-of features (``team_features_asof_2025.csv``), the prior season's
full-season features (``team_features_full_2024.csv``, the pipeline's own
priors file) and pace (``team_pace_full.csv``), the season's ``pace_hist``
slice, CFBD games (2024, 2025) and lines (2025).

Bars: the 268-column order exact; row set exact; meta exact; the 136 feature
columns, 8 pace columns and 6 roll columns within 1e-6 (the three frozen-
master ELO artifacts of ``test_matchup_elo`` partitioned out); consensus lines
within 1e-9; ``game_type`` / ``prev_season`` exact. Side inputs, weather and
venue / team metadata are Phase-6 joins and are not asserted here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cfb_data_build.matchup_features import TEAM_NAME_MAPPING
from cfb_data_build.matchup_line import (
    FEATURE_COLS,
    LINE_COLUMNS,
    PACE_COLS,
    build_matchup_line,
)
from tests.cfb_data_build.test_matchup_elo import FROZEN_ARTIFACTS

FIX = Path(__file__).parent / "fixtures" / "matchup"


def _rd(name: str) -> pl.DataFrame:
    return pl.read_csv(FIX / name, infer_schema_length=10000, null_values=["NA", ""])


def _line() -> tuple[pl.DataFrame, pl.DataFrame]:
    py = build_matchup_line(
        games=pl.read_parquet(FIX / "cfbd_games_elo_2025.parquet"),
        prev_games=pl.read_parquet(FIX / "cfbd_games_elo_2024.parquet"),
        team_features=_rd("team_features_asof_2025.csv"),
        prev_features=_rd("team_features_full_2024.csv"),
        pace_hist=_rd("pace_hist_2025.csv"),
        prev_pace_full=_rd("team_pace_full.csv").filter(pl.col("season") == 2024),
        lines=pl.read_parquet(FIX / "cfbd_lines_2025.parquet"),
    )
    oracle = _rd("matchup_line_2025.csv")
    return py.sort("game_id"), oracle.sort("game_id")


def _close(py: pl.Series, oracle: pl.Series, tol: float, name: str) -> None:
    a = py.cast(pl.Float64).to_numpy()
    b = oracle.cast(pl.Float64).to_numpy()
    assert (np.isnan(a) == np.isnan(b)).all(), f"{name}: null placement differs"
    ok = ~np.isnan(a)
    d = np.abs(a[ok] - b[ok])
    assert d.size == 0 or d.max() <= tol, f"{name}: max |diff| {d.max():.3e} > {tol}"


def test_columns_and_rows_match_the_delivered_line() -> None:
    py, oracle = _line()
    assert list(LINE_COLUMNS) == oracle.columns
    assert py.columns == oracle.columns
    assert py["game_id"].to_list() == oracle["game_id"].to_list()
    for c in (
        "season",
        "week",
        "season_type",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "home_division",
        "away_division",
        "home_points",
        "away_points",
        "home_mov",
        "game_type",
        "venue_id",
        "neutral_site",
        "conference_game",
    ):
        assert py[c].cast(pl.Utf8).to_list() == oracle[c].cast(pl.Utf8).to_list(), c
    assert (
        py["start_date"].cast(pl.Utf8).to_list()
        == oracle["start_date"].cast(pl.Utf8).to_list()
    )


def test_features_pace_and_lines_match() -> None:
    py, oracle = _line()
    for side in ("home", "away"):
        for c in FEATURE_COLS:
            _close(py[f"{side}_{c}"], oracle[f"{side}_{c}"], 1e-6, f"{side}_{c}")
            _close(
                py[f"prev_{side}_{c}"],
                oracle[f"prev_{side}_{c}"],
                1e-6,
                f"prev_{side}_{c}",
            )
        # pace: strict except for the six renamed teams, where the source's
        # pace_hist join failed on the spelling (its pace file says "UConn", its
        # line "Connecticut") and every week silently took the prior-season
        # full-season fallback; the port joins on one spelling and carries the
        # real as-of value, asserted against pace_hist directly below.
        renamed = py[f"{side}_team"].is_in(RENAMED)
        for c in PACE_COLS:
            _close(
                py.filter(~renamed)[f"{side}_{c}"],
                oracle.filter(~renamed)[f"{side}_{c}"],
                1e-6,
                f"{side}_{c}",
            )
        ph = _rd("pace_hist_2025.csv").with_columns(
            pl.col("team").replace(TEAM_NAME_MAPPING)
        )
        j = py.filter(renamed).join(
            ph.rename({"team": f"{side}_team"}),
            on=["game_id", f"{side}_team"],
            how="left",
        )
        assert j.height >= 20
        for c in PACE_COLS:
            got, want = (
                j[f"{side}_{c}"],
                j[f"{c}_right"] if f"{c}_right" in j.columns else j[c],
            )
            both = got.is_not_null() & want.is_not_null()
            assert (got.filter(both) - want.filter(both)).abs().max() <= 1e-6, c
    for c in ("spread", "spread_open", "over_under", "over_under_open"):
        _close(py[c], oracle[c], 1e-9, c)


RENAMED = tuple(TEAM_NAME_MAPPING.values())


def test_elo_and_rolls_match_outside_the_frozen_artifacts() -> None:
    py, oracle = _line()
    for side in ("home", "away"):
        _close(
            py[f"{side}_pregame_elo"],
            oracle[f"{side}_pregame_elo"],
            1e-6,
            f"{side}_pregame_elo",
        )
        keep = ~py["game_id"].is_in(list(FROZEN_ARTIFACTS[side]))
        for stat in ("avg", "median", "sum"):
            c = f"{side}_opp_elo_roll_{stat}"
            _close(py.filter(keep)[c], oracle.filter(keep)[c], 1e-6, c)
