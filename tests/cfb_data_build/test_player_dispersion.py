"""Per-game dispersion and the rushing tiers on the player tables (TFD-5d).

Real carries: Auburn's first five 2024 games, three rushers (fixture README). The
expected values below are worked by hand from the CSV, not by the code under test.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from cfb_data_build.team_summaries import player_dispersion, rusher_tiers

FIX = (
    Path(__file__).parent
    / "fixtures"
    / "player_dispersion"
    / "rushers_2024_auburn_wk02_06.csv"
)
KEYS = ["pos_team_id", "rush_player_id"]
ALSTON, COBB, BROWN = 4685699, 4870642, 5089314


@pytest.fixture(scope="module")
def carries() -> pl.DataFrame:
    return pl.read_csv(
        FIX, schema_overrides={"game_id": pl.Utf8, "pos_team_id": pl.Utf8}
    )


def _row(df: pl.DataFrame, player: int) -> dict:
    return df.filter(pl.col("rush_player_id") == player).row(0, named=True)


def test_dispersion_hand_computed(carries):
    # Cobb's EPA/carry by game: 401628337 -0.781288 (1 carry), 401628352
    # -0.065002/7 = -0.009286, 401628363 0.163470 (1), 401628381 -0.533672 (1).
    # mean -0.290194; population sd 0.382484 -> boom above 0.092290 (the 0.163470
    # game), bust below -0.672678 (the -0.781288 game). Sorted
    # [-0.781288, -0.533672, -0.009286, 0.163470]: p10 at position 0.3 =
    # -0.781288 + 0.3 * 0.247616 = -0.707003; p90 at 2.7 = -0.009286 + 0.7 * 0.172756
    # = 0.111643.
    cobb = _row(player_dispersion(carries, KEYS), COBB)
    assert cobb["dispersion_games"] == 4
    assert cobb["EPAplay_sd"] == pytest.approx(0.382484, abs=1e-6)
    assert cobb["EPAplay_p10"] == pytest.approx(-0.707003, abs=1e-6)
    assert cobb["EPAplay_p90"] == pytest.approx(0.111643, abs=1e-6)
    assert (cobb["boom_rate"], cobb["bust_rate"]) == (0.25, 0.25)
    # Alston: one game (-1.861613) sits below mean - sd = -1.126307, none above 0.369017
    alston = _row(player_dispersion(carries, KEYS), ALSTON)
    assert alston["dispersion_games"] == 5
    assert (alston["boom_rate"], alston["bust_rate"]) == (0.0, 0.2)


def test_under_three_games_is_null(carries):
    brown = _row(player_dispersion(carries, KEYS), BROWN)
    assert brown["dispersion_games"] == 2
    for c in ("EPAplay_sd", "EPAplay_p10", "EPAplay_p90", "boom_rate", "bust_rate"):
        assert brown[c] is None, c


def test_identical_games_are_neither_boom_nor_bust():
    # sd == 0: a non-strict threshold would make every game both a boom and a bust
    flat = pl.DataFrame(
        {
            "pos_team_id": ["2"] * 3,
            "rush_player_id": [1] * 3,
            "game_id": ["a", "b", "c"],
            "EPA": [0.25] * 3,
        }
    )
    row = player_dispersion(flat, KEYS).row(0, named=True)
    assert row["EPAplay_sd"] == 0.0
    assert (row["boom_rate"], row["bust_rate"]) == (0.0, 0.0)


def test_tier_shares_and_stuff_rate_hand_counted(carries):
    tiers = rusher_tiers(carries, KEYS)
    # Alston, 26 carries: 13 for <= 4 yards (incl. a 4), 8 for 5-10, 5 for 11+ (incl.
    # an 11); 5 for <= 0 (five -2s)
    a = _row(tiers, ALSTON)
    assert (a["line_yards_share"], a["second_level_share"], a["open_field_share"]) == (
        13 / 26,
        8 / 26,
        5 / 26,
    )
    assert a["stuff_rate"] == 5 / 26
    # Cobb, 10 carries: yards 0, -1, 5, 8, 3, 1, 13, 5, 5, 2
    c = _row(tiers, COBB)
    assert (c["line_yards_share"], c["second_level_share"], c["open_field_share"]) == (
        5 / 10,
        4 / 10,
        1 / 10,
    )
    assert c["stuff_rate"] == 2 / 10


def test_one_score_split(carries):
    tiers = rusher_tiers(carries, KEYS)
    # Alston: the 4 carries with |score diff| > 8 at the snap are three at +19 in
    # 401628352 (0.298942, 0.568582, 3.766980) and one at -10 in 401628363 (-1.403703)
    a = _row(tiers, ALSTON)
    assert (a["EPAplay_one_score_n"], a["EPAplay_not_one_score_n"]) == (22, 4)
    assert a["EPAplay_not_one_score"] == pytest.approx(0.807700, abs=1e-6)
    assert a["EPAplay_one_score"] == pytest.approx(-0.372067, abs=1e-6)
    # Brown never carried outside one score: no rate, a count of 0
    b = _row(tiers, BROWN)
    assert b["EPAplay_not_one_score"] is None and b["EPAplay_not_one_score_n"] == 0
