"""Offline tests for predict.predict_matchup + build_season_strength_tables.

Uses a deterministic linear fake model (predicted MOV = slope * 5FRDiff) so the
WP / SoS / HFA logic is asserted independently of XGBoost's behavior on toy data.
The faithful LSU-Clemson ~0.70 validation lives in the integration suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pregame_wp.constants import HFA_COVID, HFA_NORMAL
from pregame_wp.predict import build_season_strength_tables, predict_matchup


class LinearModel:
    """predict(X) -> slope * X[:, 0]; mirrors an XGBRegressor's .predict API."""

    def __init__(self, slope: float = 3.0):
        self.slope = slope

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.slope * X[:, 0]


def _box_corpus() -> pd.DataFrame:
    """A 4-team, 1-season corpus where A > C > B > D by season-avg 5FR."""
    rows = []
    # (GameID, Week, [(team, 5FR, 5FRDiff, PtsDiff), (team, ...)])
    games = [
        ("g1", 1, [("A", 8.0, 4.0, 21), ("B", 4.0, -4.0, -21)]),
        ("g2", 2, [("A", 8.0, 3.0, 14), ("C", 5.0, -3.0, -14)]),
        ("g3", 3, [("A", 8.0, 5.0, 28), ("D", 3.0, -5.0, -28)]),
        ("g4", 1, [("C", 6.0, 2.0, 10), ("B", 4.0, -2.0, -10)]),
        ("g5", 2, [("C", 5.0, 1.0, 7), ("D", 3.0, -1.0, -7)]),
        ("g6", 3, [("B", 4.0, 1.0, 3), ("D", 3.0, -1.0, -3)]),
    ]
    for gid, wk, pair in games:
        for team, fr, frd, pd_ in pair:
            rows.append({
                "GameID": gid, "Season": 2019, "Week": wk, "Team": team,
                "5FR": fr, "5FRDiff": frd, "PtsDiff": pd_,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def corpus():
    return _box_corpus()


# --- build_season_strength_tables --------------------------------------------

def test_team_strength_is_season_mean(corpus):
    ts, _ = build_season_strength_tables(corpus)
    a = ts[(ts["Team"] == "A") & (ts["Season"] == 2019)]["5FR"].iloc[0]
    assert a == pytest.approx(8.0)  # A's three games all 8.0
    # C plays three games: g2=5.0, g4=6.0, g5=5.0 → mean 16/3
    c = ts[(ts["Team"] == "C") & (ts["Season"] == 2019)]["5FR"].iloc[0]
    assert c == pytest.approx((5.0 + 6.0 + 5.0) / 3)


def test_opponent_game_ids_columns_and_opponents(corpus):
    _, og = build_season_strength_tables(corpus)
    assert set(og.columns) == {"GameID", "Team", "Opponent", "Season", "Week"}
    a_opps = set(og[og["Team"] == "A"]["Opponent"])
    assert a_opps == {"B", "C", "D"}


# --- predict_matchup ----------------------------------------------------------

def test_returns_pair(corpus):
    out = predict_matchup("A", "B", 2019, stored_game_boxes=corpus,
                          model=LinearModel(), mu=0.0, std=10.0)
    assert isinstance(out, list) and len(out) == 2
    wp, mov = out
    assert 0.0 < wp < 1.0


def test_stronger_home_team_favored(corpus):
    wp, mov = predict_matchup("A", "B", 2019, stored_game_boxes=corpus,
                              model=LinearModel(), mu=0.0, std=10.0)
    assert mov > 0  # A stronger than B → positive home MOV
    assert wp > 0.5


def test_weaker_home_team_underdog(corpus):
    wp, mov = predict_matchup("B", "A", 2019, stored_game_boxes=corpus,
                              model=LinearModel(), mu=0.0, std=10.0)
    assert mov < 0
    assert wp < 0.5


def test_hfa_increases_home_mov(corpus):
    wp_no, mov_no = predict_matchup("A", "B", 2019, stored_game_boxes=corpus,
                                    model=LinearModel(), mu=0.0, std=10.0,
                                    adjust_hfa=False)
    wp_hfa, mov_hfa = predict_matchup("A", "B", 2019, stored_game_boxes=corpus,
                                      model=LinearModel(), mu=0.0, std=10.0,
                                      adjust_hfa=True)
    assert mov_hfa == pytest.approx(mov_no + HFA_NORMAL)
    assert wp_hfa > wp_no


def test_covid_hfa_is_smaller(corpus):
    _, mov_normal = predict_matchup("A", "B", 2019, stored_game_boxes=corpus,
                                    model=LinearModel(), mu=0.0, std=10.0,
                                    adjust_hfa=True, adjust_covid=False)
    _, mov_covid = predict_matchup("A", "B", 2019, stored_game_boxes=corpus,
                                   model=LinearModel(), mu=0.0, std=10.0,
                                   adjust_hfa=True, adjust_covid=True)
    assert mov_covid == pytest.approx(mov_normal - (HFA_NORMAL - HFA_COVID))


def test_week0_falls_back_to_prior_season(corpus):
    # Add a 2020 season with no games for A → week=0 must use 2019 ratings.
    box = corpus.copy()
    # team with no 2020 rows: ask for 2020 week 0 → applied_year 2019
    wp, mov = predict_matchup("A", "B", 2020, week=0, stored_game_boxes=box,
                              model=LinearModel(), mu=0.0, std=10.0)
    # A should still be favored using the prior-season basis
    assert mov > 0


def test_unknown_team_uses_national_average(corpus):
    # "Z" has no rows; its avg_ffr falls back to natl_avg → near-even matchup
    wp, mov = predict_matchup("Z", "Y", 2019, stored_game_boxes=corpus,
                              model=LinearModel(), mu=0.0, std=10.0)
    assert mov == pytest.approx(0.0, abs=1e-9)
    assert wp == pytest.approx(0.5, abs=1e-9)


def test_conference_sos_skips_fbs_independents(corpus):
    confs = {"A": "FBS Independents", "B": "SEC", "C": "SEC", "D": "ACC"}
    # Should run without error and not apply the conference adjustment for A.
    wp, mov = predict_matchup("A", "B", 2019, stored_game_boxes=corpus,
                              model=LinearModel(), mu=0.0, std=10.0,
                              conferences=confs)
    assert 0.0 < wp < 1.0


def test_returning_production_adjustment_applies_in_preseason(corpus):
    talent = {"A": 0.9, "B": 0.5}
    ret = {"A": 0.8, "B": 0.4}
    wp_w3, _ = predict_matchup("A", "B", 2019, week=3, stored_game_boxes=corpus,
                               model=LinearModel(), mu=0.0, std=10.0,
                               roster_talent=lambda t, y: talent[t],
                               returning_production=lambda t, y: ret[t])
    # B is the weaker returning side → its 5FR is scaled down → A still favored
    assert wp_w3 > 0.5
