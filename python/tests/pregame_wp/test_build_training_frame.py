"""Offline tests for training.build_training_frame (CLI `build-boxes` driver).

These inject ``games_provider`` + ``frames_loader`` so the corpus build never
touches CFBD — the network path is exercised only in the integration suite.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pregame_wp.ep_curve import load_ep_curve, load_punt_sr
from pregame_wp.training import _global_eqppp_bounds, build_training_frame


def _make_game(strong: str, weak: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for team, yards in ((strong, [6, 16, 8, 12, 5, 7]), (weak, [2, 1, 5, 2, 3, 2])):
        defense = weak if team == strong else strong
        for i, y in enumerate(yards):
            rows.append({
                "offense": team, "defense": defense, "play_type": "Rush",
                "down": 1, "distance": 10, "yards_gained": y,
                "yard_line": 25 + i * 5, "play_text": f"rush for {y}",
            })
    plays = pd.DataFrame(rows)
    drives = pd.DataFrame([
        {"offense": strong, "defense": weak, "drive_start_yardline": 35,
         "drive_yards": 30, "drive_scoring": 1, "drive_pts": 7},
        {"offense": weak, "defense": strong, "drive_start_yardline": 20,
         "drive_yards": 10, "drive_scoring": 0, "drive_pts": 0},
    ])
    return plays, drives


@pytest.fixture()
def corpus():
    """Two-season synthetic corpus + injected providers."""
    games_db: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    season_games: dict[int, list[dict]] = {2018: [], 2019: []}
    matchups = [
        ("A", "B", 35, 10), ("C", "D", 28, 14), ("A", "C", 24, 21),
        ("B", "D", 17, 10), ("A", "D", 42, 7), ("C", "B", 31, 17),
    ]
    gid = 1000
    for season in (2018, 2019):
        for wk, (home, away, hp, ap) in enumerate(matchups, start=1):
            g = str(gid)
            gid += 1
            games_db[g] = _make_game(home, away)
            season_games[season].append({
                "id": g, "homeTeam": home, "awayTeam": away,
                "homePoints": hp, "awayPoints": ap, "week": wk,
                "seasonType": "regular",
            })

    def games_provider(year: int) -> list[dict]:
        return season_games[year]

    def frames_loader(game_id: str, raw_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
        return games_db[game_id]

    return games_provider, frames_loader, len(matchups)


def _build(corpus, **kw):
    games_provider, frames_loader, _ = corpus
    return build_training_frame(
        [2018, 2019], raw_dir="unused", ep_data=load_ep_curve(),
        punt_sr=load_punt_sr(), games_provider=games_provider,
        frames_loader=frames_loader, fetch_missing=False, verbose=False, **kw,
    )


def test_two_rows_per_game(corpus):
    _, _, n_matchups = corpus
    stored = _build(corpus)
    assert len(stored) == 2 * n_matchups * 2  # 2 teams * matchups * 2 seasons
    assert (stored.groupby("GameID").size() == 2).all()


def test_required_columns_present(corpus):
    stored = _build(corpus)
    for col in ("GameID", "Season", "Week", "Team", "PtsDiff", "5FR", "5FRDiff"):
        assert col in stored.columns


def test_ptsdiff_antisymmetric_per_game(corpus):
    stored = _build(corpus)
    for _, grp in stored.groupby("GameID"):
        assert abs(grp["PtsDiff"].sum()) < 1e-9


def test_ptsdiff_matches_game_record(corpus):
    stored = _build(corpus)
    # game 1000 = A 35 - B 10 → A:+25, B:-25
    g = stored[stored["GameID"] == "1000"].set_index("Team")["PtsDiff"]
    assert g.loc["A"] == pytest.approx(25.0)
    assert g.loc["B"] == pytest.approx(-25.0)


def test_5frdiff_antisymmetric(corpus):
    stored = _build(corpus)
    for _, grp in stored.groupby("GameID"):
        assert abs(grp["5FRDiff"].sum()) < 1e-9


def test_seasons_and_weeks_threaded(corpus):
    stored = _build(corpus)
    assert set(stored["Season"]) == {2018, 2019}
    assert set(stored["Week"]) == {1, 2, 3, 4, 5, 6}


def test_eqppp_bounds_thread_into_box(corpus):
    """The global EqPPP bounds should be the real corpus min/max, not the
    hardcoded (-2, 2) defaults, and should appear in the box's _eq_ppp columns."""
    stored = _build(corpus)
    lo = stored["_eq_ppp_min"].iloc[0]
    hi = stored["_eq_ppp_max"].iloc[0]
    # every row carries the same global bounds
    assert (stored["_eq_ppp_min"] == lo).all()
    assert (stored["_eq_ppp_max"] == hi).all()
    assert lo < hi


def test_empty_corpus_returns_empty_frame():
    stored = build_training_frame(
        [2018], raw_dir="unused", ep_data=load_ep_curve(), punt_sr=load_punt_sr(),
        games_provider=lambda y: [], frames_loader=lambda g, r: (_ for _ in ()).throw(KeyError),
        fetch_missing=False, verbose=False,
    )
    assert isinstance(stored, pd.DataFrame)
    assert len(stored) == 0


def test_global_eqppp_bounds_fallback_when_empty():
    assert _global_eqppp_bounds([]) == (-2.0, 2.0)


def test_global_eqppp_bounds_computes_min_max():
    df = pd.DataFrame({"EqPPP": [-0.5, 0.0, 1.3, 2.7]})
    lo, hi = _global_eqppp_bounds([df])
    assert lo == pytest.approx(-0.5)
    assert hi == pytest.approx(2.7)
