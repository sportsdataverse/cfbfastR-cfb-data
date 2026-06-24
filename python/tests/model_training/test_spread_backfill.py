"""Unit tests for the odds-based spread backfill (no live data).

Covers the three correctness-critical pieces that the WP model depends on:
  1. crosswalk-free home/away resolution (_abbr_to_name + game_desc parsing) — each
     focus team plays multiple games so the modal-name heuristic resolves (as on the
     real 8k-game corpus),
  2. the SIGN convention (consensus negated into ESPN pbp_full's positive==home-favored),
  3. apply only touches missing-spread games and recomputes start.spread_time exactly
     (+pos_team_spread * exp(-4 * elapsed_share)).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from model_training import spread_backfill as sb


def _odds_frame() -> pl.DataFrame:
    """A double round-robin of teams A/B/C so EVERY team is multi-game and the
    modal-name resolver is unambiguous for all of them (as on the real corpus).

    Per game we pass the HOME team's market line ``hl`` (negative == home favored);
    the away team gets ``-hl``. Test games of interest:
      g1 "B@A": A home favored 7  -> consensus home_spread +7.0
      g6 "B@C": C home underdog 4 -> consensus home_spread -4.0
    """
    rows: list[dict] = []

    def add(gid, desc, abbr, mt, line, book):
        rows.append({"game_id": gid, "game_desc": desc, "abbr": abbr,
                     "market_type": mt, "lines": float(line), "book": book})

    # (game_id, away, home, home_market_line)  — every team plays 4 games.
    games = [
        (1, "B", "A", -7.0), (2, "C", "A", -10.0),
        (3, "A", "B", -3.0), (4, "C", "B", -5.0),
        (5, "A", "C", -6.0), (6, "B", "C", 4.0),   # g6: C home underdog (line +4)
    ]
    for gid, away, home, hl in games:
        for bk, jitter in [("x", 0.0), ("y", 0.5)]:
            add(gid, f"{away}@{home}", home, "spread", hl - jitter, bk)
            add(gid, f"{away}@{home}", away, "spread", -(hl - jitter), bk)
        add(gid, f"{away}@{home}", home, "total", 55.0, "x")
        add(gid, f"{away}@{home}", home, "total", 57.0, "y")
    return pl.DataFrame(rows)


def test_consensus_sign_is_espn_convention(tmp_path):
    """ESPN pbp_full convention: POSITIVE home_spread == home favored."""
    p = tmp_path / "odds.parquet"
    _odds_frame().write_parquet(p)
    sp = sb.load_consensus_spreads(p)
    d = {r["game_id"]: r for r in sp.iter_rows(named=True)}
    # g1 home line median([-7.0,-7.5])=-7.25 -> negated -> +7.25 (home favored)
    assert abs(d[1]["home_spread"] - 7.25) < 1e-9
    # g6 home line median([4.0,3.5])=3.75 -> negated -> -3.75 (home underdog)
    assert abs(d[6]["home_spread"] - (-3.75)) < 1e-9
    assert abs(d[1]["over_under"] - 56.0) < 1e-9


def test_apply_only_touches_missing_and_recomputes_spread_time(tmp_path):
    """Backfill fills missing games only; spread_time = +pos*exp(-4*elapsed)."""
    p = tmp_path / "odds.parquet"
    _odds_frame().write_parquet(p)
    spreads = sb.load_consensus_spreads(p)
    pbp = pl.DataFrame({
        "game_id": [1, 4, 99],                     # G1 missing, G4 has real spread, G99 no odds
        "homeTeamSpread": [None, 2.0, None],
        "gameSpreadAvailable": [False, True, False],
        "gameSpread": [None, 2.0, None],
        "overUnder": [None, 50.0, None],
        "start.is_home": [1, 1, 1],
        "start.pos_team_spread": [None, 2.0, None],
        "start.spread_time": [None, 1.0, None],
        "start.elapsed_share": [0.25, 0.25, 0.25],
    })
    out, stats = sb.apply_spread_backfill(pbp, spreads)
    by = {r["game_id"]: r for r in out.iter_rows(named=True)}
    assert stats["games_backfilled"] == 1                        # only G1
    assert abs(by[1]["start.pos_team_spread"] - 7.25) < 1e-9     # home posteam, +7.25
    assert abs(by[1]["start.spread_time"] - 7.25 * np.exp(-4 * 0.25)) < 1e-6
    assert by[4]["start.pos_team_spread"] == 2.0 and by[4]["start.spread_time"] == 1.0
    assert by[99]["start.pos_team_spread"] is None


def test_away_posteam_flips_sign(tmp_path):
    """For an away possessing team, pos_team_spread = -homeTeamSpread."""
    p = tmp_path / "odds.parquet"
    _odds_frame().write_parquet(p)
    spreads = sb.load_consensus_spreads(p)
    pbp = pl.DataFrame({
        "game_id": [1], "homeTeamSpread": [None], "gameSpreadAvailable": [False],
        "gameSpread": [None], "overUnder": [None], "start.is_home": [0],
        "start.pos_team_spread": [None], "start.spread_time": [None],
        "start.elapsed_share": [0.0],
    })
    out, _ = sb.apply_spread_backfill(pbp, spreads)
    # home_spread +7.25, away posteam -> -7.25; elapsed 0 -> spread_time == -7.25
    assert abs(out["start.pos_team_spread"][0] - (-7.25)) < 1e-9
    assert abs(out["start.spread_time"][0] - (-7.25)) < 1e-9
