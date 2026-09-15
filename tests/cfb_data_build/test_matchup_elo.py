"""Units G (ELO) and H (betting) vs the delivered 2025 matchup line.

Oracle: ``fixtures/matchup/matchup_line_2025.csv`` (773 FBS-vs-FBS games, bowls
excluded, CFP kept) -- its ``home/away_pregame_elo``, ``*_opp_elo_roll_*``,
``spread`` / ``spread_open`` / ``over_under`` / ``over_under_open`` columns.
Inputs: the CFBD ``/games`` rows for 2024 and 2025 and the 2025 ``/lines`` rows
(``cfbd_games_elo_*.parquet`` / ``cfbd_lines_2025.parquet``, fetched 2026-09-15).

Parity bars: ELO values and rolls exact to 1e-6 (they are sums / means /
medians of CFBD's own numbers); consensus lines exact to 1e-9. The line
carries the pipeline's display names (UConn -> Connecticut, ...), so the CFBD
names are mapped before joining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cfb_data_build.matchup_elo import (
    consensus_lines,
    elo_games,
    fill_pregame_elo,
    matchup_scope,
    opponent_elo_rolls,
    prior_final_elo,
    season_elo,
)

FIX = Path(__file__).parent / "fixtures" / "matchup"

#: the pipeline's ``team_name_mapping`` (CFBD name -> master name)
TEAM_NAME_MAPPING = {
    "UConn": "Connecticut",
    "UTSA": "UT San Antonio",
    "UL Monroe": "Louisiana Monroe",
    "Southern Miss": "Southern Mississippi",
    "Sam Houston": "Sam Houston State",
    "Massachusetts": "UMass",
    "Appalachian State": "App State",
}


def _oracle() -> pl.DataFrame:
    return pl.read_csv(
        FIX / "matchup_line_2025.csv", infer_schema_length=10000, null_values=["NA", ""]
    )


def _close(a: pl.Series, b: pl.Series, tol: float, name: str) -> None:
    x = a.cast(pl.Float64).to_numpy()
    y = b.cast(pl.Float64).to_numpy()
    assert (np.isnan(x) == np.isnan(y)).all(), f"{name}: null placement differs"
    ok = ~np.isnan(x)
    d = np.abs(x[ok] - y[ok])
    assert d.size == 0 or d.max() <= tol, f"{name}: max |diff| {d.max():.3e} > {tol}"


def test_pregame_elo_and_opponent_rolls_match_line() -> None:
    games = pl.read_parquet(FIX / "cfbd_games_elo_2025.parquet")
    prev = pl.read_parquet(FIX / "cfbd_games_elo_2024.parquet")
    filled, rolls = season_elo(games, prev)
    rolls = rolls.with_columns(pl.col("team").replace(TEAM_NAME_MAPPING))
    filled = filled.with_columns(
        pl.col("home_team").replace(TEAM_NAME_MAPPING),
        pl.col("away_team").replace(TEAM_NAME_MAPPING),
    )

    line = _oracle().select(
        "game_id",
        "home_team",
        "away_team",
        "home_pregame_elo",
        "away_pregame_elo",
        "home_opp_elo_roll_sum",
        "home_opp_elo_roll_avg",
        "home_opp_elo_roll_median",
        "away_opp_elo_roll_sum",
        "away_opp_elo_roll_avg",
        "away_opp_elo_roll_median",
    )
    joined = line.join(
        filled.select("game_id", "home_pregame_elo", "away_pregame_elo"),
        on="game_id",
        how="left",
        suffix="_py",
    )
    assert joined["home_pregame_elo_py"].null_count() == 0, (
        "every line game must be a CFBD game"
    )
    for side in ("home", "away"):
        _close(
            joined[f"{side}_pregame_elo_py"],
            joined[f"{side}_pregame_elo"],
            1e-6,
            f"{side}_pregame_elo",
        )
        r = rolls.select(
            "game_id",
            pl.col("team").alias(f"{side}_team"),
            pl.col("opp_elo_roll_sum").alias("py_sum"),
            pl.col("opp_elo_roll_avg").alias("py_avg"),
            pl.col("opp_elo_roll_median").alias("py_median"),
            matched=pl.lit(True),
        )
        j = line.join(r, on=["game_id", f"{side}_team"], how="left")
        # a null roll is legitimate (week 2 after an FCS opener); an UNMATCHED
        # (game_id, team) is not
        assert j["matched"].null_count() == 0, f"{side}: unmatched (game_id, team) rows"
        strict = j.filter(~pl.col("game_id").is_in(list(FROZEN_ARTIFACTS[side])))
        partitioned = j.filter(
            pl.col("game_id").is_in(list(FROZEN_ARTIFACTS[side]))
        ).height
        assert partitioned == len(FROZEN_ARTIFACTS[side])
        assert strict.height == j.height - partitioned
        for stat in ("sum", "avg", "median"):
            _close(
                strict[f"py_{stat}"],
                strict[f"{side}_opp_elo_roll_{stat}"],
                1e-6,
                f"{side}_opp_elo_roll_{stat}",
            )
    # the invariant the source violated: a team with a prior-season final never
    # takes the fallback -- San José State's 2024 final is its own 10 opponents
    sjs = rolls.filter(
        (pl.col("game_id") == 401760358) & (pl.col("team") == "San José State")
    )
    # expected: the roll at SJSU's last 2024 FBS-vs-FBS game, from the fixture
    # itself (a CFBD revision moves both sides together, never this literal)
    prev_filled = fill_pregame_elo(
        elo_games(prev), prior_final_elo(elo_games(prev)).clear()
    )
    prev_rolls = opponent_elo_rolls(
        prev_filled.join(
            matchup_scope(prev).select("game_id"), on="game_id", how="semi"
        )
    )
    final = (
        prev_rolls.filter(pl.col("team") == "San José State").sort("game_date").tail(1)
    )
    assert sjs["opp_elo_roll_sum"].item() == final["opp_elo_roll_sum"].item()
    assert abs(sjs["opp_elo_roll_avg"].item() - final["opp_elo_roll_avg"].item()) < 1e-9
    assert final["opp_elo_roll_sum"].item() > 10_000  # a real final, not a fallback


#: Week-1 rows whose oracle value is an artifact of the frozen master, not of
#: the rule (see the module docstring on rule 4), partitioned out of the strict
#: comparison and covered by the invariant above instead:
#:  401760358 home San José State -- the source's name join failed on the
#:      accent ("San Jose State" in its master), so it fell to the fallback
#:      quantile although the team had a 2024 final;
#:  401752798 away Missouri State -- new to FBS in 2025, so the fallback
#:      quantile applies; the source's quantile ran over its frozen master's
#:      2024 finals, which are not the CFBD-now finals (one revised ELO and the
#:      master's own duplicate rows);
#:  401761589 away App State -- CFBD revised game 401640992's ELO (+44 / +108)
#:      after the master froze, shifting App State's 2024 final by 108.
FROZEN_ARTIFACTS = {"home": (401760358,), "away": (401752798, 401761589)}


def test_consensus_lines_match_line() -> None:
    lines = pl.read_parquet(FIX / "cfbd_lines_2025.parquet")
    py = consensus_lines(lines)
    line = _oracle().select(
        "game_id", "spread", "spread_open", "over_under", "over_under_open"
    )
    j = line.join(py, on="game_id", how="left", suffix="_py")
    for c in ("spread", "spread_open", "over_under", "over_under_open"):
        _close(j[f"{c}_py"], j[c], 1e-9, c)


def test_pk_spread_is_zero() -> None:
    lines = pl.DataFrame(
        {
            "game_id": [1, 1, 2],
            "provider": ["a", "b", "a"],
            "spread": ["pk", "-3", None],
            "spread_open": ["PK", "-2.5", "x"],
            "over_under": ["50", "52", None],
            "over_under_open": [None, None, None],
        }
    )
    out = consensus_lines(lines).sort("game_id")
    assert out["spread"].to_list() == [-1.5, None]
    assert out["spread_open"].to_list() == [-1.25, None]
    assert out["over_under"].to_list() == [51.0, None]
    assert out["over_under_open"].null_count() == 2
