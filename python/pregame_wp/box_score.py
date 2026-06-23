"""Full-game box score: per-team Five-Factors stats + 5FR + 5FRDiff.

Port of win-prob.ipynb cell 24 calculate_box_score.
"""
from __future__ import annotations

import polars as pl

from .five_factors import calculate_five_factors_rating
from .play_features import add_play_features
from .team_stats import (
    generate_team_drive_stats,
    generate_team_play_stats,
    generate_team_st_stats,
    generate_team_turnover_stats,
)

_OFF_TYPES = [
    "Rush", "Pass Reception", "Pass Incompletion", "Rushing Touchdown",
    "Passing Touchdown", "Fumble Recovery (Opponent)", "Sack",
]
_ST_TYPES = ["Kickoff", "Punt", "Field Goal Good", "Field Goal Missed", "Kickoff Return TD"]
_BAD_TYPES = ["Interception", "Sack", "Fumble Recovery (Opponent)"]


def calculate_box_score_from_frames(
    plays: pl.DataFrame,
    drives: pl.DataFrame,
    ep_data: list[float],
    punt_sr: dict[int, float],
    eq_ppp_global_min: float = -2.0,
    eq_ppp_global_max: float = 2.0,
) -> pl.DataFrame:
    """Compute per-team 5FR box score from pre-loaded play/drive frames.

    Args:
        plays: Play-by-play with columns offense, defense, play_type, down,
               distance, yards_gained, yard_line, play_text.
        drives: Drive log with offense, defense, drive_start_yardline,
                drive_yards, drive_scoring, drive_pts.
        ep_data: EP curve list (len 101).
        punt_sr: {yardline: ExpPuntNet} dict.
        eq_ppp_global_min: Global EqPPP min from training PBP (for expl index domain).
        eq_ppp_global_max: Global EqPPP max from training PBP.

    Returns:
        DataFrame with one row per team: OffSR, OffER, AvgEqPPP, IsoPPP,
        OppRate, OppEff, OppPPD, OppSR, ExpTO, ActualTO, HavocRate, SackRate,
        KickoffEqPPP, PuntEqPPP, PuntReturnEqPPP, 5FR, 5FRDiff.
    """
    teams = sorted(plays["offense"].unique().to_list())
    if len(teams) != 2:
        raise ValueError(f"Expected exactly 2 teams, got {teams}")

    # Enrich plays with EqPPP / play_successful / play_explosive
    plays = add_play_features(plays, ep_data, _ST_TYPES, _BAD_TYPES)

    rows = []
    for team in teams:
        opponent = [t for t in teams if t != team][0]
        play_stats = generate_team_play_stats(plays, team, _OFF_TYPES, _ST_TYPES).row(0, named=True)
        drive_stats = generate_team_drive_stats(drives, team).row(0, named=True)
        to_stats = generate_team_turnover_stats(plays, team, opponent).row(0, named=True)
        st_stats = generate_team_st_stats(plays, team, ep_data, punt_sr).row(0, named=True)

        row = {
            "Team": team,
            **{c: play_stats[c] for c in ["OffSR", "OffER", "AvgEqPPP", "IsoPPP", "Plays"]},
            **{c: drive_stats[c] for c in ["OppRate", "OppEff", "OppPPD", "OppSR"]},
            **{c: to_stats[c] for c in ["ExpTO", "ActualTO", "HavocRate", "SackRate"]},
            **{c: st_stats[c] for c in [
                "KickoffSR", "KickoffEqPPP", "KickoffReturnEqPPP",
                "PuntSR", "PuntEqPPP", "PuntReturnEqPPP",
            ]},
        }
        rows.append(row)

    box = pl.DataFrame(rows)

    # Compute per-factor diffs (team - opponent). For the 2-row frame this is the
    # antisymmetric reversal: [a, b] -> [a-b, b-a] (== pandas col - col.iloc[::-1]).
    box = box.with_columns([
        (pl.col(stat) - pl.col(stat).reverse()).alias(f"{stat}Diff")
        for stat in ["OffSR", "AvgEqPPP", "OppPPD", "OppRate", "OppSR",
                     "ActualTO", "SackRate", "HavocRate"]
    ])

    # Attach global EqPPP bounds for explosiveness domain
    box = box.with_columns([
        pl.lit(eq_ppp_global_min).alias("_eq_ppp_min"),
        pl.lit(eq_ppp_global_max).alias("_eq_ppp_max"),
    ])

    # 5FR composite (row-wise; mirrors pandas box.apply(..., axis=1))
    five_fr = [calculate_five_factors_rating(r) for r in box.iter_rows(named=True)]
    box = box.with_columns(pl.Series("5FR", five_fr, dtype=pl.Float64))

    # 5FRDiff is antisymmetric (A's diff = -B's diff)
    box = box.with_columns((pl.col("5FR") - pl.col("5FR").reverse()).alias("5FRDiff"))

    return box
