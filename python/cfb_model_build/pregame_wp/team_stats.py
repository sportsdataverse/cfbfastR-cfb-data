"""Per-team per-game statistics for the Five-Factors pipeline.

Faithful port of win-prob.ipynb cells 22 and 24.

OQ-5 note: generate_team_st_stats assigns punt_eqppp (not punt_ret_eqppp) to
PuntReturnEqPPP, matching the notebook bug. PuntEqPPP - PuntReturnEqPPP = 0 always.
"""
from __future__ import annotations

import polars as pl

from .constants import EXP_TO_FUM_WEIGHT, EXP_TO_INT_WEIGHT, SCORING_OPP_THRESHOLD


def _col_or_zero(df: pl.DataFrame, col: str) -> pl.Series:
    """Return ``df[col]`` (float) or an all-zero Series when the column is absent.

    Mirrors the pandas ``row.get(col, 0)`` default used in the original
    row-wise ``.apply()`` lambdas for optional CFBD fields (``kick_yards``,
    ``return_yards``) that aren't part of the normalized required-column set.
    """
    if col in df.columns:
        return df[col].cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0)
    return pl.Series(col, [0.0] * df.height, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Play-level stats
# ---------------------------------------------------------------------------

def generate_team_play_stats(
    df: pl.DataFrame,
    team: str,
    off_types: list[str],
    st_types: list[str],
) -> pl.DataFrame:
    """OffSR, OffER, AvgEqPPP, IsoPPP for one team in one game."""
    off = df.filter(
        (pl.col("offense") == team) & (pl.col("play_type").is_in(off_types))
    )
    if off.height == 0:
        return pl.DataFrame([{"Team": team, "OffSR": 0.0, "OffER": 0.0,
                              "AvgEqPPP": 0.0, "IsoPPP": 0.0, "Plays": 0}])
    n = off.height
    off_sr = off["play_successful"].mean()
    off_er = off["play_explosive"].mean()
    avg_eqppp = off["EqPPP"].mean()
    successful = off.filter(pl.col("play_successful") == True)  # noqa: E712
    iso_ppp = successful["EqPPP"].mean() if successful.height > 0 else 0.0
    return pl.DataFrame([{
        "Team": team,
        "OffSR": float(off_sr),
        "OffER": float(off_er),
        "AvgEqPPP": float(avg_eqppp),
        "IsoPPP": float(iso_ppp),
        "Plays": n,
    }])


# ---------------------------------------------------------------------------
# Drive-level stats
# ---------------------------------------------------------------------------

def generate_team_drive_stats(
    df: pl.DataFrame,
    team: str,
) -> pl.DataFrame:
    """OppRate, OppEff, OppPPD for one team in one game."""
    drives = df.filter(pl.col("offense") == team)
    if drives.height == 0:
        return pl.DataFrame([{"Team": team, "OppRate": 0.0, "OppEff": 0.0,
                              "OppPPD": 0.0, "OppSR": 0.0}])
    n_drives = drives.height
    scoring_opps = drives.filter(
        pl.col("drive_start_yardline") + pl.col("drive_yards") >= SCORING_OPP_THRESHOLD
    )
    n_opps = scoring_opps.height
    opp_rate = n_opps / n_drives if n_drives > 0 else 0.0
    if n_opps == 0:
        return pl.DataFrame([{"Team": team, "OppRate": opp_rate, "OppEff": 0.0,
                              "OppPPD": 0.0, "OppSR": 0.0}])
    # drive_scoring is Boolean; .mean()/.sum() treat True=1 (matches pandas bool math)
    opp_eff = scoring_opps["drive_scoring"].mean()
    opp_ppd = scoring_opps["drive_pts"].mean()
    opp_sr = scoring_opps["drive_scoring"].sum() / n_drives
    return pl.DataFrame([{
        "Team": team,
        "OppRate": float(opp_rate),
        "OppEff": float(opp_eff),
        "OppPPD": float(opp_ppd),
        "OppSR": float(opp_sr),
    }])


# ---------------------------------------------------------------------------
# Turnover stats
# ---------------------------------------------------------------------------

def generate_team_turnover_stats(
    df: pl.DataFrame,
    offense: str,
    defense: str,
) -> pl.DataFrame:
    """ExpTO, ActualTO, HavocRate, SackRate for one team in one game (offense perspective)."""
    off = df.filter(pl.col("offense") == offense)
    if off.height == 0:
        return pl.DataFrame([{"Team": offense, "ExpTO": 0.0, "ActualTO": 0,
                              "HavocRate": 0.0, "SackRate": 0.0}])

    n_plays = off.height

    # Pass deflections: incomplete passes with "broken up" in play_text
    # (?i) = case-insensitive; null text never matches (mirrors pandas na=False)
    n_pd = off.filter(
        (pl.col("play_type") == "Pass Incompletion")
        & pl.col("play_text").str.contains("(?i)broken up").fill_null(False)
    ).height

    # Interceptions
    n_int = off.filter(pl.col("play_type") == "Interception").height

    # Fumbles recovered by opponent
    n_fum = off.filter(pl.col("play_type") == "Fumble Recovery (Opponent)").height

    exp_to = EXP_TO_INT_WEIGHT * (n_pd + n_int) + EXP_TO_FUM_WEIGHT * n_fum
    actual_to = n_int + n_fum

    # Havoc: interceptions + fumbles recovered by defense + sacks
    n_sack = off.filter(pl.col("play_type") == "Sack").height
    havoc = n_int + n_fum + n_sack
    havoc_rate = havoc / n_plays if n_plays > 0 else 0.0
    sack_rate = n_sack / n_plays if n_plays > 0 else 0.0

    return pl.DataFrame([{
        "Team": offense,
        "ExpTO": float(exp_to),
        "ActualTO": actual_to,
        "HavocRate": float(havoc_rate),
        "SackRate": float(sack_rate),
    }])


# ---------------------------------------------------------------------------
# Special teams stats
# ---------------------------------------------------------------------------

def generate_team_st_stats(
    df: pl.DataFrame,
    team: str,
    ep_data: list[float],
    punt_sr: dict[int, float],
) -> pl.DataFrame:
    """Kickoff/punt ST stats including EqPPP values.

    OQ-5 faithful port: PuntReturnEqPPP = PuntEqPPP (punt_eqppp), NOT punt_ret_eqppp.
    This means PuntEqPPP - PuntReturnEqPPP = 0 always, matching the notebook.
    """
    kicks = df.filter((pl.col("offense") == team) & (pl.col("play_type") == "Kickoff"))
    punts = df.filter((pl.col("offense") == team) & (pl.col("play_type") == "Punt"))

    last = len(ep_data) - 1  # EP curve indexed by clamped yardline [0, 100]

    def _ep_at(yardlines: list[int]) -> list[float]:
        return [ep_data[max(0, min(last, int(y)))] for y in yardlines]

    # --- kickoff ---
    if kicks.height == 0:
        kick_sr, kick_eqppp, kick_ret_eqppp = 0.0, 0.0, 0.0
    else:
        yl = kicks["yard_line"].to_list()
        ky = _col_or_zero(kicks, "kick_yards").to_list()
        ry = _col_or_zero(kicks, "return_yards").to_list()
        kick_sr = (sum(ky) / len(ky) / 100.0) if "kick_yards" in kicks.columns else 0.0
        land = _ep_at([y + k for y, k in zip(yl, ky)])
        start = _ep_at(yl)
        kick_eqppp = sum(a - b for a, b in zip(land, start)) / len(yl)
        kick_ret_eqppp = sum(_ep_at(ry)) / len(ry)

    # --- punt ---
    if punts.height == 0:
        punt_sr_val, punt_eqppp, punt_ret_eqppp = 0.0, 0.0, 0.0
    else:
        yl = punts["yard_line"].to_list()
        ky = _col_or_zero(punts, "kick_yards").to_list()
        ry = _col_or_zero(punts, "return_yards").to_list()
        punt_sr_val = sum(
            float(k > punt_sr.get(int(y), 40.0)) for y, k in zip(yl, ky)
        ) / len(yl)
        land = _ep_at([y + k for y, k in zip(yl, ky)])
        start = _ep_at(yl)
        punt_eqppp = sum(a - b for a, b in zip(land, start)) / len(yl)
        punt_ret_eqppp = sum(_ep_at(ry)) / len(ry)

    return pl.DataFrame([{
        "Team": team,
        "KickoffSR": float(kick_sr),
        "KickoffEqPPP": float(kick_eqppp),
        "KickoffReturnEqPPP": float(kick_ret_eqppp),
        "PuntSR": float(punt_sr_val),
        "PuntEqPPP": float(punt_eqppp),
        # OQ-5: faithful port uses punt_eqppp (not punt_ret_eqppp) for PuntReturnEqPPP
        "PuntReturnEqPPP": float(punt_eqppp),
        "PuntReturnIsoPPP": float(punt_ret_eqppp),
    }])
