"""Win probability inference: XGBRegressor prediction → normal-CDF WP.

Ports win-prob.ipynb cells 45, 53, 58, 59 (single-game WP + season-strength
tables + ``predict_matchup`` future-game projection).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .constants import HFA_COVID, HFA_NORMAL, PRESEASON_WEEKS, TALENT_FCS_PERCENTILE


def five_fr_to_wp(predicted_mov: float, mu: float, std: float) -> float:
    """Convert a predicted margin-of-victory to a win probability.

    Args:
        predicted_mov: Model output (5FRDiff → predicted PtsDiff).
        mu: Mean of the training prediction distribution (0.0 per OQ-7).
        std: Std dev of training predictions (full-set, per OQ-7).

    Returns:
        Win probability in (0, 1).
    """
    if std <= 0:
        return 0.5
    return float(norm.cdf((predicted_mov - mu) / std))


def generate_win_prob(
    fr_diff: float,
    model,
    mu: float,
    std: float,
    hfa: float = 0.0,
) -> float:
    """Generate WP from a 5FRDiff value + trained model + normalization params.

    Args:
        fr_diff: Pre-computed 5FRDiff (home - away).
        model: Fitted XGBRegressor.
        mu: Normalization mu (0.0 per OQ-7).
        std: Normalization std.
        hfa: Home-field advantage adjustment in points (added to predicted MOV).

    Returns:
        Win probability for the home team in (0, 1).
    """
    X = np.array([[fr_diff]])
    predicted_mov = float(model.predict(X)[0]) + hfa
    return five_fr_to_wp(predicted_mov, mu=mu, std=std)


# ---------------------------------------------------------------------------
# Season-strength tables (notebook cells 53 + 58)
# ---------------------------------------------------------------------------

def build_season_strength_tables(
    stored_game_boxes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive ``team_strength`` + ``opponent_game_ids`` from the box corpus.

    Ports notebook cells 53 (``team_strength`` = per-(Team, Season) mean 5FR)
    and 58 (``opponent_game_ids`` = the per-team, week-ordered opponent log).

    Args:
        stored_game_boxes: Output of ``training.build_training_frame`` — must
            carry ``Team``, ``Season``, ``GameID``, ``5FR`` and (optionally)
            ``Week`` columns.

    Returns:
        ``(team_strength, opponent_game_ids)`` where

        * ``team_strength`` has columns ``Team``, ``Season``, ``5FR``
          (season-average 5FR per team).
        * ``opponent_game_ids`` has columns ``Team``, ``Opponent``, ``Season``,
          ``Week``, ``GameID`` — a 1-based per-team game sequence within a
          season (week order preserved when a ``Week`` column is present).
    """
    box = stored_game_boxes.copy()

    team_strength = (
        box.groupby(["Team", "Season"])["5FR"].mean().reset_index()
    )

    # opponent_game_ids: for each (team, season), order its games and record the
    # opponent + a 1-based per-team week index.
    sort_cols = ["Season", "Team"]
    if "Week" in box.columns:
        sort_cols.append("Week")
    box_sorted = box.sort_values(sort_cols)

    rows: list[dict[str, Any]] = []
    # Map GameID -> the two teams so we can resolve each row's opponent.
    teams_by_game = box.groupby("GameID")["Team"].apply(list).to_dict()
    for (team, season), grp in box_sorted.groupby(["Team", "Season"]):
        for seq, (_, r) in enumerate(grp.iterrows(), start=1):
            members = teams_by_game.get(r["GameID"], [])
            opp = next((t for t in members if t != team), None)
            rows.append({
                "GameID": r["GameID"], "Team": team, "Opponent": opp,
                "Season": season,
                "Week": int(r["Week"]) if ("Week" in box.columns and pd.notna(r.get("Week"))) else seq,
            })
    opponent_game_ids = pd.DataFrame(
        rows, columns=["GameID", "Team", "Opponent", "Season", "Week"]
    )
    return team_strength, opponent_game_ids


def _team_avg_ffr(
    box: pd.DataFrame,
    team: str,
    season: int,
    considered_weeks: int,
    games_to_consider: int,
    fallback: float,
) -> float:
    """Mean of a team's last ``games_to_consider`` game-level 5FR up to a week.

    Faithful to cell 59:
    ``grouped_by_year.get_group((team, season))[:considered_weeks]['5FR']
    .tail(games_to_consider).mean()`` with a national-average fallback when the
    team has no games that season.
    """
    grp = box[(box["Team"] == team) & (box["Season"] == season)]
    if grp.empty:
        return fallback
    if "Week" in grp.columns:
        grp = grp.sort_values("Week")
    series = grp["5FR"]
    series = series.iloc[:considered_weeks].tail(games_to_consider)
    val = series.mean()
    return float(val) if pd.notna(val) else fallback


def _conf_of(conferences: Optional[dict[str, str]], team: str) -> Optional[str]:
    if not conferences:
        return None
    return conferences.get(team)


def predict_matchup(
    team1: str,
    team2: str,
    year: int,
    week: int = -1,
    games_to_consider: int = 4,
    *,
    stored_game_boxes: pd.DataFrame,
    model: Any,
    mu: float,
    std: float,
    team_strength: Optional[pd.DataFrame] = None,
    opponent_game_ids: Optional[pd.DataFrame] = None,
    conferences: Optional[dict[str, str]] = None,
    p5_conferences: Optional[list[str]] = None,
    g5_conferences: Optional[list[str]] = None,
    roster_talent: Optional[Callable[[str, int], float]] = None,
    returning_production: Optional[Callable[[str, int], float]] = None,
    adjust_hfa: bool = False,
    adjust_covid: bool = False,
) -> list[float]:
    """Project a future-matchup home win probability (faithful port of cell 59).

    ``team1`` is the home team. The projection takes each team's last
    ``games_to_consider`` games' average 5FR (prior season if ``week == 0``),
    layers the three strength-of-schedule adjustments (opponent-5FR ratio, P5/G5
    subdivision, conference — the latter two skipped without ``conferences`` /
    skipped for FBS Independents), applies the returning-production × roster-
    talent adjustment for weeks 1–4, converts the resulting ``5FRDiff`` through
    the model, adds HFA, and returns ``[win_prob, proj_MOV]``.

    Args:
        team1: Home team.
        team2: Away team.
        year: Season to project for.
        week: Week of season (``<=0`` = consider all weeks; ``0`` also rolls the
            *applied* season back one year for a preseason / prior-year basis).
        games_to_consider: Recent games to average per team (default 4).
        stored_game_boxes: Box corpus (from ``build_training_frame``).
        model: Fitted XGBRegressor (5FRDiff → PtsDiff).
        mu: WP-normalization mean (0.0 per OQ-7).
        std: WP-normalization std (full-training-set preds, per OQ-7).
        team_strength: Pre-built season-strength table; derived from
            ``stored_game_boxes`` if omitted.
        opponent_game_ids: Pre-built opponent log; derived if omitted.
        conferences: Optional ``{team: conference}`` map enabling the P5/G5 +
            conference SoS adjustments. When absent, those two layers are
            skipped (the opponent-5FR SoS adjustment still applies).
        p5_conferences: Power-5 conference names (defaults to the notebook set).
        roster_talent: Optional ``(team, year) -> talent`` callable for the
            weeks 1–4 returning-production adjustment.
        returning_production: Optional ``(team, year) -> production`` callable.
        adjust_hfa: Add home-field advantage to the projected MOV.
        adjust_covid: Use the reduced COVID-2020 HFA (+1.0) instead of +2.5.

    Returns:
        ``[win_prob, proj_MOV]`` for the home team (``team1``).
    """
    box = stored_game_boxes
    if team_strength is None or opponent_game_ids is None:
        ts, og = build_season_strength_tables(box)
        team_strength = team_strength if team_strength is not None else ts
        opponent_game_ids = opponent_game_ids if opponent_game_ids is not None else og

    p5 = p5_conferences or [
        "SEC", "Pac-12", "FBS Independents", "Big 12", "ACC", "Big Ten",
    ]

    considered_weeks = week if week > 0 else 16
    applied_year = year - 1 if week == 0 else year
    if games_to_consider <= 0:
        games_to_consider = 16

    season_strength = team_strength[team_strength["Season"] == applied_year]
    natl_avg = float(season_strength["5FR"].mean()) if not season_strength.empty else 0.0
    fcs = (
        float(season_strength["5FR"].quantile(TALENT_FCS_PERCENTILE))
        if not season_strength.empty else natl_avg
    )

    def _mean_5fr(team_list: list[str]) -> float:
        sub = season_strength[season_strength["Team"].isin(team_list)]
        v = sub["5FR"].mean()
        return float(v) if pd.notna(v) else natl_avg

    def _opponents(team: str) -> list[str]:
        og = opponent_game_ids
        sub = og[
            (og["Team"] == team)
            & (og["Season"] == applied_year)
            & (og["Week"] < considered_weeks)
        ].iloc[:considered_weeks]
        return [o for o in sub["Opponent"].tolist() if o is not None]

    def _subdiv_members(is_p5: bool) -> list[str]:
        if not conferences:
            return []
        return [t for t, c in conferences.items() if (c in p5) == is_p5]

    def _conf_members(conf: Optional[str]) -> list[str]:
        if not conferences or conf is None:
            return []
        return [t for t, c in conferences.items() if c == conf]

    t1_conf = _conf_of(conferences, team1)
    t2_conf = _conf_of(conferences, team2)

    fbs = set(p5) | set(g5_conferences or [
        "Mountain West", "Mid-American", "Sun Belt",
        "Conference USA", "American Athletic",
    ])

    # FBS membership: known only when a conferences map is supplied. A team
    # whose conference is absent from the FBS set is treated as FCS and pinned
    # to the FCS-floor 5FR (cell 59's ``if ~conference.isin(fbs)`` branch).
    def _is_fbs(conf: Optional[str]) -> bool:
        if not conferences:
            return True  # no conference data → assume FBS (skip the FCS pin)
        return conf in fbs

    t1_fbs = _is_fbs(t1_conf)
    t2_fbs = _is_fbs(t2_conf)

    # --- per-team attributes (FCS teams pinned to the FCS floor) ---
    if t1_fbs:
        t1_ffr = _team_avg_ffr(box, team1, applied_year, considered_weeks, games_to_consider, natl_avg)
        t1_sos = _mean_5fr(_opponents(team1)) or natl_avg
        t1_conf_sos = _mean_5fr(_conf_members(t1_conf)) if conferences else None
        t1_subdiv_sos = _mean_5fr(_subdiv_members(t1_conf in p5)) if conferences else None
    else:
        t1_ffr = t1_sos = t1_conf_sos = t1_subdiv_sos = fcs

    if t2_fbs:
        t2_ffr = _team_avg_ffr(box, team2, applied_year, considered_weeks, games_to_consider, natl_avg)
        t2_sos = _mean_5fr(_opponents(team2)) or natl_avg
        t2_conf_sos = _mean_5fr(_conf_members(t2_conf)) if conferences else None
        t2_subdiv_sos = _mean_5fr(_subdiv_members(t2_conf in p5)) if conferences else None
    else:
        t2_ffr = t2_sos = t2_conf_sos = t2_subdiv_sos = fcs

    # --- SoS adjustment (opponent 5FR ratio); penalize the weaker-SoS team ---
    if t2_sos < t1_sos and t1_sos:
        t2_ffr *= (t2_sos / t1_sos)
    elif t2_sos > t1_sos and t2_sos:
        t1_ffr *= (t1_sos / t2_sos)

    # --- P5/G5 subdivision adjustment ---
    if t1_subdiv_sos is not None and t2_subdiv_sos is not None:
        if t2_subdiv_sos < t1_subdiv_sos and t1_subdiv_sos:
            t2_ffr *= (t2_subdiv_sos / t1_subdiv_sos)
        elif t2_subdiv_sos > t1_subdiv_sos and t2_subdiv_sos:
            t1_ffr *= (t1_subdiv_sos / t2_subdiv_sos)

    # --- conference SoS adjustment (skip FBS Independents) ---
    if (
        t1_conf_sos is not None and t2_conf_sos is not None
        and t1_conf != "FBS Independents" and t2_conf != "FBS Independents"
    ):
        if t2_conf_sos < t1_conf_sos and t1_conf_sos:
            t2_ffr *= (t2_conf_sos / t1_conf_sos)
        elif t2_conf_sos > t1_conf_sos and t2_conf_sos:
            t1_ffr *= (t1_conf_sos / t2_conf_sos)

    # --- returning-production × roster-talent adjustment (weeks 1–4) ---
    if 0 < week <= PRESEASON_WEEKS and roster_talent is not None and returning_production is not None:
        t1_talent = float(roster_talent(team1, year))
        t2_talent = float(roster_talent(team2, year))
        t1_ret = float(returning_production(team1, year)) * t1_talent
        t2_ret = float(returning_production(team2, year)) * t2_talent
        if t2_ret < t1_ret and t1_talent:
            t2_ffr *= (t2_talent / t1_talent)
        elif t2_ret > t1_ret and t2_talent:
            t1_ffr *= (t1_talent / t2_talent)

    ffr_diff = t1_ffr - t2_ffr  # team1 is home
    proj_mov = float(model.predict(np.array([[ffr_diff]]))[0])
    if adjust_hfa:
        proj_mov += (HFA_COVID if adjust_covid else HFA_NORMAL)

    win_prob = five_fr_to_wp(proj_mov, mu=mu, std=std)
    return [win_prob, proj_mov]
