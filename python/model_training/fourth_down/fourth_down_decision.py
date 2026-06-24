"""Fourth-down decision layer: expected win probability of GOING FOR IT.

Python port of ``cfb4th/R/decision_functions.R::get_go_wp()`` against THIS repo's
retrained EP / WP / fourth-down models (Track 1 + Track 2 artifacts).

Algorithm (faithful to the R reference):
  1. ``fd_model`` predicts a 76-class yards-gained distribution per 4th-down play
     (objective ``multi:softprob``; class k -> gain = k - 10, range -10..65).
  2. Expand to a long (play x gain) frame.
  3. Cap gain at ``yards_to_goal`` (a longer gain is a TD); floor an impossible loss
     so the ball can't go past the offense's own 1 (``yards_to_goal - gain >= 100``).
  4. Update the hypothetical post-play game state per outcome:
       * turnover on downs (gain < distance, no TD): possession flips -> mirror
         ``yards_to_goal`` (100 - ytg), swap pos/def timeouts, flip
         ``pos_team_receives_2H_kickoff`` (1st half only), negate ``pos_team_spread``
         and ``pos_score_diff_start``, flip ``is_home``.
       * touchdown (ytg hits 0): give +6 to the scoring team then hand the ball to the
         other team receiving a kickoff at the 25 (``yards_to_goal = 75``); the same
         possession-flip bookkeeping as a turnover (the new posteam is the OTHER team).
       * 6-second runoff on ``TimeSecsRem`` / ``adj_TimeSecsRem`` (floored at 0).
       * ``distance`` becomes min(10, new ytg) (goal-to-go shrink).
  5. Compute EP then WP of each resulting state with THIS repo's models.
  6. ``go_wp = sum_k P(gain=k) * WP(state_k)``; also return ``first_down_prob``
     (P(gain >= distance | not TD-handling) == P(turnover == 0)), ``wp_succeed``
     (prob-weighted WP over conversion outcomes) and ``wp_fail`` (over failures).

Feature-contract mapping (cfb4th name -> this repo's pbp_full column):
  down                          start.down
  distance                      start.distance
  yards_to_goal                 start.yardsToEndzone
  pos_team_spread               start.pos_team_spread        (already posteam perspective)
  pos_team_total                derived: (homeTeamSpread+overUnder)/2 if is_home else
                                         (overUnder-homeTeamSpread)/2   (see fd features.py)
  era                           derived from season via FD_ERA_BOUNDS
  pos_score_diff_start          pos_score_diff_start
  TimeSecsRem                   start.TimeSecsRem
  adj_TimeSecsRem               start.adj_TimeSecsRem
  pos_team_receives_2H_kickoff  start.pos_team_receives_2H_kickoff
  pos_team_timeouts_rem_before  start.posTeamTimeouts
  def_pos_team_timeouts_rem_before  start.defPosTeamTimeouts
  is_home                       start.is_home
  period                        period

Approximations vs. the R reference (documented, not silently fudged):
  * Possession identity: cfb4th tracks pos_team/def_pos_team/home/away by NAME and
    looks up away/home timeouts via case_when. In pbp_full those team columns are
    IDs, but the per-row ``start.posTeamTimeouts`` / ``start.defPosTeamTimeouts``
    already carry the posteam/defteam-perspective timeout counts, so a possession
    flip is exactly a SWAP of the two (the net effect of the R case_when), and
    ``is_home`` is flipped directly. This is behaviorally identical and avoids the
    name/id mismatch.
  * EP: the R model is a 7-class nnet whose probs are dotted with a hand-written
    point table (TD == +-6.95). THIS repo's ep.ubj is a 7-class xgboost softprob
    whose class->score table is model_vars.ep_class_to_score_mapping
    ({0:+7,1:-7,2:+3,3:-3,4:+2,5:-2,6:0}); EP = sum(prob_k * score_k). We use the
    shipped table, not cfb4th's, so EP is consistent with this repo's WP model.
  * TD bonus: the R code adds +6 to the score differential on a TD (no PAT modeled);
    we keep the +6 to stay faithful to get_go_wp() (the 2-pt / PAT branch is a
    separate, unported get_2pt_wp path).
  * end_game_fn kneel-out clamps are ported (the win-when-leading / lose-when-failed
    late-game timeout cascades) exactly as in the R reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .. import constants as MC
from .constants import FD_ERA_BOUNDS, FD_FEATURES, FD_NUM_CLASS

# cfb4th name -> pbp_full column. Every state field the port needs is read through here.
_PBP_COLS = {
    "down": "start.down",
    "distance": "start.distance",
    "yards_to_goal": "start.yardsToEndzone",
    "pos_team_spread": "start.pos_team_spread",
    "pos_score_diff_start": "pos_score_diff_start",
    "TimeSecsRem": "start.TimeSecsRem",
    "adj_TimeSecsRem": "start.adj_TimeSecsRem",
    "pos_team_receives_2H_kickoff": "start.pos_team_receives_2H_kickoff",
    "pos_team_timeouts_rem_before": "start.posTeamTimeouts",
    "def_pos_team_timeouts_rem_before": "start.defPosTeamTimeouts",
    "is_home": "start.is_home",
    "period": "period",
    "season": "season",
    "overUnder": "overUnder",
    "homeTeamSpread": "homeTeamSpread",
}

# EP softprob class -> point value, from the shipped contract.
_EP_SCORES = np.array(
    [MC.EP_CLASS_TO_SCORE[i] for i in range(len(MC.EP_CLASS_TO_SCORE))],
    dtype=np.float64,
)


def _to_pandas(df):
    """Accept polars or pandas; return a pandas copy with the cfb4th-named state cols."""
    if hasattr(df, "to_pandas"):  # polars.DataFrame
        df = df.to_pandas()
    out = pd.DataFrame(index=range(len(df)))
    for short, src in _PBP_COLS.items():
        if src in df.columns:
            out[short] = df[src].to_numpy()
        else:
            out[short] = np.nan
    return out


def _posteam_total(state: pd.DataFrame) -> np.ndarray:
    """(homeTeamSpread + overUnder)/2 if posteam is home else (overUnder - homeTeamSpread)/2.

    Mirrors fourth_down/features.py posteam_total derivation.
    """
    is_home = state["is_home"].to_numpy().astype(bool)
    ou = state["overUnder"].to_numpy().astype(float)
    hs = state["homeTeamSpread"].to_numpy().astype(float)
    return np.where(is_home, (hs + ou) / 2.0, (ou - hs) / 2.0)


def _era(season: np.ndarray) -> np.ndarray:
    lo, mid, hi = FD_ERA_BOUNDS
    out = np.full(len(season), 3, dtype=np.int32)
    out = np.where(season <= hi, 2, out)
    out = np.where(season <= mid, 1, out)
    out = np.where(season <= lo, 0, out)
    return out


def _predict_ep(state: pd.DataFrame, ep_model: xgb.Booster) -> np.ndarray:
    """EP for each state row: prep_ep + add_ep, using this repo's softprob EP model."""
    down = state["down"].to_numpy().astype(int)
    X = pd.DataFrame(
        {
            "TimeSecsRem": state["TimeSecsRem"].to_numpy().astype(float),
            "yards_to_goal": state["yards_to_goal"].to_numpy().astype(float),
            "distance": state["distance"].to_numpy().astype(float),
            "down_1": (down == 1).astype(int),
            "down_2": (down == 2).astype(int),
            "down_3": (down == 3).astype(int),
            "down_4": (down == 4).astype(int),
            "pos_score_diff_start": state["pos_score_diff_start"].to_numpy().astype(float),
        }
    )[MC.EP_FEATURES]
    probs = ep_model.predict(xgb.DMatrix(X))
    if probs.ndim == 1:
        probs = probs.reshape(-1, len(_EP_SCORES))
    return probs @ _EP_SCORES


def _predict_wp(state: pd.DataFrame, ep: np.ndarray, wp_model: xgb.Booster) -> np.ndarray:
    """WP for each state row: prep_wp (ExpScoreDiff/spread_time/...) + add_wp."""
    adj = state["adj_TimeSecsRem"].to_numpy().astype(float)
    pos_diff = state["pos_score_diff_start"].to_numpy().astype(float)
    exp_score_diff = pos_diff + ep
    exp_ratio = exp_score_diff / (adj + 1.0)
    elapsed_share = (3600.0 - adj) / 3600.0
    # spread_time MUST match the trained-on convention: pbp_full's start.spread_time is
    # +pos_team_spread * exp(-4 * elapsed_share) (verified MAE 0.0 against the training
    # frame). The previous `-1.0 *` form fed the WP model a sign-inverted spread.
    spread_time = state["pos_team_spread"].to_numpy().astype(float) * np.exp(
        -4.0 * elapsed_share
    )
    X = pd.DataFrame(
        {
            "pos_team_receives_2H_kickoff": state["pos_team_receives_2H_kickoff"]
            .to_numpy()
            .astype(float),
            "spread_time": spread_time,
            "TimeSecsRem": state["TimeSecsRem"].to_numpy().astype(float),
            "adj_TimeSecsRem": adj,
            "ExpScoreDiff_Time_Ratio": exp_ratio,
            "pos_score_diff_start": pos_diff,
            "down": state["down"].to_numpy().astype(float),
            "distance": state["distance"].to_numpy().astype(float),
            "yards_to_goal": state["yards_to_goal"].to_numpy().astype(float),
            "is_home": state["is_home"].to_numpy().astype(float),
            "pos_team_timeouts_rem_before": state["pos_team_timeouts_rem_before"]
            .to_numpy()
            .astype(float),
            "def_pos_team_timeouts_rem_before": state["def_pos_team_timeouts_rem_before"]
            .to_numpy()
            .astype(float),
            "period": state["period"].to_numpy().astype(float),
        }
    )[MC.WP_SPREAD_FEATURES]
    return wp_model.predict(xgb.DMatrix(X))


def get_go_wp_py(pbp_df, fd_model, ep_model, wp_model):
    """Compute the expected win probability of going for it on 4th down.

    Args:
        pbp_df: Play-by-play DataFrame (polars or pandas) of fourth-down situations.
            Must carry the ``start.*`` state columns listed in this module's docstring.
        fd_model: Trained fourth-down yards-gained Booster (6-feat, 76-class softprob).
        ep_model: Trained EP Booster (8-feat multi:softprob, 7-class).
        wp_model: Trained WP-spread Booster (13-feat binary:logistic).

    Returns:
        A pandas DataFrame copy of ``pbp_df`` augmented with four columns:
        ``go_wp`` (prob-weighted WP of going for it), ``first_down_prob``
        (P(conversion)), ``wp_succeed`` (mean WP over conversion outcomes) and
        ``wp_fail`` (mean WP over failure outcomes). ``go_wp`` is always defined
        and in [0, 1]; the conditional columns are in [0, 1] but can be NaN for
        degenerate goal-line plays where one outcome bucket is empty (e.g.
        4th-and-goal at the 1, where every modeled gain is a touchdown) -- this
        matches the R reference's ``pivot_wider`` NA behavior.

    Raises:
        ValueError: if ``pbp_df`` is missing the fourth-down state columns.
    """
    n_plays = len(pbp_df)
    base = pd.DataFrame(pbp_df.to_pandas() if hasattr(pbp_df, "to_pandas") else pbp_df).reset_index(
        drop=True
    )
    if n_plays == 0:
        out = base.copy()
        for c in ("go_wp", "first_down_prob", "wp_succeed", "wp_fail"):
            out[c] = pd.Series([], dtype=float)
        return out

    st = _to_pandas(pbp_df)

    # --- step 1: fd_model 76-class yards-gained distribution per play ---
    fd_X = pd.DataFrame(
        {
            "down": st["down"].to_numpy().astype(float),
            "distance": st["distance"].to_numpy().astype(float),
            "yards_to_goal": st["yards_to_goal"].to_numpy().astype(float),
            "posteam_total": _posteam_total(st),
            "posteam_spread": st["pos_team_spread"].to_numpy().astype(float),
            "era": _era(st["season"].to_numpy().astype(float)),
        }
    )[FD_FEATURES]
    fd_probs = fd_model.predict(xgb.DMatrix(fd_X))
    if fd_probs.ndim == 1:
        fd_probs = fd_probs.reshape(n_plays, FD_NUM_CLASS)

    # --- step 2: expand to long (play x gain) ---
    gains = np.arange(FD_NUM_CLASS) - 10  # -10..65
    play_idx = np.repeat(np.arange(n_plays), FD_NUM_CLASS)
    gain = np.tile(gains, n_plays).astype(np.int64)
    prob = fd_probs.reshape(-1).astype(np.float64)

    ytg0 = st["yards_to_goal"].to_numpy()[play_idx].astype(np.int64)
    dist0 = st["distance"].to_numpy()[play_idx].astype(np.int64)

    # --- step 3: cap at TD, floor impossible loss (ball on the 1) ---
    gain = np.where(gain > ytg0, ytg0, gain)
    gain = np.where(ytg0 - gain >= 100, ytg0 - 99, gain)

    # collapse duplicate (play, gain) rows produced by the cap (all the TD classes) ---
    long = pd.DataFrame({"play_idx": play_idx, "gain": gain, "prob": prob})
    long = long.groupby(["play_idx", "gain"], as_index=False)["prob"].sum()
    play_idx = long["play_idx"].to_numpy()
    gain = long["gain"].to_numpy()
    prob = long["prob"].to_numpy()

    # broadcast the per-play base state to every (play, gain) row ---
    state = st.iloc[play_idx].reset_index(drop=True).copy()

    # --- step 4: update game situation per outcome ---
    ytg = state["yards_to_goal"].to_numpy().astype(np.int64) - gain
    turnover = ((gain < state["distance"].to_numpy().astype(np.int64))).astype(int)
    state["down"] = 1

    # turnover on downs: flip field + possession bookkeeping (TDs handled next) ---
    to_mask = turnover == 1
    ytg = np.where(to_mask, 100 - ytg, ytg)

    pos_to = state["pos_team_timeouts_rem_before"].to_numpy().astype(float)
    def_to = state["def_pos_team_timeouts_rem_before"].to_numpy().astype(float)
    # possession flip == swap posteam/defteam timeouts (net effect of the R case_when)
    new_pos_to = np.where(to_mask, def_to, pos_to)
    new_def_to = np.where(to_mask, pos_to, def_to)

    period = state["period"].to_numpy().astype(float)
    recv = state["pos_team_receives_2H_kickoff"].to_numpy().astype(float)
    recv = np.where((period <= 2) & (recv == 0) & to_mask, 1.0, recv)
    recv = np.where((period <= 2) & (recv == 1) & to_mask, 0.0, recv)

    is_home = state["is_home"].to_numpy().astype(float)
    is_home = np.where(to_mask, 1.0 - is_home, is_home)

    spread = state["pos_team_spread"].to_numpy().astype(float)
    spread = np.where(to_mask, -spread, spread)
    pos_diff = state["pos_score_diff_start"].to_numpy().astype(float)
    pos_diff = np.where(to_mask, -pos_diff, pos_diff)

    # touchdown: ytg hit 0 (after the TD cap). Score the offense (+6), then the OTHER
    # team receives a kickoff at the 25 (ytg = 75) -- same flip bookkeeping again.
    td_mask = ytg == 0
    pos_diff = np.where(td_mask, -pos_diff - 6.0, pos_diff)
    ytg = np.where(td_mask, 75, ytg)
    # on a TD the possessing team changes -> swap timeouts (from the post-turnover values)
    td_pos_to = np.where(td_mask, new_def_to, new_pos_to)
    td_def_to = np.where(td_mask, new_pos_to, new_def_to)
    new_pos_to, new_def_to = td_pos_to, td_def_to
    recv = np.where((period <= 2) & (recv == 0) & td_mask, 1.0, recv)
    recv = np.where((period <= 2) & (recv == 1) & td_mask, 0.0, recv)
    is_home = np.where(td_mask, 1.0 - is_home, is_home)
    spread = np.where(td_mask, -spread, spread)

    # 6-second runoff, floored at 0
    tsr = np.maximum(state["TimeSecsRem"].to_numpy().astype(float) - 6.0, 0.0)
    adj = np.maximum(state["adj_TimeSecsRem"].to_numpy().astype(float) - 6.0, 0.0)

    # goal-to-go distance shrink
    distance = np.where(ytg < 10, ytg, 10)

    state["yards_to_goal"] = ytg
    state["distance"] = distance
    state["pos_team_timeouts_rem_before"] = new_pos_to
    state["def_pos_team_timeouts_rem_before"] = new_def_to
    state["pos_team_receives_2H_kickoff"] = recv
    state["is_home"] = is_home
    state["pos_team_spread"] = spread
    state["pos_score_diff_start"] = pos_diff
    state["TimeSecsRem"] = tsr
    state["adj_TimeSecsRem"] = adj

    # --- step 5: EP then WP of each resulting state ---
    ep = _predict_ep(state, ep_model)
    wp = _predict_wp(state, ep, wp_model)

    # flip WP for possession change (turnover or TD both hand the ball over).
    # is_home was flipped exactly on those rows, so flip == (is_home != original).
    orig_is_home = st["is_home"].to_numpy().astype(float)[play_idx]
    flipped = is_home != orig_is_home
    wp = np.where(flipped, 1.0 - wp, wp)

    # --- end_game_fn kneel-out clamps (ported from get_go_wp) ---
    # success (no turnover, no TD, ytg>0) and leading -> win
    succ_alive = (turnover == 0) & (~td_mask) & (ytg > 0) & (pos_diff > 0)
    wp = np.where(succ_alive & (adj < 120) & (new_def_to == 0), 1.0, wp)
    wp = np.where(succ_alive & (adj < 80) & (new_def_to == 1), 1.0, wp)
    wp = np.where(succ_alive & (adj < 40) & (new_def_to == 2), 1.0, wp)
    # failure (turnover) and was leading -> loss (other team kneels out)
    fail_lead = (turnover == 1) & (pos_diff < 0)  # pos_diff already negated on turnover
    # NOTE: after the turnover negation, "we were leading" == new pos_diff < 0.
    wp = np.where(fail_lead & (adj < 120) & (new_def_to == 0), 0.0, wp)
    wp = np.where(fail_lead & (adj < 80) & (new_def_to == 1), 0.0, wp)
    wp = np.where(fail_lead & (adj < 40) & (new_def_to == 2), 0.0, wp)

    # --- step 6: aggregate ---
    res = pd.DataFrame(
        {"play_idx": play_idx, "turnover": turnover, "prob": prob, "wp": wp}
    )
    res["wt_wp"] = res["prob"] * res["wp"]

    go = res.groupby("play_idx").agg(go_wp=("wt_wp", "sum")).reset_index()

    # per (play, turnover) renormalized conditional WP -> wp_succeed / wp_fail
    grp = res.groupby(["play_idx", "turnover"])
    cond = grp.apply(
        lambda g: pd.Series(
            {"pct": g["prob"].sum(), "wp": (g["prob"] * g["wp"]).sum() / g["prob"].sum()}
        ),
        include_groups=False,
    ).reset_index()
    piv = cond.pivot(index="play_idx", columns="turnover")
    pct0 = piv["pct"].get(0)
    wp0 = piv["wp"].get(0)
    wp1 = piv["wp"].get(1)
    report = pd.DataFrame({"play_idx": piv.index})
    report["first_down_prob"] = (pct0 if pct0 is not None else 0.0).to_numpy()
    report["wp_succeed"] = (wp0 if wp0 is not None else np.nan).to_numpy()
    report["wp_fail"] = (wp1 if wp1 is not None else np.nan).to_numpy()

    merged = (
        pd.DataFrame({"play_idx": np.arange(n_plays)})
        .merge(go, on="play_idx", how="left")
        .merge(report, on="play_idx", how="left")
    )

    out = base.copy()
    out["go_wp"] = merged["go_wp"].to_numpy()
    out["first_down_prob"] = merged["first_down_prob"].to_numpy()
    out["wp_succeed"] = merged["wp_succeed"].to_numpy()
    out["wp_fail"] = merged["wp_fail"].to_numpy()
    return out
