"""Shared constants for the CFB CPOE pipeline (Track 5, Approach A).

Hyper-parameters mirror the nflfastR `cpoe_model.R` recipe:
    binary:logistic, eta=0.025, gamma=5, subsample=0.8,
    colsample_bytree=0.8, max_depth=4, min_child_weight=6, nrounds=560.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature / target columns
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "down",
    "distance",
    "yards_to_goal",
    "score_diff",
    "seconds_remaining",
    "is_home",
    "period",
    "passing_down",
]

TARGET_COL: str = "completion"

# ---------------------------------------------------------------------------
# Throw-depth proxy buckets (yards-to-first-down based; open upper on "long")
# (lo, hi_inclusive)  —  hi=None means unbounded.
# ---------------------------------------------------------------------------

THROW_DEPTH_BUCKETS: dict[str, tuple[int, int | None]] = {
    "short": (0, 3),
    "intermediate": (4, 8),
    "long": (9, None),
}

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters (exact nflfastR parity)
# ---------------------------------------------------------------------------

XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eta": 0.025,
    "gamma": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 4,
    "min_child_weight": 6,
    "eval_metric": "logloss",
}

XGB_NROUNDS: int = 560

# ---------------------------------------------------------------------------
# Pass-play type filter (ESPN playType values)
# ---------------------------------------------------------------------------

PASS_PLAY_TYPES: frozenset[str] = frozenset(
    {
        "Pass Reception",
        "Pass Incompletion",
        "Pass Interception Return",
        "Passing Touchdown",
        "Sack",
        "Interception Return Touchdown",
        "Pass",
    }
)
# final.json play-type values (dotted key type.text); additive for backward compat.
PASS_PLAY_TYPES = PASS_PLAY_TYPES | {"Pass Completion", "Interception Return"}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

# Training-window floor. Approach A uses ONLY game-state features (down, distance,
# yards_to_goal, score_diff, seconds_remaining, is_home, period, passing_down) plus
# the `completion` target — all of which are 100% populated for every ESPN season
# back to 2004 (verified: completion non-null == 1.000 and a sane 0.50-0.59 rate per
# season 2004-2025). The previous 2014 floor was a LEGACY artifact of the abandoned
# air-yards "Approach B" (CFBD air_yards only had usable coverage from ~2014; that
# approach was found INFEASIBLE — see __init__/FEASIBILITY). Approach A has no such
# constraint, so the floor is 2004 (3x more training data, and the rule-era factor
# now spans all four buckets instead of collapsing to era2/era3).
MIN_SEASON: int = 2004
MODEL_FILENAME: str = "cfb_cp_model.ubj"
