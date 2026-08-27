"""Feature contract, XGBoost params, and label bounds for the fourth-down yards model.

Recipe source: fourth-downs.ipynb (akeaswaran / Jason Lee lineage), confirmed by
cfb4th:::fd_model tree count: 157 rounds × 76 classes = 11932 trees.
"""
from __future__ import annotations

# --- feature contract (exact column order) ---
# The original cfb4th recipe is 5-feat; we add an ordinal CFB rule-era factor
# (era, 0..3) as a Stage-2 enhancement so the full-history (2004-2025) model can
# absorb the secular drift in fourth-down play across rule eras.
FD_FEATURES: list[str] = [
    "down",
    "distance",
    "yards_to_goal",
    "posteam_total",
    "posteam_spread",
    "era",
]

# --- CFB rule-era factor (ordinal), derived from the play's season ---
# Boundaries track major clock / targeting / tempo rule changes:
#   0: 2004-2006 (pre clock-rule era; 2006 clock experiment)
#   1: 2007-2013 (post-2006 revert, pre-targeting-ejection)
#   2: 2014-2020 (targeting + 10-second runoff + up-tempo)
#   3: 2021+     (modern / post-2020)
FD_SEASON_COL: str = "season"
FD_ERA_BOUNDS: tuple[int, int, int] = (2006, 2013, 2020)
# One-hot era dummies (era-experiment encoding) — one column per bucket.
FD_ERA_ONEHOT_COLS: list[str] = [f"era{i}" for i in range(len(FD_ERA_BOUNDS) + 1)]  # era0..era3
# The 6-feat one-hot variant feature order (ordinal `era` swapped for era0..era3).
FD_FEATURES_ERA_ONEHOT: list[str] = [f for f in FD_FEATURES if f != "era"] + FD_ERA_ONEHOT_COLS

# --- label bounds (clip + offset: label = clip(yardsGained, LOW, HIGH) + OFFSET) ---
FD_CLIP_LOW: int = -10  # 10-yard loss = class 0
FD_CLIP_HIGH: int = 65  # 65-yard gain = class 75
FD_LABEL_OFFSET: int = 10
FD_NUM_CLASS: int = 76  # classes 0..75 covering integer gains -10..65

# --- XGBoost params (exact, from notebook cell 7 + _go_for_it_cfb_mod.R lines 188-200) ---
FD_PARAMS: dict = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": FD_NUM_CLASS,
    "eta": 0.07,
    "gamma": 4.325037e-09,
    "subsample": 0.5385424,
    "colsample_bytree": 0.6666667,
    "max_depth": 4,
    "min_child_weight": 7,
}
FD_NROUNDS: int = 157

# --- source column names in final.json plays ---
FD_SOURCE: dict[str, str] = {
    "down": "start.down",
    "distance": "start.distance",
    "yards_to_goal": "start.yardsToEndzone",
    # posteam_total is DERIVED (not a direct source column)
    # posteam_spread is read from start.pos_team_spread
    "posteam_spread": "start.pos_team_spread",
}

# Spread + total source columns (doc-level, broadcast to every play by CFBPlayProcess)
FD_SPREAD_COL: str = "homeTeamSpread"  # home-team-perspective spread (negative = home favored)
FD_OVERUNDER_COL: str = "overUnder"  # game total
FD_IS_HOME_COL: str = "start.is_home"  # 1/True if possessing team is home
# Label source — per-play yards gained. ESPN enriched final.json (the cfb-raw
# corpus) carries this as ``statYardage``; the cfb4th/CFBD lineage + the
# synthetic unit-test fixtures use ``yardsGained``. Resolve whichever is present
# (statYardage preferred); FD_YARDS_GAINED_COL is the default + model-card label.
FD_YARDS_GAINED_COLS: tuple[str, ...] = ("statYardage", "yardsGained")
FD_YARDS_GAINED_COL: str = "statYardage"  # label source (default)
FD_RUSH_COL: str = "rush"  # boolean/int — play filter
FD_PASS_COL: str = "pass"  # boolean/int — play filter
FD_FIRST_DOWN_PENALTY_COLS: tuple[str, ...] = ("firstD_by_penalty", "start.firstD_by_penalty")
