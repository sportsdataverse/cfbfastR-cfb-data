"""Phase 2/3 -- predictive game-level heads over the as-of feature spine.

Every head is a ``fit_predict(train, test) -> pred_margin`` closure registered in
:data:`HEADS`, scored by the same walk-forward runner against the same
baselines. That uniformity is the point: the reason the shipped constants could
claim 3.23 against a 15.13 reality is that there was no single place where a
candidate had to post an out-of-sample number next to the incumbent.

Feature families (for ablation) are named by their column suffixes in the
384-column weekly substrate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .backtest import shipped_margin, walk_forward
from .data import build_game_frame, diff_features, paired_features

# Column-name fragments that define a feature family. Used for ablation: which
# part of the play-level substrate actually carries pregame signal?
FAMILIES: dict[str, tuple[str, ...]] = {
    "rating": ("adj_off_epa", "adj_def_epa", "net_adj_epa", "strength_faced"),
    "efficiency": ("success", "EPAplay", "EPAdrive", "EPAgame", "TEPA", "early_down_EPA"),
    "explosive": ("explosive", "nonExplosiveEpaPerPlay"),
    "trench": ("line_yards", "play_stuffed", "opportunity_rate", "havoc"),
    "finishing": ("available_yards", "gained_yards", "red_zone"),
    "situational": ("third_down", "late_down"),
    "pace": ("plays", "drives", "passrate", "rushrate"),
    "field_pos": ("start_position",),
    "volume": ("yards", "valid_games"),
}


def family_of(col: str) -> str:
    for fam, frags in FAMILIES.items():
        if any(f in col for f in frags):
            return fam
    return "other"


def _xy(df: pl.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feats).to_numpy().astype(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), df["margin"].to_numpy()


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------
def head_shipped(_train, test, **_):
    return shipped_margin(test)


def head_closed_form(train, test, **_):
    """Phase-1 refit: margin ~ rating_diff + HFA. The honest closed form."""
    from .fit_pregame import fit

    return fit(train).predict(test, use_curve=True)


def head_ridge(train, test, *, feats, alpha: float = 10.0, **_):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = _xy(train, feats)
    Xte, _ = _xy(test, feats)
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def head_gbm(train, test, *, feats, params=None, rounds: int = 400, **_):
    import xgboost as xgb

    Xtr, ytr = _xy(train, feats)
    Xte, _ = _xy(test, feats)
    p = {
        "objective": "reg:squarederror",
        "eta": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 20,
        "reg_lambda": 2.0,
        "nthread": 4,
        **(params or {}),
    }
    bst = xgb.train(p, xgb.DMatrix(Xtr, label=ytr), num_boost_round=rounds)
    return bst.predict(xgb.DMatrix(Xte))


HEADS = {
    "shipped": head_shipped,
    "closed_form_refit": head_closed_form,
    "ridge_all": head_ridge,
    "gbm_all": head_gbm,
}


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run(
    frame: pl.DataFrame,
    heads: dict | None = None,
    *,
    feats: list[str] | None = None,
    min_train: int = 3,
) -> dict[str, dict]:
    heads = heads or HEADS
    out = {}
    for name, fn in heads.items():
        try:
            rep, oof = walk_forward(
                frame,
                lambda tr, te, _fn=fn: _fn(tr, te, feats=feats),
                name=name,
                min_train=min_train,
            )
        except Exception as e:  # noqa: BLE001 -- one broken head must not kill the sweep
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")
            continue
        print(rep, "\n")
        out[name] = {"margin": rep.margin, "wp": rep.wp, "base": rep.base}
    return out


def main(seasons: list[int] | None = None, out_dir: str = "artifacts/higher_models") -> int:
    seasons = seasons or list(range(2016, 2026))
    frame = build_game_frame(seasons)
    frame, diffs = diff_features(frame, paired_features(frame))
    feats = diffs + ["neutral_site"]
    frame = frame.with_columns(pl.col("neutral_site").cast(pl.Float64))
    print(f"frame: {frame.height} games, {len(diffs)} diff features, seasons {min(seasons)}-{max(seasons)}\n")

    res = run(frame, feats=feats)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "game_heads.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {Path(out_dir) / 'game_heads.json'}")
    return 0
