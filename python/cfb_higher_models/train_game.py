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
    "efficiency": (
        "success",
        "EPAplay",
        "EPAdrive",
        "EPAgame",
        "TEPA",
        "early_down_EPA",
    ),
    "explosive": ("explosive", "nonExplosiveEpaPerPlay"),
    "trench": ("line_yards", "play_stuffed", "opportunity_rate", "havoc"),
    "finishing": ("available_yards", "gained_yards", "red_zone"),
    "situational": ("third_down", "late_down"),
    "pace": ("plays", "drives", "passrate", "rushrate"),
    "field_pos": ("start_position",),
    "volume": ("yards", "valid_games"),
}


#: The recommended feature set: 60 columns, not 244.
#:
#: Measured walk-forward on the corrected spine, 2014-2025 (n=5,089):
#:
#:     all (244)                 MAE 13.027  Brier 0.1875  cal_err 0.052
#:     minus efficiency (167)    MAE 12.962  Brier 0.1864  cal_err 0.045
#:     core+prior+finishing (60) MAE 12.966  Brier 0.1870  cal_err 0.043
#:     core: other+rating (35)   MAE 13.135  Brier 0.1889  cal_err 0.053
#:     other only (18)           MAE 13.217  Brier 0.1910  cal_err 0.068
#:
#: 60 features match 167 to within 0.004 MAE and beat the full 244 on every
#: metric. Dropping 77 efficiency columns IMPROVES the model -- they are
#: re-measurements of a dimension already present, and their only marginal
#: contribution is variance. Feature count was never the constraint.
LEAN_FAMILIES = ("other", "rating", "finishing")


def lean_features(diffs: list[str]) -> list[str]:
    """The 60-ish column set: cfb_ratings block + ratings + finishing + carryover.

    ``other`` is the rt_* cfb_ratings block (special teams, FEI, adj_net) --
    the only family whose removal measurably hurts, confirmed three separate
    ways (family ablation, leave-one-block-out, and this prune).
    """
    keep, seen = [], set()
    for c in diffs:
        if family_of(c) in LEAN_FAMILIES or c.startswith(("prior_", "blend_")):
            if c not in seen:
                seen.add(c)
                keep.append(c)
    return keep


def family_of(col: str) -> str:
    for fam, frags in FAMILIES.items():
        if any(f in col for f in frags):
            return fam
    return "other"


def _xy(
    df: pl.DataFrame, feats: list[str], target: str = "margin"
) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feats).to_numpy().astype(np.float64)
    y = df[target].to_numpy()
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y.astype(np.float64)


#: Features that are SHARED by both teams and therefore vanish under a
#: home-minus-away difference. `valid_games_diff` is ~0 in every week because
#: both teams have played the same number of games -- so the attenuation signal
#: that Phase 1 measured at 0.58 MAE is invisible to a diff-only model. These
#: restore it. Differencing is the right encoding for STRENGTH and the wrong
#: one for CONTEXT.
CONTEXT_FEATURES = ("week", "min_games", "neutral_site")

#: Schedule-spot columns from features.add_rest, present only when the frame
#: was built with enrich=True. Off-substrate: rest and byes appear nowhere in
#: the play-level data, so unlike a 201st efficiency metric they are not
#: another measurement of the same latent team quality.
REST_FEATURES = ("rest_diff", "bye_home", "bye_away")


def add_context(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the shared-context columns a pure diff encoding destroys."""
    games = (
        pl.min_horizontal("valid_games_home", "valid_games_away")
        if "valid_games_home" in frame.columns
        else (pl.col("week") - 1).cast(pl.Float64)
    )
    return frame.with_columns(
        games.cast(pl.Float64).alias("min_games"),
        pl.col("week").cast(pl.Float64),
        pl.col("neutral_site").cast(pl.Float64),
    )


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


def head_gbm_wp(train, test, *, feats, rounds: int = 400, **_):
    """Classify home_won DIRECTLY instead of deriving WP from a margin.

    A margin model asked for a win probability has to assume a residual shape
    (we use a normal with a fitted sd). Modelling the binary outcome directly
    drops that assumption -- but loses the margin, so it is a *different head*,
    not a replacement. Returns probabilities; scored on Brier/logloss only.
    """
    import xgboost as xgb

    Xtr, ytr = _xy(train, feats, target="home_won")
    Xte, _ = _xy(test, feats, target="home_won")
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 20,
        "reg_lambda": 2.0,
        "nthread": 4,
    }
    bst = xgb.train(p, xgb.DMatrix(Xtr, label=ytr), num_boost_round=rounds)
    return bst.predict(xgb.DMatrix(Xte))


HEADS = {
    "shipped": head_shipped,
    "closed_form_refit": head_closed_form,
    "ridge_all": head_ridge,
    "gbm_all": head_gbm,
}


def ablate(frame: pl.DataFrame, diffs: list[str], *, min_train: int = 3) -> dict:
    """Which feature families actually carry pregame signal?

    Two passes per family: ONLY that family (does it stand alone?) and
    everything EXCEPT it (is it redundant?). A family can score well alone and
    still add nothing on top of the rest -- that is the useful distinction, and
    a single "importance" number hides it.
    """
    ctx = list(CONTEXT_FEATURES)
    by_fam: dict[str, list[str]] = {}
    for c in diffs:
        by_fam.setdefault(family_of(c), []).append(c)

    out = {}
    print(f"{'family':>12} {'n_feat':>7} {'only':>8} {'without':>9}")
    for fam, cols in sorted(by_fam.items()):
        row = {}
        for label, fs in (
            ("only", cols + ctx),
            ("without", [c for c in diffs if c not in cols] + ctx),
        ):
            try:
                rep, _ = walk_forward(
                    frame,
                    lambda tr, te, _f=fs: head_gbm(tr, te, feats=_f),
                    name=f"{fam}:{label}",
                    min_train=min_train,
                )
                row[label] = rep.margin["mae"]
            except Exception:  # noqa: BLE001
                row[label] = float("nan")
        out[fam] = row
        print(f"{fam:>12} {len(cols):>7} {row['only']:>8.2f} {row['without']:>9.2f}")
    return out


def compare_feature_sets(
    frame: pl.DataFrame, sets: dict[str, list[str]], *, min_train: int = 3, head=None
) -> dict:
    """A/B one head across named feature sets on ONE frame.

    The frame is fixed so the only thing varying is the feature list -- the
    comparison a "does X help?" question actually asks. Building a fresh frame
    per arm (as sweep_blend_k must, since k changes the data) would confound
    the feature change with any nondeterminism in the build.
    """
    head = head or head_gbm
    out = {}
    print(f"{'set':>22} {'MAE':>8} {'Brier':>8} {'corr':>7} {'cal_err':>8}")
    for name, feats in sets.items():
        rep, _ = walk_forward(
            frame,
            lambda tr, te, _f=feats: head(tr, te, feats=_f),
            name=name,
            min_train=min_train,
        )
        out[name] = {"margin": rep.margin, "wp": rep.wp, "n_feat": len(feats)}
        print(
            f"{name:>22} {rep.margin['mae']:>8.3f} {rep.wp['brier']:>8.4f} "
            f"{rep.margin['corr']:>7.3f} {rep.wp['max_cal_err']:>8.3f}"
        )
    return out


def sweep_blend_k(seasons: list[int], ks=(2.0, 4.0, 8.0, 12.0, 20.0)) -> dict:
    """How long should last season's rating keep mattering?

    ``k`` is the games-played half-life in ``n/(n+k)``: k=4 means a team is
    judged half on prior form until about a month in. The by-week profile at
    k=4 showed weeks 5-8 (MAE 13.32) WORSE than weeks 2-4 (12.52) -- the
    transition zone where the prior has been discarded but the current season
    is still thin. That is the signature of a prior decaying too fast.
    """
    out = {}
    print(
        f"{'k':>6} {'MAE':>7} {'Brier':>8} {'wk2-4':>7} {'wk5-8':>7} {'wk9-12':>7} {'wk13+':>7}"
    )
    for k in ks:
        frame = build_game_frame(seasons, enrich=True, blend_k=k, verbose=False)
        frame, diffs = diff_features(frame, paired_features(frame))
        frame = add_context(frame)
        feats = diffs + list(CONTEXT_FEATURES)
        rep, oof = walk_forward(
            frame,
            lambda tr, te, _f=feats: head_gbm(tr, te, feats=_f),
            name=f"k={k}",
            min_train=3,
        )
        d = oof.with_columns(
            (pl.col("pred_margin") - pl.col("margin")).abs().alias("ae")
        )
        buckets = []
        for lo, hi in ((2, 4), (5, 8), (9, 12), (13, 25)):
            s = d.filter(pl.col("week").is_between(lo, hi))
            buckets.append(s["ae"].mean() if s.height else float("nan"))
        out[str(k)] = {
            "mae": rep.margin["mae"],
            "brier": rep.wp["brier"],
            "by_week": buckets,
        }
        print(
            f"{k:>6.1f} {rep.margin['mae']:>7.3f} {rep.wp['brier']:>8.4f} "
            + " ".join(f"{b:>7.2f}" for b in buckets)
        )
    return out


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


def main(
    seasons: list[int] | None = None,
    out_dir: str = "artifacts/higher_models",
    *,
    enrich: bool = False,
    blend_k: float = 4.0,
    do_ablate: bool = False,
) -> int:
    seasons = seasons or list(range(2016, 2026))
    frame = build_game_frame(seasons, enrich=enrich, blend_k=blend_k)
    frame, diffs = diff_features(frame, paired_features(frame))
    frame = add_context(frame)
    feats = diffs + list(CONTEXT_FEATURES)
    print(
        f"frame: {frame.height} games, {len(diffs)} diff + "
        f"{len(CONTEXT_FEATURES)} context features, "
        f"seasons {min(seasons)}-{max(seasons)}\n"
    )

    res = run(frame, feats=feats)

    # Direct win-probability head, scored on its own terms (it has no margin).
    from .backtest import season_splits
    from .metrics import evaluate as _eval

    try:
        preds, tests = [], []
        for tr_s, te_s in season_splits(frame, min_train=3):
            tr = frame.filter(pl.col("season").is_in(tr_s))
            te = frame.filter(pl.col("season") == te_s)
            preds.append(head_gbm_wp(tr, te, feats=feats))
            tests.append(te)
        te_all = pl.concat(tests)
        rep = _eval(
            "gbm_winprob (direct classifier)",
            boundary="as_of",
            prob=np.concatenate(preds),
            won=te_all["home_won"].to_numpy(),
        )
        print(rep, "\n")
        res["gbm_winprob"] = {"wp": rep.wp}
    except Exception as e:  # noqa: BLE001
        print(f"  gbm_winprob: FAILED ({type(e).__name__}: {e})")

    # Where does the gain land? The prior-season blend should help EARLY weeks
    # (few games -> noisy rating) and do nothing late. If the by-week profile
    # is flat, the blend is not doing what it claims and the pooled MAE gain is
    # coming from somewhere else.
    try:
        rep, oof = walk_forward(
            frame,
            lambda tr, te: head_gbm(tr, te, feats=feats),
            name="_byweek",
            min_train=3,
        )
        print("gbm by week bucket (MAE):")
        d = oof.with_columns(
            (pl.col("pred_margin") - pl.col("margin")).abs().alias("ae")
        )
        for lo, hi, lbl in (
            (2, 4, "2-4"),
            (5, 8, "5-8"),
            (9, 12, "9-12"),
            (13, 25, "13+"),
        ):
            s = d.filter(pl.col("week").is_between(lo, hi))
            if s.height:
                print(f"    wk {lbl:>5}  n={s.height:>5}  MAE={s['ae'].mean():.2f}")
        print()
    except Exception as e:  # noqa: BLE001
        print(f"  by-week: FAILED ({type(e).__name__}: {e})")

    if do_ablate:
        print("=== feature-family ablation (MAE) ===")
        res["ablation"] = ablate(frame, diffs)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "game_heads.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {Path(out_dir) / 'game_heads.json'}")
    return 0
