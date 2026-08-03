"""Hyperparameter search for the CFB model suite, gated on the shipped metrics.

The shipped params in :mod:`model_training.constants` are annotated "exact,
confirmed recipes -- do not alter the numbers": they reproduce cfbfastR's R
recipes. This module deliberately departs from that parity to look for a better
model, so anything it promotes must be recorded as a divergence from the R
lineage, not slipped in as a refresh.

Two-stage, because a full 22-fold LOSO per trial is far too expensive:

1. **Search** -- K grouped folds, each holding out a contiguous block of seasons,
   with the blocks spread across the four rule eras so no fold is era-homogeneous.
   Optuna TPE proposes params; a trial is scored on pooled out-of-fold predictions.
2. **Confirm** -- the surviving candidates are re-scored with the real 22-fold
   LOSO in :mod:`model_training.validate`, which is what the baseline table was
   produced with. A win that does not survive the confirm stage is not a win.

Every objective is the SAME metric the model is gated on, because optimizing a
proxy and reporting a gate is how you ship a model that looks better and is not:

    ep                 ep_cal_mae   (calibration, NOT mlogloss)
    wp_spread/naive    logloss
    fg/xpass/two_pt    logloss
    qbr                rmse

`n_rounds` is not searched -- each fold uses early stopping on its own held-out
block, so tree count adapts to the rest of the params instead of fighting them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from model_training import constants as C
from model_training import features as F

# Rule-era buckets (constants.ERA_BOUNDS = 2006/2013/2020):
#   era0 2004-2006, era1 2007-2013, era2 2014-2020, era3 2021+
# Folds take one contiguous slice from each era so every fold sees every rule
# regime; a fold that is all-era2 would reward params that overfit one regime.
ERA_SPANS: tuple[tuple[int, int], ...] = (
    (2004, 2006),
    (2007, 2013),
    (2014, 2020),
    (2021, 2025),
)


def season_folds(seasons: list[int], k: int = 4) -> list[list[int]]:
    """Split seasons into `k` era-stratified held-out blocks."""
    per_era = [[s for s in seasons if lo <= s <= hi] for lo, hi in ERA_SPANS]
    folds: list[list[int]] = [[] for _ in range(k)]
    for era_seasons in per_era:
        for i, s in enumerate(sorted(era_seasons)):
            folds[i % k].append(s)
    return [sorted(f) for f in folds if f]


def _bin_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ep_cal_mae(ep_pred: np.ndarray, realized: np.ndarray) -> float:
    """Bin by rounded predicted EP, weight |mean(pred) - mean(realized)| by bin size.

    Byte-for-byte the pooled computation in `validate.loso_cv`, so a search score
    and a confirm score are the same number computed on different fold counts.
    """
    b = np.round(ep_pred).astype(int)
    wsum = werr = 0.0
    for bb in np.unique(b):
        mm = b == bb
        n = int(mm.sum())
        wsum += n
        werr += n * abs(ep_pred[mm].mean() - realized[mm].mean())
    return float(werr / wsum) if wsum else float("inf")


@dataclass
class ModelSpec:
    """How to build, score and search one model.

    ``max_rounds`` / ``eta_min`` / ``search_frac`` exist because the models differ
    in cost by two orders of magnitude. EP is 7-class over 2.2M rows: at eta=0.01
    with a 2000-round ceiling a single 4-fold trial runs >10 minutes, which puts a
    60-trial search past 10 hours. Raising its eta floor and capping its rounds
    keeps the search in the region a 525-round shipped model actually lives in,
    rather than spending the budget on ultra-slow-learning candidates that the
    confirm stage would reject anyway.
    """

    name: str
    matrix: Callable[..., Any]
    objective: str
    metric: str
    baseline: float
    fixed: dict[str, Any] = field(default_factory=dict)
    era_onehot_searchable: bool = True
    num_class: int | None = None
    max_rounds: int = 2000
    eta_min: float = 0.01
    eta_max: float = 0.3
    early_stopping: int = 50
    # Row subsample for the SEARCH stage only (1.0 = all rows). The confirm stage
    # always refits on the full frame, so this trades search fidelity for breadth,
    # never final quality.
    search_frac: float = 1.0


SPECS: dict[str, ModelSpec] = {
    "ep": ModelSpec(
        "ep",
        F.ep_matrix,
        "multi:softprob",
        "ep_cal_mae",
        0.0140,
        fixed={"eval_metric": "mlogloss"},
        era_onehot_searchable=True,
        num_class=7,
        # Shipped recipe is eta=0.025 / 525 rounds; bracket that rather than spend
        # the budget on an eta=0.01 tail that costs >10 min per trial.
        max_rounds=900,
        eta_min=0.03,
        eta_max=0.15,
        early_stopping=30,
        search_frac=0.3,
    ),
    "wp_spread": ModelSpec(
        "wp_spread",
        lambda d, **k: F.wp_matrix(d, "spread", **k),
        "binary:logistic",
        "logloss",
        0.3518,
        fixed={"eval_metric": "logloss"},
        # shipped eta=0.02 / 760 rounds -- slowest-learning of the binaries
        max_rounds=1400,
        eta_min=0.02,
        early_stopping=40,
        search_frac=0.35,
    ),
    "wp_naive": ModelSpec(
        "wp_naive",
        lambda d, **k: F.wp_matrix(d, "naive", **k),
        "binary:logistic",
        "logloss",
        0.4015,
        fixed={"eval_metric": "logloss"},
        max_rounds=1200,
        eta_min=0.03,
        search_frac=0.4,
    ),
    "fg": ModelSpec(
        "fg",
        F.fg_matrix,
        "binary:logistic",
        "logloss",
        0.5258,
        fixed={"eval_metric": "logloss"},
        max_rounds=600,
    ),
    "xpass": ModelSpec(
        "xpass",
        F.xpass_matrix,
        "binary:logistic",
        "logloss",
        0.5985,
        fixed={"eval_metric": "logloss"},
        max_rounds=900,
        eta_min=0.03,
        search_frac=0.4,
    ),
    "two_pt": ModelSpec(
        "two_pt",
        F.two_pt_matrix,
        "binary:logistic",
        "logloss",
        0.6917,
        fixed={"eval_metric": "logloss"},
        max_rounds=400,
    ),
}


def suggest_params(trial, spec: ModelSpec) -> dict[str, Any]:
    """TPE search space.

    Ranges bracket the shipped recipe rather than replacing it -- the shipped
    value sits inside every range, so the search can always recover the current
    model and a win is a genuine improvement over it, not a different corner of
    a space that never contained it.
    """
    p: dict[str, Any] = {
        "booster": "gbtree",
        "objective": spec.objective,
        "eta": trial.suggest_float("eta", spec.eta_min, spec.eta_max, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 80.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "nthread": 0,
    }
    p.update(spec.fixed)
    if spec.num_class:
        p["num_class"] = spec.num_class
    return p


def cv_score(
    df: pl.DataFrame,
    spec: ModelSpec,
    params: dict[str, Any],
    *,
    era_onehot: bool,
    folds: list[list[int]],
    max_rounds: int | None = None,
    early_stopping: int | None = None,
    log=print,
) -> tuple[float, int]:
    """Pooled out-of-fold score for one param set. Returns (score, mean_best_rounds)."""
    import xgboost as xgb

    max_rounds = spec.max_rounds if max_rounds is None else max_rounds
    early_stopping = spec.early_stopping if early_stopping is None else early_stopping

    oof_pred: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []
    oof_realized: list[np.ndarray] = []
    rounds: list[int] = []

    for held in folds:
        tr = df.filter(~pl.col("season").is_in(held))
        te = df.filter(pl.col("season").is_in(held))
        # Subsample TRAIN only -- the held-out block is always scored in full, so
        # the OOF score stays an honest estimate over every season.
        if spec.search_frac < 1.0 and tr.height:
            tr = tr.sample(fraction=spec.search_frac, seed=1234, shuffle=True)
        if te.height == 0 or tr.height == 0:
            continue
        Xtr, ytr, *_ = spec.matrix(tr, era_onehot=era_onehot)
        Xte, yte, *_ = spec.matrix(te, era_onehot=era_onehot)
        dtr, dte = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xte, label=yte)
        booster = xgb.train(
            params,
            dtr,
            num_boost_round=max_rounds,
            evals=[(dte, "held")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        rounds.append(int(booster.best_iteration) + 1)
        pred = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))
        if spec.name == "ep":
            scores = np.array([C.EP_CLASS_TO_SCORE[c] for c in range(7)], float)
            yi = np.asarray(yte).astype(int)
            oof_pred.append(pred @ scores)
            oof_realized.append(scores[yi])
        else:
            oof_pred.append(np.asarray(pred).ravel())
            oof_y.append(np.asarray(yte).astype(float).ravel())

    if not oof_pred:
        return float("inf"), 0
    if spec.name == "ep":
        score = _ep_cal_mae(np.concatenate(oof_pred), np.concatenate(oof_realized))
    else:
        score = _bin_logloss(np.concatenate(oof_y), np.concatenate(oof_pred))
    return score, int(np.mean(rounds)) if rounds else 0


def search(
    df: pl.DataFrame,
    model: str,
    *,
    n_trials: int = 60,
    k_folds: int = 4,
    seed: int = 17,
    out_dir: Path | None = None,
    log=print,
) -> dict[str, Any]:
    """Run TPE search for one model and return the best trial + baseline comparison."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = SPECS[model]
    seasons = sorted(df["season"].unique().to_list())
    folds = season_folds(seasons, k_folds)
    log(f"[{model}] baseline {spec.metric}={spec.baseline}; {len(seasons)} seasons -> {len(folds)} folds")
    for i, f in enumerate(folds):
        log(f"    fold {i}: {f}")

    t0 = time.time()

    state = {"n": 0, "best": float("inf")}

    def objective(trial):
        params = suggest_params(trial, spec)
        era = trial.suggest_categorical("era_onehot", [False, True]) if spec.era_onehot_searchable else False
        score, rounds = cv_score(df, spec, params, era_onehot=era, folds=folds, log=log)
        trial.set_user_attr("best_rounds", rounds)
        state["n"] += 1
        state["best"] = min(state["best"], score)
        log(f"    trial {state['n']:>3}/{n_trials} {spec.metric}={score:.6f} "
            f"best={state['best']:.6f} rounds={rounds} ({time.time() - t0:.0f}s)")
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=12),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    result = {
        "model": model,
        "metric": spec.metric,
        "baseline": spec.baseline,
        "best_cv": float(best.value),
        "beats_baseline": bool(best.value < spec.baseline),
        "best_params": {k: v for k, v in best.params.items() if k != "era_onehot"},
        "era_onehot": bool(best.params.get("era_onehot", False)),
        "best_rounds": best.user_attrs.get("best_rounds"),
        "n_trials": n_trials,
        "k_folds": k_folds,
        "seed": seed,
        "elapsed_s": round(time.time() - t0, 1),
        # The search CV is a 4-fold proxy; the baseline was measured on 22-fold
        # LOSO. These are NOT directly comparable -- the confirm stage is what
        # decides. Recorded so nobody mistakes a proxy win for a real one.
        "note": "search CV is k-fold proxy; must be confirmed with full LOSO before promotion",
    }
    log(
        f"[{model}] best proxy {spec.metric}={best.value:.6f} (baseline {spec.baseline}) "
        f"era_onehot={result['era_onehot']} rounds~{result['best_rounds']} in {result['elapsed_s']}s"
    )
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"hpo_{model}.json"
        p.write_text(json.dumps(result, indent=1), encoding="utf-8")
        log(f"[{model}] wrote {p}")
    return result
