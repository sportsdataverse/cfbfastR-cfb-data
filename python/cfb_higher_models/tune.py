"""Hyperparameter search for the game-level GBM.

The parameters in ``head_gbm`` (eta 0.03, depth 4, subsample 0.8, colsample
0.6, min_child_weight 20, lambda 2.0) were hand-picked and never tuned. They
are defensible defaults for a small, noisy, heavily-correlated feature set,
but "defensible" is not "measured".

THE OBJECTIVE IS THE REPORTED METRIC. Tuning against a different split or a
different loss than the one used to report results is how a model comes to
look better in the tuner than in production. This searches on walk-forward MAE
-- train on prior seasons, score the next -- exactly as ``walk_forward`` does.

Search runs on a SEASON SUBSET by default. A full 12-season walk-forward is
~9 fits per trial; 60 trials is 540 fits. The subset keeps the search honest
about time while the confirmation run uses the full span.

A tuned result is not automatically an improvement: the winner still has to
clear the incumbent under `significance.compare_oof`, season-clustered. At the
effect sizes seen here (0.01-0.06 MAE) an untested "win" is indistinguishable
from a lucky draw over 60 trials -- which is exactly what a large search is
good at manufacturing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .backtest import season_splits
from .train_game import _xy

#: Hand-picked incumbent, kept here so a tuned candidate always has something
#: concrete to beat and so the comparison is reproducible.
BASELINE_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 20,
    "reg_lambda": 2.0,
    "nthread": 4,
}
BASELINE_ROUNDS = 400


def _walk_forward_mae(
    frame: pl.DataFrame, feats: list[str], params: dict, rounds: int, min_train: int = 3
) -> float:
    import xgboost as xgb

    errs = []
    for train_seasons, test_season in season_splits(frame, min_train=min_train):
        tr = frame.filter(pl.col("season").is_in(train_seasons))
        te = frame.filter(pl.col("season") == test_season)
        if not tr.height or not te.height:
            continue
        Xtr, ytr = _xy(tr, feats)
        Xte, yte = _xy(te, feats)
        bst = xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=rounds)
        errs.append(np.abs(bst.predict(xgb.DMatrix(Xte)) - yte))
    if not errs:
        raise ValueError("no folds produced -- too few seasons?")
    return float(np.concatenate(errs).mean())


def search(
    frame: pl.DataFrame,
    feats: list[str],
    *,
    n_trials: int = 40,
    seed: int = 17,
    out_dir: str | Path = "artifacts/higher_models",
) -> dict:
    """Optuna TPE over the GBM's parameters, scored on walk-forward MAE."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base = _walk_forward_mae(frame, feats, BASELINE_PARAMS, BASELINE_ROUNDS)
    print(f"incumbent (hand-picked): MAE {base:.4f}")

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "reg:squarederror",
            "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 80, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "nthread": 4,
        }
        rounds = trial.suggest_int("num_boost_round", 100, 900, step=50)
        mae = _walk_forward_mae(frame, feats, params, rounds)
        print(
            f"  trial {trial.number:>3}: MAE {mae:.4f}  eta={params['eta']:.3f} "
            f"depth={params['max_depth']} mcw={params['min_child_weight']} "
            f"lambda={params['reg_lambda']:.2f} rounds={rounds}",
            flush=True,
        )
        return mae

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)

    best = dict(study.best_params)
    rounds = best.pop("num_boost_round")
    result = {
        "baseline_mae": base,
        "best_mae": float(study.best_value),
        "improvement": base - float(study.best_value),
        "best_params": best,
        "best_rounds": rounds,
        "n_trials": n_trials,
        "n_features": len(feats),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gbm_tuning.json").write_text(json.dumps(result, indent=2))
    print(
        f"\nbest MAE {result['best_mae']:.4f} vs incumbent {base:.4f} "
        f"(improvement {result['improvement']:+.4f})"
    )
    print(
        "An improvement here is a CANDIDATE, not a result. 40 trials searching "
        "for a 0.0x MAE edge will find one whether or not it exists -- confirm "
        "with significance.compare_oof, season-clustered, before shipping."
    )
    return result
