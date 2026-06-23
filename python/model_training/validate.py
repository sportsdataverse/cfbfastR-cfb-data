"""Validation: prediction-parity vs reference models + LOSO calibration tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb


def prediction_parity(model_a: xgb.Booster, model_b: xgb.Booster, X: pd.DataFrame,
                      tol: float = 1e-3) -> dict:
    d = xgb.DMatrix(X)
    pa, pb = model_a.predict(d), model_b.predict(d)
    max_abs = float(np.max(np.abs(pa - pb)))
    return {"max_abs_diff": max_abs, "within_tol": max_abs <= tol, "tol": tol}


def calibration_table(pred_prob, outcome, by, bin_size: float = 0.05) -> pl.DataFrame:
    df = pl.DataFrame({"pred": pred_prob, "outcome": outcome, "by": by})
    df = df.with_columns(bin=(pl.col("pred") / bin_size).round() * bin_size)
    return (
        df.group_by(["by", "bin"])
        .agg(n_plays=pl.len(), n_pos=pl.col("outcome").sum())
        .with_columns(actual=pl.col("n_pos") / pl.col("n_plays"))
        .sort(["by", "bin"])
    )


def weighted_cal_error(table: pl.DataFrame) -> float:
    t = table.with_columns(diff=(pl.col("bin") - pl.col("actual")).abs())
    per = t.group_by("by").agg(
        wce=(pl.col("diff") * pl.col("n_plays")).sum() / pl.col("n_plays").sum(),
        n=pl.col("n_pos").sum(),
    )
    return float((per["wce"] * per["n"]).sum() / per["n"].sum())


# --- Leave-One-Season-Out cross-validation -----------------------------------
# Honest out-of-sample evaluation: for each season s, train on all OTHER seasons
# and predict the held-out season s, then pool the out-of-fold predictions. The
# shipped XGBoost params/rounds (constants.py) are used unchanged.

def _mc_logloss(y, P) -> float:
    P = np.clip(P, 1e-15, 1 - 1e-15)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def _bin_logloss(y, p) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y, p) -> float:
    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p), float)
    r[order] = np.arange(1, len(p) + 1)
    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def loso_cv(df: pl.DataFrame, model_type: str, espn_qbr: pl.DataFrame | None = None,
            log=print) -> dict:
    """Leave-one-season-out CV for ``model_type`` in {ep, wp, qbr}.

    Args:
        df: training frame (``pbp_full.parquet``); must carry ``add_winner`` columns
            for WP and ``season``.
        model_type: ``"ep"`` | ``"wp"`` | ``"qbr"``.
        espn_qbr: required for ``model_type == "qbr"`` — ESPN QBR reference with
            ``game_id`` / ``passer_player_name`` / ``raw_qbr``.
        log: per-fold progress callback (default ``print``); pass ``lambda *_: None`` to silence.

    Returns:
        ``{"model": str, "pooled": {...}, "per_season": [...], "oof": pl.DataFrame}``.
        Pooled keys: EP ``mlogloss``/``accuracy``/``ep_cal_mae``; WP
        ``logloss``/``brier``/``auc``/``weighted_cal_err``; QBR ``rmse``/``mae``/``r2``/``corr``.
    """
    import xgboost as xgb

    from . import constants as C
    from .features import (
        ep_matrix,
        fg_matrix,
        qbr_matrix,
        two_pt_matrix,
        wp_matrix,
        xpass_matrix,
    )
    from .train_ep import train_ep
    from .train_fg import train_fg
    from .train_qbr import train_qbr
    from .train_two_pt import train_two_pt
    from .train_wp import train_wp
    from .train_xpass import train_xpass

    if model_type not in ("ep", "wp", "qbr", "fg", "xpass", "two_pt"):
        raise ValueError(f"model_type must be ep|wp|qbr|fg|xpass|two_pt, got {model_type!r}")
    if model_type == "qbr" and espn_qbr is None:
        raise ValueError("model_type='qbr' requires espn_qbr reference")
    seasons = sorted(df["season"].unique().to_list())
    per_season: list[dict] = []
    frames: list[pl.DataFrame] = []

    for s in seasons:
        tr = df.filter(pl.col("season") != s)
        te = df.filter(pl.col("season") == s)
        if model_type == "ep":
            m = train_ep(tr)
            X, y, _ = ep_matrix(te)
            y = y.astype(int)
            P = m.predict(xgb.DMatrix(X))
            scores = np.array([C.EP_CLASS_TO_SCORE[c] for c in range(7)], float)
            ep_pred = P @ scores
            row = {"season": s, "n": int(len(y)),
                   "mlogloss": _mc_logloss(y, P), "accuracy": float(np.mean(P.argmax(1) == y))}
            frames.append(pl.DataFrame({"season": np.full(len(y), s), "y": y,
                                        "ep_pred": ep_pred, "realized": scores[y]}))
        elif model_type == "wp":
            m = train_wp(tr, variant="spread", stage=2)
            X, y, _ = wp_matrix(te, "spread")
            y = y.astype(int)
            p = m.predict(xgb.DMatrix(X))
            row = {"season": s, "n": int(len(y)), "logloss": _bin_logloss(y, p),
                   "brier": float(np.mean((p - y) ** 2)), "auc": _auc(y, p)}
            frames.append(pl.DataFrame({"season": np.full(len(y), s), "y": y, "wp_pred": p}))
        elif model_type == "fg":
            m = train_fg(tr)
            X, y, _ = fg_matrix(te)
            y = y.astype(int)
            p = m.predict(xgb.DMatrix(X))
            row = {"season": s, "n": int(len(y)), "logloss": _bin_logloss(y, p),
                   "brier": float(np.mean((p - y) ** 2)), "auc": _auc(y, p)}
            frames.append(pl.DataFrame({"season": np.full(len(y), s),
                                        "yards_to_goal": X["yards_to_goal"].to_numpy(),
                                        "made": y, "fg_pred": p}))
        elif model_type == "xpass":
            m = train_xpass(tr)
            X, y, _ = xpass_matrix(te)
            y = y.astype(int)
            p = m.predict(xgb.DMatrix(X))
            row = {"season": s, "n": int(len(y)), "logloss": _bin_logloss(y, p),
                   "brier": float(np.mean((p - y) ** 2)), "auc": _auc(y, p)}
            frames.append(pl.DataFrame({"season": np.full(len(y), s),
                                        "down": X["down"].to_numpy(),
                                        "is_pass": y, "xpass": p}))
        elif model_type == "two_pt":
            m = train_two_pt(tr)
            X, y, _ = two_pt_matrix(te)
            y = y.astype(int)
            p = m.predict(xgb.DMatrix(X))
            row = {"season": s, "n": int(len(y)), "logloss": _bin_logloss(y, p),
                   "brier": float(np.mean((p - y) ** 2)), "auc": _auc(y, p)}
            frames.append(pl.DataFrame({"season": np.full(len(y), s),
                                        "made": y, "two_pt_pred": p}))
        else:  # qbr
            m = train_qbr(tr, espn_qbr)
            X, _, keys = qbr_matrix(te)
            feat = pl.from_pandas(keys).hstack(pl.from_pandas(X))
            j = feat.join(espn_qbr, on=["game_id", "passer_player_name"], how="inner").drop_nulls("raw_qbr")
            if j.height == 0:
                row = {"season": s, "n": 0, "rmse": None, "mae": None}
                log(f"[qbr] fold {s}: no joined rows")
                per_season.append(row)
                continue
            pj = m.predict(xgb.DMatrix(j.select(C.QBR_FEATURES).to_pandas()))
            yj = j["raw_qbr"].to_numpy()
            row = {"season": s, "n": int(len(yj)),
                   "rmse": float(np.sqrt(np.mean((pj - yj) ** 2))), "mae": float(np.mean(np.abs(pj - yj)))}
            frames.append(pl.DataFrame({"season": np.full(len(yj), s), "y": yj, "qbr_pred": pj}))
        per_season.append(row)
        log(f"[{model_type}] fold {s}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                    for k, v in row.items() if k != "season"))

    oof = pl.concat(frames) if frames else pl.DataFrame()
    pooled: dict = {}
    if model_type == "ep" and oof.height:
        y = oof["y"].to_numpy().astype(int)
        ep_pred, realized = oof["ep_pred"].to_numpy(), oof["realized"].to_numpy()
        b = np.round(ep_pred).astype(int)
        wsum = werr = 0.0
        for bb in np.unique(b):
            mm = b == bb
            n = int(mm.sum())
            wsum += n
            werr += n * abs(ep_pred[mm].mean() - realized[mm].mean())
        # rebuild full prob matrix is unnecessary; report pooled accuracy/mlogloss per-season-weighted
        pooled = {"mlogloss": float(np.average([r["mlogloss"] for r in per_season], weights=[r["n"] for r in per_season])),
                  "accuracy": float(np.average([r["accuracy"] for r in per_season], weights=[r["n"] for r in per_season])),
                  "ep_cal_mae": float(werr / wsum), "mean_pred_ep": float(ep_pred.mean()),
                  "mean_realized": float(realized.mean())}
    elif model_type == "wp" and oof.height:
        y = oof["y"].to_numpy().astype(int)
        p = oof["wp_pred"].to_numpy()
        tab = calibration_table(p.tolist(), y.tolist(), oof["season"].to_list(), bin_size=0.05)
        pooled = {"logloss": _bin_logloss(y, p), "brier": float(np.mean((p - y) ** 2)),
                  "auc": _auc(y, p), "weighted_cal_err": weighted_cal_error(tab)}
    elif model_type in ("fg", "xpass", "two_pt") and oof.height:
        _pred_col = {"fg": "fg_pred", "xpass": "xpass", "two_pt": "two_pt_pred"}[model_type]
        _y_col = {"fg": "made", "xpass": "is_pass", "two_pt": "made"}[model_type]
        y = oof[_y_col].to_numpy().astype(int)
        p = oof[_pred_col].to_numpy()
        pooled = {"logloss": _bin_logloss(y, p), "brier": float(np.mean((p - y) ** 2)),
                  "auc": _auc(y, p)}
    elif model_type == "qbr" and oof.height:
        y, pj = oof["y"].to_numpy(), oof["qbr_pred"].to_numpy()
        ss_res, ss_tot = float(np.sum((y - pj) ** 2)), float(np.sum((y - y.mean()) ** 2))
        pooled = {"n": int(len(y)), "rmse": float(np.sqrt(np.mean((pj - y) ** 2))),
                  "mae": float(np.mean(np.abs(pj - y))), "r2": (1 - ss_res / ss_tot) if ss_tot else float("nan"),
                  "corr": float(np.corrcoef(pj, y)[0, 1])}
    return {"model": model_type, "pooled": pooled, "per_season": per_season, "oof": oof}
