"""Metric runners for model reports — provenance extraction, classification metrics, importance."""
from __future__ import annotations


def provenance_from_card(card: dict) -> dict:
    """Extract provenance fields from a model card, tolerating missing keys.

    Args:
        card: Model card dict with optional keys (features, hyperparameters,
            training_seasons, trained_date, xgboost_version).

    Returns:
        Dict with keys (features, hyperparameters, training_seasons, trained_date,
        xgboost_version), using safe defaults for missing keys.
    """
    return {
        "features": card.get("features") or [],
        "hyperparameters": card.get("hyperparameters") or {},
        "training_seasons": card.get("training_seasons"),
        "trained_date": card.get("trained_date"),
        "xgboost_version": card.get("xgboost_version"),
    }


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Compute classification metrics: log-loss and Brier score (binary only).

    Args:
        y_true: True labels (array-like, 0/1).
        y_pred: Predicted probabilities (array-like, shape same as y_true).

    Returns:
        Dict with keys: n (sample count), log_loss (sklearn), brier_score (binary only).
    """
    import numpy as np
    from sklearn.metrics import log_loss

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape[0] == 0:
        return {"n": 0}
    if yp.ndim == 1:  # binary
        out = {"n": int(yt.shape[0]), "log_loss": float(log_loss(yt, yp, labels=[0, 1]))}
        out["brier_score"] = float(np.mean((yp - yt) ** 2))
    else:  # multiclass
        out = {"n": int(yt.shape[0]), "log_loss": float(log_loss(yt, yp))}
    return out


def xgb_importance(model_path, top_n: int = 15) -> dict:
    """Extract XGBoost model feature importance (gain).

    Args:
        model_path: Path to XGBoost model file (.ubj, .json, etc.).
        top_n: Number of top features to return (default: 15).

    Returns:
        Dict mapping feature names to importance scores (rounded to 4 decimals).
    """
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(model_path))
    score = b.get_score(importance_type="gain")
    top = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {k: round(float(v), 4) for k, v in top}


def loso_metrics(model_type: str, loso_dir) -> dict:
    """Real out-of-fold LOSO metrics for a model, read from its OOF artifact.

    Sources by model_type (all under ``loso_dir``):
      * ``wp_spread`` -> ``loso_wp_oof.parquet`` (season,y,wp_pred) -> log-loss/Brier/AUC
      * ``ep``        -> ``loso_ep_oof.parquet`` (season,y,ep_pred,realized) -> EP-value
                        calibration MAE (1-pt bins) + mean predicted/realized points
      * ``qbr``       -> ``loso_qbr_oof.parquet`` (season,y,qbr_pred) -> RMSE/MAE/R2/corr
      * ``cpoe``      -> ``cpoe/loso_cv.json`` summary (mean log-loss/Brier)

    Returns an empty dict when no OOF artifact is present (so the report falls
    back to its "requires a warmed cache" note).
    """
    import json
    from pathlib import Path

    import numpy as np

    d = Path(loso_dir)
    if model_type == "cpoe":
        cj = d / "cpoe" / "loso_cv.json"
        if not cj.exists():
            return {}
        s = (json.loads(cj.read_text()) or {}).get("summary", {})
        if not s:
            return {}
        return {"loso_log_loss": round(float(s["mean_log_loss"]), 4),
                "loso_brier": round(float(s["mean_brier_score"]), 4)}

    fname = {"wp_spread": "loso_wp_oof.parquet", "ep": "loso_ep_oof.parquet",
             "qbr": "loso_qbr_oof.parquet"}.get(model_type)
    if not fname or not (d / fname).exists():
        return {}

    import polars as pl
    df = pl.read_parquet(d / fname)
    if model_type == "wp_spread":
        from sklearn.metrics import log_loss
        y = df["y"].to_numpy().astype(int)
        p = np.clip(df["wp_pred"].to_numpy(), 1e-15, 1 - 1e-15)
        order = np.argsort(p, kind="mergesort")
        r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
        npos, nneg = float(y.sum()), float((1 - y).sum())
        auc = (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")
        return {"loso_log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
                "loso_brier": round(float(np.mean((p - y) ** 2)), 4),
                "loso_auc": round(float(auc), 4), "loso_n": int(len(y))}
    if model_type == "ep":
        ep = df["ep_pred"].to_numpy(); rl = df["realized"].to_numpy()
        b = np.round(ep).astype(int); wsum = werr = 0.0
        for bb in np.unique(b):
            mm = b == bb; n = int(mm.sum()); wsum += n; werr += n * abs(ep[mm].mean() - rl[mm].mean())
        return {"loso_ep_cal_mae_pts": round(float(werr / wsum), 4),
                "loso_mean_pred_ep": round(float(ep.mean()), 4),
                "loso_mean_realized": round(float(rl.mean()), 4), "loso_n": int(len(ep))}
    # qbr
    y = df["y"].to_numpy(); p = df["qbr_pred"].to_numpy()
    ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"loso_rmse": round(float(np.sqrt(np.mean((p - y) ** 2))), 4),
            "loso_mae": round(float(np.mean(np.abs(p - y))), 4),
            "loso_r2": round(1 - ss_res / ss_tot, 4) if ss_tot else float("nan"),
            "loso_corr": round(float(np.corrcoef(p, y)[0, 1]), 4), "loso_n": int(len(y))}


def rb_eval_metrics(loso_parquet) -> dict:
    """Compute RB evaluation metrics from LOSO cross-validation parquet.

    Reads a parquet file containing exp_rb_epa and target columns,
    bins predictions, and computes weighted R² and calibration error.

    Args:
        loso_parquet: Path to xrepa_loso.parquet (str or Path).

    Returns:
        Dict with keys: weighted_r2, weighted_cal_error (rounded to 4 decimals),
        and n (sample count).
    """
    import polars as pl
    from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2

    cv = pl.read_parquet(loso_parquet)
    tbl = calibration_table(cv)
    return {
        "weighted_r2": round(float(weighted_r2(tbl)), 4),
        "weighted_cal_error": round(float(weighted_cal_error(tbl)), 4),
        "n": int(cv.height),
    }
