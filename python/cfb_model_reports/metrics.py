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


def _bin_logloss(y, p):
    import numpy as np

    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y, p):
    import numpy as np

    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p), float)
    r[order] = np.arange(1, len(p) + 1)
    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ep_loso_metrics(oof_parquet) -> dict:
    """Pooled EP metrics from the LOSO out-of-fold parquet.

    Args:
        oof_parquet: parquet with columns ``ep_pred`` (predicted EP) and
            ``realized`` (realized next-score points).

    Returns:
        Dict: ``n``, ``ep_cal_mae`` (weighted, points), ``mean_pred_ep``,
        ``mean_realized``.
    """
    import numpy as np
    import polars as pl

    df = pl.read_parquet(oof_parquet)
    ep_pred = df["ep_pred"].to_numpy()
    realized = df["realized"].to_numpy()
    b = np.round(ep_pred).astype(int)
    wsum = werr = 0.0
    for bb in np.unique(b):
        mm = b == bb
        n = int(mm.sum())
        wsum += n
        werr += n * abs(ep_pred[mm].mean() - realized[mm].mean())
    return {
        "n": int(len(ep_pred)),
        "ep_cal_mae": round(float(werr / wsum), 4),
        "mean_pred_ep": round(float(ep_pred.mean()), 4),
        "mean_realized": round(float(realized.mean()), 4),
    }


def wp_loso_metrics(oof_parquet) -> dict:
    """Pooled spread-WP metrics from the LOSO out-of-fold parquet.

    Args:
        oof_parquet: parquet with columns ``y`` (win label) and ``wp_pred``.

    Returns:
        Dict: ``n``, ``logloss``, ``brier``, ``auc``.
    """
    import numpy as np
    import polars as pl

    df = pl.read_parquet(oof_parquet)
    y = df["y"].to_numpy().astype(int)
    p = df["wp_pred"].to_numpy()
    return {
        "n": int(len(y)),
        "logloss": round(_bin_logloss(y, p), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "auc": round(_auc(y, p), 4),
    }


def qbr_loso_metrics(oof_parquet) -> dict:
    """Pooled QBR metrics from the LOSO out-of-fold parquet.

    Args:
        oof_parquet: parquet with columns ``y`` (ESPN raw QBR) and ``qbr_pred``.

    Returns:
        Dict: ``n``, ``rmse``, ``mae``, ``r2``, ``corr``.
    """
    import numpy as np
    import polars as pl

    df = pl.read_parquet(oof_parquet)
    y = df["y"].to_numpy()
    pj = df["qbr_pred"].to_numpy()
    ss_res = float(np.sum((y - pj) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "rmse": round(float(np.sqrt(np.mean((pj - y) ** 2))), 4),
        "mae": round(float(np.mean(np.abs(pj - y))), 4),
        "r2": round((1 - ss_res / ss_tot) if ss_tot else float("nan"), 4),
        "corr": round(float(np.corrcoef(pj, y)[0, 1]), 4),
    }


def wp_naive_metrics(model_path, pbp_parquet, wp_oof_parquet) -> dict:
    """In-sample naive-WP calibration + correlation with the spread WP.

    No LOSO OOF is shipped for the naive variant, so this predicts naive WP over
    the full ``pbp_full`` corpus (the naive feature matrix is deterministic from
    game state), pairs it with the spread OOF's win label (same season-sorted row
    order), and reports calibration plus correlation against the spread WP.

    Args:
        model_path: ``wp_naive.ubj``.
        pbp_parquet: ``pbp_full.parquet``.
        wp_oof_parquet: ``loso_wp_oof.parquet`` (supplies ``y`` and ``wp_pred``).

    Returns:
        Dict: ``n``, ``logloss``, ``brier``, ``corr_vs_spread``,
        ``q1_abs_div``, ``q4_abs_div`` — or ``{}`` if inputs/deps are missing.
    """
    from pathlib import Path

    import numpy as np
    import polars as pl

    if not (Path(model_path).exists() and Path(pbp_parquet).exists() and Path(wp_oof_parquet).exists()):
        return {}
    try:
        import xgboost as xgb

        import model_training.constants as C
    except Exception:
        return {}
    df = pl.read_parquet(pbp_parquet)
    oof = pl.read_parquet(wp_oof_parquet)
    seasons = sorted(df["season"].unique().to_list())
    df_ord = pl.concat([df.filter(pl.col("season") == s) for s in seasons])
    if df_ord.height != oof.height:
        return {}
    feats = C.WP_NAIVE_FEATURES
    source = {k: v for k, v in C.WP_SOURCE.items() if k in feats}
    X = df_ord.select([pl.col(src).alias(name) for name, src in source.items()]).to_pandas()[feats]
    b = xgb.Booster()
    b.load_model(str(model_path))
    p = b.predict(xgb.DMatrix(X))
    spread = oof["wp_pred"].to_numpy()
    y = oof["y"].to_numpy().astype(int)
    period_src = C.WP_SOURCE.get("period", "period")
    period = df_ord[period_src].to_numpy() if period_src in df_ord.columns else np.zeros(len(p))
    out = {
        "n": int(len(p)),
        "logloss": round(_bin_logloss(y, p), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "corr_vs_spread": round(float(np.corrcoef(p, spread)[0, 1]), 4),
    }
    for q, key in ((1, "q1_abs_div"), (4, "q4_abs_div")):
        m = period == q
        if m.sum():
            out[key] = round(float(np.mean(np.abs(p[m] - spread[m]))), 4)
    return out


def _weighted_cal_err(pred, event, *, bin_size: float = 0.05) -> float:
    """Single-facet weighted calibration error: bin the predicted prob into
    ``bin_size`` buckets and weight ``|bin - empirical|`` by per-bin ``n``."""
    import numpy as np

    b = np.round(pred / bin_size) * bin_size
    wsum = werr = 0.0
    for bb in np.unique(b):
        mm = b == bb
        n = int(mm.sum())
        wsum += n
        werr += n * abs(float(bb) - float(event[mm].mean()))
    return float(werr / wsum) if wsum else float("nan")


def binary_loso_metrics(oof_parquet, *, pred_col: str, event_col: str) -> dict:
    """Pooled binary-classifier LOSO metrics from an out-of-fold parquet.

    Shared runner for the fg / xpass / two_pt heads — each ships an OOF parquet
    with a predicted-probability column and a 0/1 outcome column. Returns the
    pooled log-loss, Brier, AUC, base rate, and the binned weighted calibration
    error (the same 0.05-bucket recipe the calibration figures use).

    Args:
        oof_parquet: parquet with the prediction + event columns.
        pred_col: predicted-probability column name (e.g. ``fg_pred``).
        event_col: 0/1 outcome column name (e.g. ``made``).

    Returns:
        Dict: ``n``, ``logloss``, ``brier``, ``auc``, ``base_rate``,
        ``weighted_cal_err``.
    """
    import numpy as np
    import polars as pl

    df = pl.read_parquet(oof_parquet)
    y = df[event_col].to_numpy().astype(int)
    p = df[pred_col].to_numpy().astype(float)
    return {
        "n": int(len(y)),
        "logloss": round(_bin_logloss(y, p), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "auc": round(_auc(y, p), 4),
        "base_rate": round(float(y.mean()), 4),
        "weighted_cal_err": round(_weighted_cal_err(p, y), 4),
    }


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
