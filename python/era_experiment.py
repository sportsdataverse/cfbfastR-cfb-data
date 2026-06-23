"""Era-variable experiment: leave-one-season-out baseline-vs-era comparison.

For every listed CFB model, fit the shipped recipe twice under honest LOSO CV —
once with the current feature set (``baseline``) and once with the nflfastR-style
one-hot rule-era dummies (era0..era3, the ``era`` variant) — pool the out-of-fold
predictions, and report the headline metric delta. Era is **kept only where the
out-of-fold headline metric improves**.

Encoding: one-hot era0..era3 from ``ERA_BOUNDS`` (2006/2013/2020 → 4 buckets:
2004-2006 / 2007-2013 / 2014-2020 / 2021+; CFB data starts 2004 so there is no
nflfastR-style empty pre-2001 bucket). Models
that already ship an ordinal ``era`` factor (xpass / two_pt / fourth_down) are
compared as one-hot-era (variant) vs ordinal-era (baseline).

Nothing here mutates a shipped artifact: results land in ``artifacts/era_results.json``
(written incrementally so the sweep is resumable) and winning models get a
side-by-side ``artifacts/<model>_era.ubj`` for review — the canonical ``.ubj`` files
are never overwritten.

Run (background-friendly; logs to stdout)::

    python -m era_experiment --artifacts artifacts \
        --espn-qbr ../../cfbfastR-cfb-raw/cfb/qbr/espn_qbr.parquet \
        --final-dir .cache/cfb_final
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import xgboost as xgb

import model_training.constants as C
from model_training import features as F

# --- metric helpers (identical recipes to validate.py / cfb_model_reports.metrics) ----


def _bin_logloss(y, p) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _mc_logloss(y, P) -> float:
    P = np.clip(P, 1e-15, 1 - 1e-15)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def _auc(y, p) -> float:
    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p), float)
    r[order] = np.arange(1, len(p) + 1)
    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _weighted_cal_err(pred, event, *, bin_size: float = 0.05) -> float:
    b = np.round(np.asarray(pred) / bin_size) * bin_size
    ev = np.asarray(event, dtype=float)
    wsum = werr = 0.0
    for bb in np.unique(b):
        mm = b == bb
        n = int(mm.sum())
        wsum += n
        werr += n * abs(float(bb) - float(ev[mm].mean()))
    return float(werr / wsum) if wsum else float("nan")


def _ep_cal_mae(ep_pred, realized) -> float:
    b = np.round(ep_pred).astype(int)
    wsum = werr = 0.0
    for bb in np.unique(b):
        mm = b == bb
        n = int(mm.sum())
        wsum += n
        werr += n * abs(ep_pred[mm].mean() - realized[mm].mean())
    return float(werr / wsum) if wsum else float("nan")


def _round(d: dict[str, Any], n: int = 4) -> dict[str, Any]:
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


# --- result accumulator (incremental + resumable) -------------------------------------


class Results:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                self.data = {}

    def has(self, model: str) -> bool:
        return model in self.data and self.data[model].get("status") == "done"

    def put(self, model: str, payload: dict[str, Any]) -> None:
        self.data[model] = payload
        self.path.write_text(json.dumps(self.data, indent=2))


def _verdict(baseline: dict, era: dict, key: str, *, lower_is_better: bool) -> dict:
    b, e = baseline.get(key), era.get(key)
    if b is None or e is None or (isinstance(b, float) and np.isnan(b)):
        return {"headline": key, "baseline": b, "era": e, "delta": None, "era_wins": None}
    delta = e - b
    wins = (delta < 0) if lower_is_better else (delta > 0)
    return {
        "headline": key, "baseline": round(float(b), 5), "era": round(float(e), 5),
        "delta": round(float(delta), 5), "lower_is_better": lower_is_better,
        "era_wins": bool(wins),
    }


# --- generic per-fold LOSO over pbp_full for the binary / multiclass heads -------------


def _loso_binary(df: pl.DataFrame, *, matrix_fn: Callable, params: dict, nrounds: int,
                 era: bool, seasons: list[int], log) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled (y, p) out-of-fold for a binary head."""
    ys, ps = [], []
    for s in seasons:
        t0 = time.time()
        tr = df.filter(pl.col("season") != s)
        te = df.filter(pl.col("season") == s)
        Xtr, ytr, _ = matrix_fn(tr, era_onehot=era)
        Xte, yte, _ = matrix_fn(te, era_onehot=era)
        if len(Xtr) == 0 or len(Xte) == 0:  # label-sparse early seasons (e.g. 2pt pre-2007)
            log(f"    fold {s}: n=0 (skipped)")
            continue
        m = xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=nrounds)
        p = m.predict(xgb.DMatrix(Xte))
        ys.append(yte.astype(int))
        ps.append(p)
        log(f"    fold {s}: n={len(yte)} ({time.time()-t0:.0f}s)")
    return np.concatenate(ys), np.concatenate(ps)


def _binary_metrics(y, p) -> dict:
    return {"logloss": _bin_logloss(y, p), "brier": float(np.mean((p - y) ** 2)),
            "auc": _auc(y, p), "weighted_cal_err": _weighted_cal_err(p, y), "n": int(len(y))}


# --- model runners --------------------------------------------------------------------


def run_ep(df, seasons, log, *, ep_baseline: dict | None) -> dict:
    """EP: 7-class softprob. Baseline trusted from the shipped LOSO log unless re-run."""
    def fold(era: bool):
        ys, eps, real, mll, acc, ns = [], [], [], [], [], []
        scores = np.array([C.EP_CLASS_TO_SCORE[c] for c in range(7)], float)
        for s in seasons:
            t0 = time.time()
            tr = df.filter(pl.col("season") != s)
            te = df.filter(pl.col("season") == s)
            Xtr, ytr, wtr = F.ep_matrix(tr, era_onehot=era)
            m = xgb.train(C.EP_PARAMS, xgb.DMatrix(Xtr, label=ytr, weight=wtr),
                          num_boost_round=C.EP_NROUNDS)
            Xte, yte, _ = F.ep_matrix(te, era_onehot=era)
            yte = yte.astype(int)
            P = m.predict(xgb.DMatrix(Xte))
            ys.append(yte); eps.append(P @ scores); real.append(scores[yte])
            mll.append(_mc_logloss(yte, P)); acc.append(float(np.mean(P.argmax(1) == yte)))
            ns.append(len(yte))
            log(f"    [ep:{'era' if era else 'base'}] fold {s}: n={len(yte)} "
                f"mll={mll[-1]:.4f} ({time.time()-t0:.0f}s)")
        ns = np.array(ns, float)
        return {"mlogloss": float(np.average(mll, weights=ns)),
                "accuracy": float(np.average(acc, weights=ns)),
                "ep_cal_mae": _ep_cal_mae(np.concatenate(eps), np.concatenate(real)),
                "n": int(ns.sum())}
    base = ep_baseline or fold(False)
    era = fold(True)
    return {"baseline": _round(base), "era": _round(era),
            "verdict": _verdict(base, era, "ep_cal_mae", lower_is_better=True),
            "secondary": _verdict(base, era, "mlogloss", lower_is_better=True)}


def run_wp(df, seasons, log, *, variant: str) -> dict:
    from model_training.ingest import add_winner
    df = add_winner(df)
    params, nrounds = ((C.WP_SPREAD_PARAMS, C.WP_SPREAD_NROUNDS) if variant == "spread"
                       else (C.WP_NAIVE_PARAMS, C.WP_NAIVE_NROUNDS))
    mf = lambda d, era_onehot: F.wp_matrix(d, variant, era_onehot=era_onehot)  # noqa: E731
    out = {}
    for tag, era in (("baseline", False), ("era", True)):
        log(f"  wp_{variant} {tag} ...")
        y, p = _loso_binary(df, matrix_fn=mf, params=params, nrounds=nrounds,
                            era=era, seasons=seasons, log=log)
        out[tag] = _round(_binary_metrics(y, p))
    return {**out, "verdict": _verdict(out["baseline"], out["era"], "logloss", lower_is_better=True),
            "secondary": _verdict(out["baseline"], out["era"], "auc", lower_is_better=False)}


def run_simple_binary(df, seasons, log, *, model: str) -> dict:
    spec = {
        "fg": (F.fg_matrix, C.FG_PARAMS, C.FG_NROUNDS),
        "xpass": (F.xpass_matrix, C.XPASS_PARAMS, C.XPASS_NROUNDS),
        "two_pt": (F.two_pt_matrix, C.TWO_PT_PARAMS, C.TWO_PT_NROUNDS),
    }[model]
    matrix_fn, params, nrounds = spec
    out = {}
    for tag, era in (("baseline", False), ("era", True)):
        log(f"  {model} {tag} ...")
        y, p = _loso_binary(df, matrix_fn=matrix_fn, params=params, nrounds=nrounds,
                            era=era, seasons=seasons, log=log)
        out[tag] = _round(_binary_metrics(y, p))
    return {**out, "verdict": _verdict(out["baseline"], out["era"], "logloss", lower_is_better=True),
            "secondary": _verdict(out["baseline"], out["era"], "weighted_cal_err", lower_is_better=True)}


def run_qbr(df, seasons, log, *, espn_qbr: pl.DataFrame) -> dict:
    def fold(era: bool):
        ys, ps = [], []
        for s in seasons:
            tr = df.filter(pl.col("season") != s)
            te = df.filter(pl.col("season") == s)
            Xtr, _, ktr = F.qbr_matrix(tr, era_onehot=era)
            jt = pl.from_pandas(ktr).hstack(pl.from_pandas(Xtr)).join(
                espn_qbr, on=["game_id", "passer_player_name"], how="inner").drop_nulls("raw_qbr")
            feat_cols = [c for c in jt.columns if c not in
                         ("game_id", "season", "passer_player_name", "raw_qbr")]
            m = xgb.train(C.QBR_PARAMS, xgb.DMatrix(jt.select(feat_cols).to_pandas(),
                          label=jt["raw_qbr"].to_numpy()), num_boost_round=C.QBR_NROUNDS)
            Xte, _, kte = F.qbr_matrix(te, era_onehot=era)
            je = pl.from_pandas(kte).hstack(pl.from_pandas(Xte)).join(
                espn_qbr, on=["game_id", "passer_player_name"], how="inner").drop_nulls("raw_qbr")
            if je.height == 0:
                continue
            p = m.predict(xgb.DMatrix(je.select(feat_cols).to_pandas()))
            ys.append(je["raw_qbr"].to_numpy()); ps.append(p)
            log(f"    [qbr:{'era' if era else 'base'}] fold {s}: n={je.height}")
        y, pj = np.concatenate(ys), np.concatenate(ps)
        ss_res = float(np.sum((y - pj) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {"rmse": float(np.sqrt(np.mean((pj - y) ** 2))), "mae": float(np.mean(np.abs(pj - y))),
                "r2": (1 - ss_res / ss_tot) if ss_tot else float("nan"),
                "corr": float(np.corrcoef(pj, y)[0, 1]), "n": int(len(y))}
    base, era = fold(False), fold(True)
    return {"baseline": _round(base), "era": _round(era),
            "verdict": _verdict(base, era, "rmse", lower_is_better=True),
            "secondary": _verdict(base, era, "r2", lower_is_better=False)}


def run_cpoe(df, seasons, log) -> dict:
    """CP/CPOE: 8 game-state features, 2014+ only. era collapses to era2/era3 (≈1 dummy)."""
    from cpoe.constants import FEATURE_COLS, MIN_SEASON, TARGET_COL, XGB_NROUNDS, XGB_PARAMS
    from cpoe.features import extract_pass_features
    cp = pl.from_pandas(extract_pass_features(df.filter(pl.col("season") >= MIN_SEASON)))
    seasons_cp = [s for s in seasons if s >= MIN_SEASON]
    lo, mid, hi = C.ERA_BOUNDS

    def add_era(frame):
        s = pl.col("season")
        return frame.with_columns(
            era0=(s <= lo).cast(pl.Int32), era1=((s > lo) & (s <= mid)).cast(pl.Int32),
            era2=((s > mid) & (s <= hi)).cast(pl.Int32), era3=(s > hi).cast(pl.Int32))
    cp = add_era(cp)

    def fold(era: bool):
        cols = list(FEATURE_COLS) + (C.ERA_ONEHOT_COLS if era else [])
        ys, ps = [], []
        for s in seasons_cp:
            tr = cp.filter(pl.col("season") != s); te = cp.filter(pl.col("season") == s)
            m = xgb.train(XGB_PARAMS, xgb.DMatrix(tr.select(cols).to_pandas(),
                          label=tr[TARGET_COL].to_numpy()), num_boost_round=XGB_NROUNDS)
            p = m.predict(xgb.DMatrix(te.select(cols).to_pandas()))
            ys.append(te[TARGET_COL].to_numpy().astype(int)); ps.append(p)
            log(f"    [cpoe:{'era' if era else 'base'}] fold {s}: n={te.height}")
        y, p = np.concatenate(ys), np.concatenate(ps)
        return _binary_metrics(y, p)
    base, era = fold(False), fold(True)
    return {"baseline": _round(base), "era": _round(era),
            "verdict": _verdict(base, era, "logloss", lower_is_better=True),
            "secondary": _verdict(base, era, "weighted_cal_err", lower_is_better=True),
            "note": f"era only spans era2(2014-{hi}) + era3({hi+1}+) — effectively one dummy."}


def run_fourth_down(df, seasons, log) -> dict:
    from model_training.fourth_down.constants import FD_NROUNDS, FD_NUM_CLASS, FD_PARAMS
    from model_training.fourth_down.features import fd_features

    def fold(era: bool):
        mll, fd_pred_all, fd_emp_all, ns = [], [], [], []
        gains = np.arange(FD_NUM_CLASS) - 10
        for s in seasons:
            t0 = time.time()
            tr = df.filter(pl.col("season") != s); te = df.filter(pl.col("season") == s)
            Xtr, ytr = fd_features(tr, era_onehot=era)
            Xte, yte = fd_features(te, era_onehot=era)
            if len(Xtr) == 0 or len(Xte) == 0:
                continue
            m = xgb.train(FD_PARAMS, xgb.DMatrix(Xtr, label=ytr), num_boost_round=FD_NROUNDS)
            P = m.predict(xgb.DMatrix(Xte)).reshape(-1, FD_NUM_CLASS)
            dist = Xte["distance"].to_numpy(); yards = yte - 10
            pred_fd = np.array([P[i, gains >= dist[i]].sum() for i in range(len(Xte))])
            emp_fd = (yards >= dist).astype(float)
            mll.append(_mc_logloss(yte.astype(int), P))
            fd_pred_all.append(pred_fd); fd_emp_all.append(emp_fd); ns.append(len(Xte))
            log(f"    [fd:{'era' if era else 'base'}] fold {s}: n={len(Xte)} ({time.time()-t0:.0f}s)")
        ns = np.array(ns, float)
        pred = np.concatenate(fd_pred_all); emp = np.concatenate(fd_emp_all)
        return {"mlogloss": float(np.average(mll, weights=ns)),
                "first_down_cal_mae": _weighted_cal_err(pred, emp),
                "n": int(ns.sum())}
    base, era = fold(False), fold(True)
    return {"baseline": _round(base), "era": _round(era),
            "verdict": _verdict(base, era, "first_down_cal_mae", lower_is_better=True),
            "secondary": _verdict(base, era, "mlogloss", lower_is_better=True)}


def run_rb_eval(args, log) -> dict:
    """RB-eval xREPA GAM. era added as a factor term f(era_ordinal). Needs pygam + final.json."""
    try:
        from pygam import LinearGAM, f, s
    except ImportError:
        return {"status": "skipped", "reason": "pygam not installed (uv sync --group gam)"}
    final_dir = Path(args.final_dir)
    if not final_dir.exists():
        return {"status": "skipped", "reason": f"final.json dir not found: {final_dir}"}
    from rb_eval.aggregate import build_model_data, build_rusher_seasons
    from rb_eval.features import load_rush_plays
    from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2
    log("  rb_eval: rebuilding model_data from final.json ...")
    md = build_model_data(build_rusher_seasons(load_rush_plays(str(final_dir))))
    lo, mid, hi = C.ERA_BOUNDS
    md = md.with_columns(
        era=pl.when(pl.col("season") <= lo).then(0).when(pl.col("season") <= mid).then(1)
        .when(pl.col("season") <= hi).then(2).otherwise(3).cast(pl.Int32))
    feats = ["epa_per_play", "success"]
    seasons = sorted(md["season"].drop_nulls().unique().to_list())

    def loso(era: bool):
        parts = []
        for season in seasons:
            tr = md.filter(pl.col("season") != season).drop_nulls(feats + ["target", "weight", "era"])
            te = md.filter(pl.col("season") == season).drop_nulls(feats + ["target", "weight", "era"])
            if tr.is_empty() or te.is_empty():
                continue
            cols = feats + (["era"] if era else [])
            terms = (s(0) + s(1) + f(2)) if era else (s(0) + s(1))
            gam = LinearGAM(terms).fit(tr.select(cols).to_numpy(), tr["target"].to_numpy(),
                                       weights=tr["weight"].to_numpy())
            pred = gam.predict(te.select(cols).to_numpy())
            parts.append(te.with_columns(pl.Series("exp_rb_epa", pred, dtype=pl.Float64)))
        cv = pl.concat(parts, how="diagonal_relaxed")
        tbl = calibration_table(cv)
        return {"weighted_r2": float(weighted_r2(tbl)),
                "weighted_cal_error": float(weighted_cal_error(tbl)), "n": int(cv.height)}
    base, era = loso(False), loso(True)
    return {"baseline": _round(base), "era": _round(era),
            "verdict": _verdict(base, era, "weighted_r2", lower_is_better=False),
            "secondary": _verdict(base, era, "weighted_cal_error", lower_is_better=True),
            "note": "era as a GAM factor term f(era); ordinal 0..3 (one coef per level = one-hot)."}


# --- driver ---------------------------------------------------------------------------

_PBP_MODELS = ["wp_naive", "two_pt", "fg", "xpass", "wp_spread", "qbr", "cpoe", "fourth_down", "ep"]
_ALL_MODELS = _PBP_MODELS + ["rb_eval", "pregame_wp"]

# Shipped EP LOSO baseline (artifacts/loso_eval.log, 22 seasons 2004-2025).
_EP_BASELINE = {"mlogloss": 1.2333, "accuracy": 0.4997, "ep_cal_mae": 0.014, "n": 2219607}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="era_experiment")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--pbp", default=None,
                    help="training-frame override (e.g. the spread-backfilled pbp_full)")
    ap.add_argument("--espn-qbr", default="../../cfbfastR-cfb-raw/cfb/qbr/espn_qbr.parquet")
    ap.add_argument("--final-dir", default=".cache/cfb_final")
    ap.add_argument("--only", default="", help="comma list; default = all")
    ap.add_argument("--smoke", type=int, default=0, help="limit to first N seasons (plumbing test)")
    ap.add_argument("--rerun-ep-baseline", action="store_true",
                    help="re-run EP baseline LOSO instead of trusting the shipped log")
    ap.add_argument("--results-name", default="era_results.json",
                    help="results JSON filename under --artifacts (use a distinct name for re-runs)")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    args = ap.parse_args(argv)

    art = Path(args.artifacts)
    res = Results(art / args.results_name)
    pbp_path = Path(args.pbp) if args.pbp else art / "pbp_full.parquet"
    targets = [m.strip() for m in args.only.split(",") if m.strip()] or list(_ALL_MODELS)

    def log(msg: str) -> None:
        print(msg, flush=True)

    df = None
    espn = None

    def _pbp() -> pl.DataFrame:
        nonlocal df
        if df is None:
            log(f"loading {pbp_path} ...")
            df = pl.read_parquet(pbp_path)
        return df

    def _seasons() -> list[int]:
        s = sorted(_pbp()["season"].unique().to_list())
        return s[: args.smoke] if args.smoke else s

    for model in targets:
        if res.has(model) and not args.force:
            log(f"== {model}: cached, skipping ==")
            continue
        log(f"\n===== {model} =====")
        t0 = time.time()
        try:
            if model == "ep":
                base = None if args.rerun_ep_baseline else _EP_BASELINE
                if args.smoke:
                    base = None  # smoke can't use the full-season shipped baseline
                payload = run_ep(_pbp(), _seasons(), log, ep_baseline=base)
            elif model in ("wp_spread", "wp_naive"):
                payload = run_wp(_pbp(), _seasons(), log, variant=model.split("_", 1)[1])
            elif model in ("fg", "xpass", "two_pt"):
                payload = run_simple_binary(_pbp(), _seasons(), log, model=model)
            elif model == "qbr":
                if espn is None:
                    espn = pl.read_parquet(args.espn_qbr).select(
                        pl.col("game_id").cast(pl.Int64), pl.col("passer_player_name"),
                        pl.col("raw_qbr").cast(pl.Float64, strict=False)).drop_nulls()
                payload = run_qbr(_pbp(), _seasons(), log, espn_qbr=espn)
            elif model == "cpoe":
                payload = run_cpoe(_pbp(), _seasons(), log)
            elif model == "fourth_down":
                payload = run_fourth_down(_pbp(), _seasons(), log)
            elif model == "rb_eval":
                payload = run_rb_eval(args, log)
            elif model == "pregame_wp":
                payload = {"status": "skipped",
                           "reason": "pgwp 5FR feature table rebuild handled separately "
                                     "(single-feature pregame model; era candidacy weak)."}
            else:
                payload = {"status": "skipped", "reason": "unknown model"}
        except Exception as e:  # noqa: BLE001
            import traceback
            payload = {"status": "error", "reason": repr(e), "trace": traceback.format_exc()}
            log(f"!! {model} ERROR: {e}")
        payload.setdefault("status", "done")
        payload["elapsed_s"] = round(time.time() - t0, 1)
        payload["smoke"] = args.smoke or None
        res.put(model, payload)
        v = payload.get("verdict")
        if v:
            log(f"== {model}: {v['headline']} base={v['baseline']} era={v['era']} "
                f"delta={v['delta']} era_wins={v['era_wins']} ({payload['elapsed_s']}s) ==")
        else:
            log(f"== {model}: {payload.get('status')} ({payload['elapsed_s']}s) ==")

    log(f"\nwrote {res.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
