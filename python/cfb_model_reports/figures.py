"""Figure generation for model reports — derives calibration frames from LOSO OOF
parquets and renders PNGs via the bespoke cfbfastR plotnine styling in
``model_training.figures`` / ``model_training.fourth_down.figures``.

Every builder is defensive: a missing parquet / model / optional dependency
(plotnine) returns ``[]`` so the report still renders (with a note) instead of
crashing the CLI. Only figures that were actually written are returned, as
repo-relative paths (``figures/<name>.png``) for embedding in the Markdown /
Quarto reports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# WP win-prob facet labels by quarter (period column on the OOF source).
_QUARTER_LABEL = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}

# EP 7-class facet labels, in cfbscrapR `next_score_type` order (model class index
# -> human label). Mirrors model_vars `ep_class_to_score_mapping` / the report
# narrative point map {0:+7, 1:-7, 2:+3, 3:-3, 4:+2, 5:-2, 6:0}.
_EP_CLASS_LABEL = {
    0: "Touchdown (+7)",
    1: "Opp Touchdown (-7)",
    2: "Field Goal (+3)",
    3: "Opp Field Goal (-3)",
    4: "Safety (+2)",
    5: "Opp Safety (-2)",
    6: "No Score (0)",
}


def _binned_calibration(pred, event, facet, *, bin_size: float = 0.05):
    """The cfbscrapR / nflfastR probability-calibration recipe, in polars.

    ``bin_pred_prob = round(pred / bin_size) * bin_size``; group by
    ``(facet, bin)``; ``bin_actual_prob = n_event / n_plays``. Returns the frame
    in the ``write_calibration`` contract (``by``, ``bin``, ``n_plays``,
    ``actual``) plus the overall weighted calibration error,
    ``weighted.mean(|bin_pred - bin_actual|, n_plays)`` per facet then
    ``weighted.mean(per_facet, n_event)`` (matching the R ``weighted.mean`` chain).
    """
    df = pl.DataFrame({"pred": pred, "event": event, "by": facet})
    df = df.with_columns(bin=(pl.col("pred") / bin_size).round() * bin_size)
    tab = (
        df.group_by(["by", "bin"])
        .agg(n_plays=pl.len(), n_event=pl.col("event").sum())
        .with_columns(actual=pl.col("n_event") / pl.col("n_plays"))
        .sort(["by", "bin"])
    )
    per = (
        tab.with_columns(diff=(pl.col("bin") - pl.col("actual")).abs())
        .group_by("by")
        .agg(
            wce=(pl.col("diff") * pl.col("n_plays")).sum() / pl.col("n_plays").sum(),
            n_event=pl.col("n_event").sum(),
        )
    )
    nsum = float(per["n_event"].sum())
    cal_err = (
        float((per["wce"] * per["n_event"]).sum() / nsum) if nsum > 0 else float("nan")
    )
    return tab.select(["by", "bin", "n_plays", "actual"]), cal_err


def _rel(png: Path, out: Path) -> str:
    """Return a repo-relative ``figures/<name>.png`` path for embedding."""
    try:
        return png.relative_to(out).as_posix()
    except ValueError:
        return f"figures/{png.name}"


def derive_quarter_aligned(pbp_parquet, oof_parquet):
    """Per-row quarter array aligned to the season-sorted WP OOF row order.

    The WP OOF was built by ``pl.concat`` over ``sorted(seasons)``; reproducing
    that ordering over ``pbp_full`` recovers the per-play period in the same
    order. Returns ``None`` if files/columns/lengths don't line up.
    """
    pp, op = Path(pbp_parquet), Path(oof_parquet)
    if not (pp.exists() and op.exists()):
        return None
    try:
        import model_training.constants as C
    except Exception:
        return None
    df = pl.read_parquet(pp)
    oof = pl.read_parquet(op)
    period_src = C.WP_SOURCE.get("period", "period")
    if "season" not in df.columns or period_src not in df.columns:
        return None
    seasons = sorted(df["season"].unique().to_list())
    df_ord = pl.concat([df.filter(pl.col("season") == s) for s in seasons])
    if df_ord.height != oof.height:
        return None
    return df_ord[period_src].to_numpy()


def build_wp_calibration(oof_parquet, fig_dir: Path, out: Path, *, stem: str,
                         title: str, subtitle: str, by_quarter=None) -> list[str]:
    """Render a WP calibration figure (predicted bin vs observed rate, faceted).

    Args:
        oof_parquet: parquet with columns ``season``, ``y``, ``wp_pred``.
        fig_dir: ``<out>/figures`` directory to write into.
        out: report output root (for repo-relative pathing).
        stem: figure file stem (no extension).
        title / subtitle: plot text.
        by_quarter: optional per-row quarter array (same length as the OOF) used
            to facet the calibration by quarter; falls back to a single panel.

    Returns:
        ``["figures/<stem>.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
        from model_training.validate import calibration_table, weighted_cal_error
    except Exception:
        return []
    df = pl.read_parquet(p)
    pred = df["wp_pred"].to_numpy()
    y = df["y"].to_numpy().astype(int)
    if by_quarter is not None and len(by_quarter) == len(pred):
        by = [_QUARTER_LABEL.get(int(q), "other") for q in by_quarter]
    else:
        by = ["all"] * len(pred)
    tab = calibration_table(pred.tolist(), y.tolist(), by, bin_size=0.05)
    cal_err = round(weighted_cal_error(tab), 4)
    png, _ = write_calibration(tab, fig_dir / stem, title=title,
                               subtitle=subtitle, cal_error=cal_err)
    return [_rel(Path(png), out)]


def build_ep_calibration(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render an EP calibration figure: predicted EP bin vs realized EP.

    EP is a regression-style summary of a multiclass head (sum of class probs x
    point values). We bin the predicted EP and plot mean realized next-score
    value per bin against the bin centre (y=x is perfect calibration). A single
    facet (``by="EP"``) is used.

    Args:
        oof_parquet: parquet with columns ``ep_pred`` (predicted EP) and
            ``realized`` (realized next-score points).

    Returns:
        ``["figures/ep_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    ep_pred = df["ep_pred"].to_numpy()
    realized = df["realized"].to_numpy()
    bin_size = 0.5
    b = np.round(ep_pred / bin_size) * bin_size
    tab = (
        pl.DataFrame({"bin": b, "realized": realized})
        .group_by("bin")
        .agg(n_plays=pl.len(), actual=pl.col("realized").mean())
        .with_columns(by=pl.lit("EP"))
        .sort("bin")
    )
    # weighted calibration MAE in points (matches the loso eval recipe scale)
    cal_err = float(
        (tab["bin"] - tab["actual"]).abs().dot(tab["n_plays"]) / tab["n_plays"].sum()
    )
    png, _ = write_calibration(
        tab.select(["by", "bin", "n_plays", "actual"]),
        fig_dir / "ep_calibration",
        title="Expected Points — LOSO Calibration",
        subtitle="Predicted EP (points) vs Realized next-score value",
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_ep_class_calibration(class_oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render the **7-class faceted** EP calibration figure (cfbscrapR recipe).

    This is the signature ``02-EPA-Model.R`` plot: for each of the 7 next-score
    classes, melt the per-play softprob to ``(class, P(class), event_indicator)``,
    bin ``P(class)`` into 0.05 buckets, and plot binned predicted class
    probability against the empirical rate of that class — faceted by
    ``next_score_type``, point size = ``n_plays``, loess smoother, y=x reference,
    weighted-calibration-error caption.

    Args:
        class_oof_parquet: ``loso_ep_class_oof.parquet`` with columns ``season``,
            ``y`` (actual class 0..6), and ``p0``..``p6`` (softprob per class).

    Returns:
        ``["figures/ep_class_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(class_oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    prob_cols = [f"p{k}" for k in range(7)]
    if not all(c in df.columns for c in prob_cols) or "y" not in df.columns:
        return []
    y = df["y"].to_numpy().astype(int)
    preds, events, facets = [], [], []
    for k in range(7):
        preds.append(df[f"p{k}"].to_numpy())
        events.append((y == k).astype(int))
        facets.append(np.full(len(y), _EP_CLASS_LABEL[k], dtype=object))
    pred = np.concatenate(preds)
    event = np.concatenate(events)
    facet = np.concatenate(facets)
    tab, cal_err = _binned_calibration(pred, event, facet, bin_size=0.05)
    png, _ = write_calibration(
        tab,
        fig_dir / "ep_class_calibration",
        title="Expected Points — LOSO Calibration by Next-Score Type",
        subtitle="Predicted class probability vs Empirical rate (7 next-score classes)",
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_wp_quarter_calibration(period_oof_parquet, fig_dir: Path, out: Path, *,
                                 stem: str, title: str, subtitle: str) -> list[str]:
    """Render the **quarter-faceted** WP calibration figure (cfbscrapR recipe).

    The ``03-WPA-Model.R`` plot: bin predicted WP into 0.05 buckets, group by
    ``(quarter, bin)``, plot binned predicted WP against the empirical win rate —
    faceted by quarter, point size = ``n_plays``, loess smoother, y=x reference,
    weighted-calibration-error caption.

    Args:
        period_oof_parquet: OOF parquet with columns ``y`` (win indicator),
            ``wp_pred`` and ``period`` (quarter 1..4).

    Returns:
        ``["figures/<stem>.png"]`` if written, else ``[]``.
    """
    p = Path(period_oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    if not {"y", "wp_pred", "period"}.issubset(df.columns):
        return []
    pred = df["wp_pred"].to_numpy()
    y = df["y"].to_numpy().astype(int)
    per = df["period"].to_numpy()
    facet = np.array([_QUARTER_LABEL.get(int(q), "OT") for q in per], dtype=object)
    tab, cal_err = _binned_calibration(pred, y, facet, bin_size=0.05)
    png, _ = write_calibration(
        tab, fig_dir / stem, title=title, subtitle=subtitle,
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_qbr_scatter(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render a QBR predicted-vs-actual hex/scatter calibration figure.

    Args:
        oof_parquet: parquet with columns ``y`` (ESPN raw QBR) and ``qbr_pred``.

    Returns:
        ``["figures/qbr_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from plotnine import (
            aes,
            coord_fixed,
            element_rect,
            element_text,
            geom_abline,
            geom_bin2d,
            geom_smooth,
            ggplot,
            labs,
            theme,
            theme_bw,
        )
    except Exception:
        return []
    df = pl.read_parquet(p).to_pandas()
    garnet, grey95, grey99 = "#500f1b", "#f2f2f2", "#fcfcfc"
    font = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]
    smoother = "lm"
    try:
        import skmisc  # noqa: F401

        smoother = "loess"
    except ModuleNotFoundError:
        pass
    gg = (
        ggplot(df, aes("qbr_pred", "y"))
        + geom_bin2d(bins=40)
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_smooth(method=smoother, se=False, color=garnet, size=0.6)
        + coord_fixed(xlim=[0, 100], ylim=[0, 100])
        + labs(
            title="QBR — LOSO Predicted vs ESPN QBR",
            subtitle="Out-of-fold pooled across 2005-2025",
            x="Predicted QBR",
            y="ESPN raw QBR",
        )
        + theme_bw()
        + theme(
            text=element_text(family=font),
            plot_background=element_rect(fill=grey99, color="black"),
            panel_background=element_rect(fill=grey95),
            legend_position="bottom",
        )
    )
    png = fig_dir / "qbr_calibration.png"
    gg.save(str(png), width=6, height=5, dpi=200, verbose=False)
    return [_rel(png, out)]


def build_rb_calibration(loso_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render the xREPA RB-eval calibration figure from the LOSO parquet.

    Args:
        loso_parquet: ``xrepa_loso.parquet`` with columns ``exp_rb_epa`` and
            ``target``.

    Returns:
        ``["figures/rb_eval_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(loso_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
        from rb_eval.validate import calibration_table, weighted_cal_error
    except Exception:
        return []
    cv = pl.read_parquet(p)
    tbl = calibration_table(cv)
    cal_err = round(weighted_cal_error(tbl), 4)
    # remap rb_eval's column names to the write_calibration contract
    frame = tbl.select(
        by=pl.lit("xREPA"),
        bin=pl.col("bin_pred_epa"),
        n_plays=pl.col("total_instances"),
        actual=pl.col("bin_actual_epa"),
    ).sort("bin")
    png, _ = write_calibration(
        frame,
        fig_dir / "rb_eval_calibration",
        title="RB Evaluation (xREPA) — LOSO Calibration",
        subtitle="Predicted EPA/play vs Realized unadjusted EPA",
        cal_error=cal_err,
    )
    return [_rel(Path(png), out)]


def build_fg_calibration(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render the field-goal make-probability calibration figure (single panel).

    Bins predicted make-prob into 0.05 buckets and plots the binned predicted
    probability against the empirical make rate, point size = ``n_plays``, y=x
    reference, weighted-calibration-error caption — the same garnet binned style
    as the WP/EP figures.

    Args:
        oof_parquet: ``loso_fg_oof.parquet`` with columns ``season``,
            ``yards_to_goal``, ``made`` (0/1), ``fg_pred``.

    Returns:
        ``["figures/fg_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    if not {"made", "fg_pred"}.issubset(df.columns):
        return []
    pred = df["fg_pred"].to_numpy()
    made = df["made"].to_numpy().astype(int)
    facet = np.array(["FG"] * len(pred), dtype=object)
    tab, cal_err = _binned_calibration(pred, made, facet, bin_size=0.05)
    png, _ = write_calibration(
        tab,
        fig_dir / "fg_calibration",
        title="Field Goal (make prob) — LOSO Calibration",
        subtitle="Predicted make probability vs Empirical make rate",
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_xpass_calibration(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render the expected-pass calibration figure, **faceted by down**.

    Bins predicted P(pass) into 0.05 buckets, groups by ``(down, bin)``, and
    plots binned predicted P(pass) against the empirical pass rate — faceted by
    down (the xPass analogue of the WP quarter facets), weighted-calibration-error
    caption.

    Args:
        oof_parquet: ``loso_xpass_oof.parquet`` with columns ``season``, ``down``,
            ``is_pass`` (0/1), ``xpass``.

    Returns:
        ``["figures/xpass_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    if not {"down", "is_pass", "xpass"}.issubset(df.columns):
        return []
    pred = df["xpass"].to_numpy()
    y = df["is_pass"].to_numpy().astype(int)
    down = df["down"].to_numpy()
    facet = np.array([f"Down {int(d)}" for d in down], dtype=object)
    tab, cal_err = _binned_calibration(pred, y, facet, bin_size=0.05)
    png, _ = write_calibration(
        tab,
        fig_dir / "xpass_calibration",
        title="Expected Pass — LOSO Calibration by Down",
        subtitle="Predicted P(pass) vs Empirical pass rate (faceted by down)",
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_two_pt_calibration(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render the two-point conversion calibration figure (single panel).

    Bins predicted success-prob into 0.05 buckets and plots binned predicted
    probability against the empirical success rate. The sample is tiny (~1.6K
    attempts) and the model is near-constant (~48%), so the binned scatter is
    sparse by design.

    Args:
        oof_parquet: ``loso_two_pt_oof.parquet`` with columns ``season``,
            ``made`` (0/1), ``two_pt_pred``.

    Returns:
        ``["figures/two_pt_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    if not {"made", "two_pt_pred"}.issubset(df.columns):
        return []
    pred = df["two_pt_pred"].to_numpy()
    made = df["made"].to_numpy().astype(int)
    facet = np.array(["2-PT"] * len(pred), dtype=object)
    tab, cal_err = _binned_calibration(pred, made, facet, bin_size=0.05)
    png, _ = write_calibration(
        tab,
        fig_dir / "two_pt_calibration",
        title="Two-Point Conversion — LOSO Calibration",
        subtitle="Predicted success probability vs Empirical success rate",
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_pregame_wp_calibration(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render the pregame-WP (Five Factors) calibration figure (single panel).

    Bins predicted pregame WP into 0.05 buckets and plots the binned predicted
    probability against the empirical win rate, point size = ``n_plays``, y=x
    reference, weighted-calibration-error caption — the same garnet binned style
    as the FG/two-point figures. The prediction is the Gaussian-transformed
    ``WP = Phi(pred_PtsDiff / std)`` already stored in ``pred_wp``.

    Args:
        oof_parquet: ``loso_pgwp_oof.parquet`` with columns ``season``,
            ``pred_wp`` (predicted pregame WP) and ``win`` (0/1 outcome).

    Returns:
        ``["figures/pregame_wp_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    if not {"pred_wp", "win"}.issubset(df.columns):
        return []
    pred = df["pred_wp"].to_numpy().astype(float)
    win = df["win"].to_numpy().astype(int)
    facet = np.array(["Pregame WP"] * len(pred), dtype=object)
    tab, cal_err = _binned_calibration(pred, win, facet, bin_size=0.05)
    png, _ = write_calibration(
        tab,
        fig_dir / "pregame_wp_calibration",
        title="Pregame WP (Five Factors) — LOSO Calibration",
        subtitle="Predicted pregame WP vs Empirical win rate",
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_fd_figures(model_path, pbp_parquet, fig_dir: Path, out: Path,
                     *, sample_rows: int = 200_000) -> list[str]:
    """Render fourth-down first-down calibration + feature-importance figures.

    Uses ``model_training.fourth_down.validate.calibration_fd`` over a sample of
    ``pbp_full.parquet`` (full 2.2M is unnecessary for a calibration figure).

    Args:
        model_path: ``fd_model.ubj`` path.
        pbp_parquet: ``pbp_full.parquet`` path (must carry the 6 FD features +
            ``statYardage``/yards-gained label).
        sample_rows: cap on plays sampled for the figure.

    Returns:
        ``["figures/fd_calibration.png", "figures/fd_feature_importance.png"]``
        for whichever were written, else ``[]``.
    """
    mp, pp = Path(model_path), Path(pbp_parquet)
    if not mp.exists() or not pp.exists():
        return []
    try:
        import pandas as pd
        import xgboost as xgb

        from model_training.fourth_down.constants import FD_FEATURES, FD_LABEL_OFFSET
        from model_training.fourth_down.features import fd_features
        from model_training.fourth_down.figures import write_fd_figures
        from model_training.fourth_down.validate import calibration_fd
    except Exception:
        return []
    booster = xgb.Booster()
    booster.load_model(str(mp))
    df = pl.read_parquet(pp)
    try:
        X, y_label = fd_features(df)
    except Exception:
        return []
    if len(X) == 0:
        return []
    Xp = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FD_FEATURES)
    # calibration_fd wants raw yards gained; fd_features emits class labels (yards + offset).
    yv = np.asarray(y_label).astype(int) - FD_LABEL_OFFSET
    if len(Xp) > sample_rows:
        rng = np.random.default_rng(20260622)
        idx = rng.choice(len(Xp), size=sample_rows, replace=False)
        Xp = Xp.iloc[idx].reset_index(drop=True)
        yv = yv[idx]
    cal = calibration_fd(booster, Xp, yv, n_bins=10)
    cal_err = float(
        (cal["pred_fd_prob"] - cal["empirical_fd_rate"]).abs().mul(cal["n_plays"]).sum()
        / cal["n_plays"].sum()
    )
    score = booster.get_score(importance_type="gain")
    imp = pd.DataFrame(
        {"Feature": list(score.keys()), "Gain": list(score.values())}
    ).sort_values("Gain", ascending=False)
    cal_png, imp_png = write_fd_figures(cal, imp, fig_dir, cal_error=cal_err)
    return [_rel(Path(cal_png), out), _rel(Path(imp_png), out)]
