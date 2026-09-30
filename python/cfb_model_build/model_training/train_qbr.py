"""xQBR: XGBoost regression of ESPN raw QBR on the SERVED per-QB-game features, gated.

Pre-registration: ``models/qbr/PREREG_xqbr_retrain.md``. The shape of it:

* **Features are the served rows.** The trainer reads the published ``adv_passing``
  box score, which is sdv-py's ``advBoxScore.pass`` block, i.e. the same rows GOP scores.
  The previous trainer rebuilt the features from the EP/WP frame by passer name, so QB
  runs never reached ``rush_epa``, overtime games were missing, and penalty plays were
  handled differently from serving. Re-implementing the served aggregation here would
  just grow a second copy that drifts; reading the served output cannot drift.
* **Labels** are the committed ESPN labels (``qbr_labels.LABELS_PATH``), joined on
  ``(game_id, athlete_id)``.
* **Gate.** Every arm is scored against the incumbent on the frozen holdout, paired,
  with a game-clustered bootstrap CI. An arm that is not strictly better does not ship.
  ``check_gate`` is what the publisher imports, so an artifact whose sha256 has no
  passing record cannot be uploaded.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from . import constants as C

SERVED = [
    "qbr_epa",
    "sack_epa",
    "pass_epa",
    "rush_epa",
    "pen_epa",
    "spread",
    "era0",
    "era1",
    "era2",
    "era3",
]
ARMS = {"spread": SERVED, "no_spread": [f for f in SERVED if f != "spread"]}
TRAIN_SEASONS = tuple(range(2004, 2026))
# Frozen 2026-09-30 (PREREG section 2). Changing any of these is a new pre-registration.
HOLDOUT = {"season": 2026, "season_type": 2, "weeks": (1, 2, 3, 4)}
MIN_HOLDOUT = 500  # 549 matched at registration
MIN_MATCH_RATE = 0.95  # 0.998 at registration
SPREAD_COST_TOLERANCE = (
    0.30  # RMSE points no_spread may give up to spread (PREREG section 5)
)
N_BOOT, SEED = 2000, 0
GATE_EXIT = 3  # train-qbr's exit code when no arm passes: the incumbent stays


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def served_features(data_dir, seasons) -> pl.DataFrame:
    """Published ``adv_passing`` rows for ``seasons`` + the passer's ESPN athlete id.

    The id is the per-(game, team, name) mode of pbp ``passer_player_id``; a row whose id
    repeats within its game (two name spellings, one athlete) is dropped as ambiguous.
    """
    key = ["game_id", "pos_team_id", "passer_player_name"]
    out = []
    for s in seasons:
        adv = pl.read_parquet(
            Path(data_dir) / "adv_passing" / "parquet" / f"adv_passing_{s}.parquet",
            columns=[*key, "season", *SERVED, "exp_qbr"],
        )
        ids = (
            pl.read_parquet(
                Path(data_dir) / "pbp" / "parquet" / f"play_by_play_{s}.parquet",
                columns=[*key, "passer_player_id"],
            )
            .drop_nulls(["passer_player_name", "passer_player_id"])
            .group_by(key)
            .agg(
                athlete_id=pl.col("passer_player_id")
                .mode()
                .sort()
                .first()
                .cast(pl.Int64, strict=True)
            )
        )
        for k in key:
            assert adv.schema[k] == ids.schema[k], (s, k, adv.schema[k], ids.schema[k])
        out.append(adv.join(ids, on=key, how="left"))
    df = pl.concat(out, how="vertical_relaxed").drop_nulls("athlete_id")
    return df.filter(~pl.struct("game_id", "athlete_id").is_duplicated())


def labelled(features: pl.DataFrame, labels: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Inner-join served features to ESPN labels; tag each row train / holdout / unused."""
    lab = labels.select(
        "game_id", "athlete_id", "season_type", "week", "QBR", "TQBR"
    ).drop_nulls("QBR")
    for k in ("game_id", "athlete_id"):
        assert features.schema[k] == lab.schema[k] == pl.Int64, (
            k,
            features.schema[k],
            lab.schema[k],
        )
    j = features.join(lab, on=["game_id", "athlete_id"], how="inner")
    in_h = (
        (pl.col("season") == HOLDOUT["season"])
        & (pl.col("season_type") == HOLDOUT["season_type"])
        & pl.col("week").is_in(HOLDOUT["weeks"])
    )
    j = j.with_columns(
        split=pl.when(in_h)
        .then(pl.lit("holdout"))
        .when(pl.col("season").is_in(TRAIN_SEASONS))
        .then(pl.lit("train"))
        .otherwise(pl.lit("unused"))
    ).sort("game_id", "athlete_id")
    # ESPN's holdout labels are counted from the label file, not the joined frame, so an id
    # that fails to join still counts against the match rate.
    n_h_labels = labels.filter(
        (pl.col("season") == HOLDOUT["season"])
        & (pl.col("season_type") == HOLDOUT["season_type"])
        & pl.col("week").is_in(HOLDOUT["weeks"])
        & pl.col("QBR").is_not_null()
    ).height
    n_h = j.filter(pl.col("split") == "holdout").height
    return j, {
        "holdout_labels": n_h_labels,
        "holdout_matched": n_h,
        "match_rate": n_h / max(n_h_labels, 1),
    }


def fit(frame: pl.DataFrame, features: list[str]) -> xgb.Booster:
    X = frame.select(features).to_pandas()
    return xgb.train(
        C.QBR_PARAMS,
        xgb.DMatrix(X, label=frame["QBR"].to_numpy()),
        num_boost_round=C.QBR_NROUNDS,
    )


def predict(model: xgb.Booster, frame: pl.DataFrame) -> np.ndarray:
    """Score with the model's own feature list, as serving does."""
    return model.predict(xgb.DMatrix(frame.select(model.feature_names).to_pandas()))


def paired_gate(
    y, pred, pred_incumbent, clusters, *, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Δ = mean(SE_arm − SE_incumbent) with a percentile bootstrap resampling whole games."""
    y, pred, inc = (np.asarray(a, dtype=float) for a in (y, pred, pred_incumbent))
    d = (pred - y) ** 2 - (inc - y) ** 2
    _, g = np.unique(np.asarray(clusters), return_inverse=True)
    sums, counts = np.bincount(g, weights=d), np.bincount(g)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "delta_mse": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": len(d),
        "n_games": len(sums),
    }


def _fit_stats(y, p, total) -> dict:
    y, p, total = (np.asarray(a, dtype=float) for a in (y, p, total))
    ok = ~np.isnan(total)
    return {
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "r": float(np.corrcoef(p, y)[0, 1]),
        "bias": float(np.mean(p - y)),
        "r_total_qbr": float(np.corrcoef(p[ok], total[ok])[0, 1]),
    }


def choose(results: dict) -> str | None:
    """PREREG section 5: prefer no_spread when it passes and costs <= the tolerance."""
    ns, sp = results["no_spread"], results["spread"]
    if ns["passed"] and (
        not sp["passed"] or ns["rmse"] - sp["rmse"] <= SPREAD_COST_TOLERANCE
    ):
        return "no_spread"
    if sp["passed"]:
        return "spread"
    return None


def evaluate(
    frame: pl.DataFrame, join_stats: dict, incumbent: xgb.Booster
) -> tuple[dict, dict]:
    """Fit both arms on train, score all three on the holdout, apply the gate + choice rule."""
    train, hold = (
        frame.filter(pl.col("split") == "train"),
        frame.filter(pl.col("split") == "holdout"),
    )
    assert set(train["season"].unique()).isdisjoint(set(hold["season"].unique())), (
        "train/holdout share a season"
    )
    y, tot, games = (
        hold["QBR"].to_numpy(),
        hold["TQBR"].to_numpy(),
        hold["game_id"].to_numpy(),
    )
    p_inc = predict(incumbent, hold)
    sized = hold.height >= MIN_HOLDOUT and join_stats["match_rate"] >= MIN_MATCH_RATE
    # When the published exp_qbr was scored by this incumbent, a ~0 gap proves the frame IS
    # the served input. After a model swap it measures the swap instead: reported, not asserted.
    gap = np.abs(p_inc - hold["exp_qbr"].to_numpy().astype(float))
    results = {
        "incumbent": {
            **_fit_stats(y, p_inc, tot),
            "max_abs_gap_to_published_exp_qbr": float(np.nanmax(gap)),
        }
    }
    models = {}
    for arm, feats in ARMS.items():
        models[arm] = fit(train, feats)
        p = predict(models[arm], hold)
        g = paired_gate(y, p, p_inc, games)
        results[arm] = {
            **_fit_stats(y, p, tot),
            **g,
            "passed": bool(sized and g["ci_hi"] < 0),
        }
    return results, models


def loso_oof(train: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Leave-one-season-out predictions on the training seasons (report input)."""
    out = []
    for s in sorted(train["season"].unique()):
        m = fit(train.filter(pl.col("season") != s), features)
        te = train.filter(pl.col("season") == s)
        out.append(
            te.select("season", pl.col("QBR").alias("y")).with_columns(
                qbr_pred=pl.Series(predict(m, te))
            )
        )
    return pl.concat(out)


def check_gate(model_path) -> dict:
    """Refuse a QBR model without a passing gate record for exactly these bytes.

    The record is ``<stem>.gate.json`` beside the model. Imported by the artifact
    publisher -- a gate re-implemented at the call site is how a doc once passed a frame
    the publisher refused.
    """
    model_path = Path(model_path)
    rec_path = model_path.with_suffix(".gate.json")
    if not rec_path.exists():
        raise RuntimeError(
            f"{model_path.name}: no gate record {rec_path.name}; refusing to publish"
        )
    rec = json.loads(rec_path.read_text())
    if rec.get("passed") is not True:
        raise RuntimeError(
            f"{model_path.name}: gate record says passed={rec.get('passed')!r}; refusing"
        )
    if rec.get("candidate_sha256") != sha256(model_path):
        raise RuntimeError(
            f"{model_path.name}: sha256 differs from the gated candidate; refusing"
        )
    return rec


def train_qbr(data_dir, labels_path, incumbent_path, out) -> int:
    """Fit, gate, choose, and write the chosen arm's artifacts to ``out`` (a ``.ubj`` path).

    Always writes ``<stem>.gate.json``. The model, its card, the partition and the LOSO
    frame are written only when an arm passes; otherwise returns ``GATE_EXIT``.
    """
    from .model_card import write_xgb_model_card
    from .qbr_labels import load_labels

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # a previous run's outputs must not survive beside this run's record
    for stale in (out, out.with_suffix(".json"), out.parent / "qbr_partition.parquet", out.parent / "loso_qbr_oof.parquet"):
        stale.unlink(missing_ok=True)
    labels = load_labels(labels_path)
    feats = served_features(data_dir, [*TRAIN_SEASONS, HOLDOUT["season"]])
    frame, join_stats = labelled(feats, labels)
    incumbent = xgb.Booster()
    incumbent.load_model(str(incumbent_path))
    results, models = evaluate(frame, join_stats, incumbent)
    chosen = choose(results)
    record = {
        "prereg": "models/qbr/PREREG_xqbr_retrain.md",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "holdout": {**HOLDOUT, "weeks": list(HOLDOUT["weeks"]), **join_stats},
        "train_seasons": [TRAIN_SEASONS[0], TRAIN_SEASONS[-1]],
        "n_train": frame.filter(pl.col("split") == "train").height,
        "gate": f"95% game-clustered bootstrap upper bound of mean(SE_arm - SE_incumbent) < 0 (B={N_BOOT}, seed={SEED})",
        "spread_cost_tolerance_rmse": SPREAD_COST_TOLERANCE,
        "labels_sha256": sha256(labels_path),
        # the exact served feature files; pbp only supplies the id crosswalk
        "adv_passing_sha256": {
            s: sha256(
                Path(data_dir) / "adv_passing" / "parquet" / f"adv_passing_{s}.parquet"
            )
            for s in [*TRAIN_SEASONS, HOLDOUT["season"]]
        },
        "incumbent_sha256": sha256(incumbent_path),
        "arms": results,
        "chosen": chosen,
        "passed": chosen is not None,
        "candidate_sha256": None,
    }
    if chosen is not None:
        model = models[chosen]
        model.save_model(str(out))
        record["candidate_sha256"] = sha256(out)
        train = frame.filter(pl.col("split") == "train")
        write_xgb_model_card(
            out,
            model_type="qbr",
            label="espn_raw_qbr",
            model=model,
            hyperparams=C.QBR_PARAMS,
            n_rows=train.height,
            seasons=TRAIN_SEASONS,
            source="adv_passing (served box score) + models/qbr/espn_qbr_labels.parquet",
            extra={
                "num_boost_round": C.QBR_NROUNDS,
                "arm": chosen,
                "gate_record": out.with_suffix(".gate.json").name,
            },
        )
        frame.filter(pl.col("split") != "unused").select(
            "game_id", "athlete_id", "season", "week", "split"
        ).write_parquet(out.parent / "qbr_partition.parquet")
        loso_oof(train, ARMS[chosen]).write_parquet(out.parent / "loso_qbr_oof.parquet")
    out.with_suffix(".gate.json").write_text(json.dumps(record, indent=2) + "\n")
    for arm, r in results.items():
        extra = (
            f" delta_mse {r['delta_mse']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] passed={r['passed']}"
            if arm != "incumbent"
            else ""
        )
        print(
            f"[qbr] {arm:10s} rmse {r['rmse']:.3f} mae {r['mae']:.3f} r {r['r']:.4f}{extra}"
        )
    print(f"[qbr] chosen: {chosen}")
    return 0 if chosen is not None else GATE_EXIT
