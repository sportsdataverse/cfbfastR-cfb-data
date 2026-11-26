"""Gated rebuild of the fourth-down yards model: candidates against both current variants, one look.

Pre-registration: ClaudeCowork ``specs/2026-11-25-cfb-fd-model-rebuild-prereg.md`` (committed
before ``run`` scored anything). The two committed era comparisons disagree on noise: both pooled
a 0.05-bin calibration error over 22 LOSO folds with no SE, their mlogloss is equal to the 4th
decimal, and the corpus changed between them (the spread backfill) by more than either era
delta. This gate scores every arm on the same frozen, never-trained-on season, paired and
clustered by game.

* **Holdout H** -- every 2026 regular-season (``season_type`` 2) game in cfb-raw's final.json
  directory at the freeze, listed in ``fd_rebuild_holdout.json`` (``freeze``). Rows are
  :func:`fd_features` rows after dropping the first-down-by-penalty flag, so they are the
  rush / pass rows the training corpus holds (it carries no penalty flag).
* **Arms** -- A the shipped one-hot booster (sdv-py ``cfb/models/fd_model.ubj``), B the
  ordinal-era recipe, R1 the one-hot recipe, R2 the one-hot recipe on 2014-2025 only; B, R1
  and R2 are refit on the training corpus (2004-2025) at seed 0, no tuning.
* **Gate** -- a candidate (R1, R2) passes when, against A AND against B, the upper 98.75th
  bootstrap percentile of mean(Brier_R - Brier_X) is below 0 and that of
  (cal_err_R - cal_err_X) is below the 0.005 margin; a game-clustered percentile bootstrap
  (2,000 draws, seed 0). 98.75 = one-sided 2.5% over the two candidates (Bonferroni). Among
  passing candidates the lower holdout Brier ships; none passing keeps A.

Usage (from ``python/``)::

    python -m cfb_model_build.model_training.fourth_down.rebuild_gate freeze --final-dir <cfb-raw>/cfb/json/final
    python -m cfb_model_build.model_training.fourth_down.rebuild_gate run --final-dir ... --pbp <pbp_full_v2.parquet> \\
        --shipped <sdv-py>/sportsdataverse/cfb/models/fd_model.ubj --out-dir ../models/fourth_down
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from .constants import (
    FD_FEATURES,
    FD_FEATURES_ERA_ONEHOT,
    FD_FIRST_DOWN_PENALTY_COLS,
    FD_NROUNDS,
    FD_NUM_CLASS,
    FD_PARAMS,
)
from .features import fd_features

HOLDOUT_SEASON, HOLDOUT_SEASON_TYPE = 2026, 2
TRAIN_SEASONS = tuple(range(2004, 2026))
ARMS = {  # name -> (one-hot?, train seasons); A is the shipped booster
    "B": (False, TRAIN_SEASONS),
    "R1": (True, TRAIN_SEASONS),
    "R2": (True, tuple(range(2014, 2026))),
}
CANDIDATES = ("R1", "R2")
BASELINES = ("A", "B")
N_BOOT, SEED = 2000, 0
UPPER_PCT = 98.75  # one-sided 2.5% over two candidates
CAL_MARGIN = 0.005
CAL_BIN = 0.05
MIN_PLAYS, MIN_GAMES = 15_000, 500
GAINS = np.arange(FD_NUM_CLASS) - 10
MANIFEST = "fd_rebuild_holdout.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holdout_games(final_dir: Path) -> list[int]:
    """2026 regular-season game ids with a final.json, from cfb-raw's schedule master beside the directory."""
    master = Path(final_dir).parents[1] / "cfb_schedule_master.parquet"
    sched = pl.read_parquet(master, columns=["id", "season", "season_type"]).filter(
        (pl.col("season") == HOLDOUT_SEASON)
        & (pl.col("season_type") == HOLDOUT_SEASON_TYPE)
    )
    have = {int(f.stem) for f in Path(final_dir).glob("*.json") if f.stem.isdigit()}
    return sorted(
        int(i) for i in sched["id"].cast(pl.Int64).to_list() if int(i) in have
    )


def holdout_plays(final_dir: Path, game_ids: list[int]) -> pl.DataFrame:
    """The frozen games' plays with ``season`` and ``game_id`` stamped, penalty flag dropped."""
    frames = []
    for gid in game_ids:
        obj = json.loads((Path(final_dir) / f"{gid}.json").read_text(encoding="utf-8"))
        plays = obj.get("plays") or []
        if plays:
            frames.append(
                pl.DataFrame(plays, infer_schema_length=None).with_columns(
                    pl.lit(HOLDOUT_SEASON).cast(pl.Int64).alias("season"),
                    pl.lit(gid).cast(pl.Int64).alias("_gid"),
                )
            )
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.drop([c for c in FD_FIRST_DOWN_PENALTY_COLS if c in df.columns])


def p_first_down(probs: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return (probs * (GAINS[None, :] >= distance[:, None])).sum(axis=1)


def cal_err(p: np.ndarray, y: np.ndarray) -> float:
    """Play-weighted mean |bin - event rate| over 0.05-wide bins of the predicted rate."""
    b = np.round(p / CAL_BIN).astype(np.int64)
    n = np.bincount(b)
    ok = n > 0
    rate = np.bincount(b, weights=y)[ok] / n[ok]
    return float((n[ok] * np.abs(np.nonzero(ok)[0] * CAL_BIN - rate)).sum() / n.sum())


def paired(y, p, q, games, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Game-clustered percentile bootstrap of Brier(p) - Brier(q) and cal_err(p) - cal_err(q)."""
    y, p, q = (np.asarray(a, dtype=float) for a in (y, p, q))
    _, g = np.unique(np.asarray(games), return_inverse=True)
    n_g = g.max() + 1
    order = np.argsort(g, kind="stable")
    starts = np.searchsorted(g[order], np.arange(n_g))
    ends = np.r_[starts[1:], len(g)]
    d = (p - y) ** 2 - (q - y) ** 2
    rng = np.random.default_rng(seed)
    bd, bc = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        rows = order[np.concatenate([np.arange(starts[k], ends[k]) for k in pick])]
        bd[i] = d[rows].mean()
        bc[i] = cal_err(p[rows], y[rows]) - cal_err(q[rows], y[rows])
    return {
        "brier_diff": float(d.mean()),
        "brier_diff_lo": float(np.percentile(bd, 100 - UPPER_PCT)),
        "brier_diff_hi": float(np.percentile(bd, UPPER_PCT)),
        "cal_err_diff": cal_err(p, y) - cal_err(q, y),
        "cal_err_diff_lo": float(np.percentile(bc, 100 - UPPER_PCT)),
        "cal_err_diff_hi": float(np.percentile(bc, UPPER_PCT)),
    }


def fit(train: pl.DataFrame, one_hot: bool, seasons: tuple[int, ...]) -> xgb.Booster:
    X, y = fd_features(
        train.filter(pl.col("season").is_in(list(seasons))), era_onehot=one_hot
    )
    params = {**FD_PARAMS, "seed": SEED}
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=FD_NROUNDS)


def decide(results: dict) -> str | None:
    """The pre-registered rule: a candidate beats both baselines on Brier and is non-inferior on cal error."""
    passing = []
    for c in CANDIDATES:
        ok = all(
            results["paired"][f"{c}-{b}"]["brier_diff_hi"] < 0
            and results["paired"][f"{c}-{b}"]["cal_err_diff_hi"] < CAL_MARGIN
            for b in BASELINES
        )
        results["arms"][c]["passed"] = ok
        if ok:
            passing.append(c)
    return min(passing, key=lambda c: results["arms"][c]["brier"]) if passing else None


def freeze(final_dir: Path, out_dir: Path) -> int:
    ids = holdout_games(final_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "season": HOLDOUT_SEASON,
        "season_type": HOLDOUT_SEASON_TYPE,
        "frozen_utc": datetime.now(UTC).isoformat(),
        "game_ids": ids,
    }
    path = out_dir / MANIFEST
    path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"froze {len(ids)} games -> {path} sha256 {sha256(path)}")
    return 0


def run(final_dir: Path, pbp: Path, shipped: Path, out_dir: Path) -> int:
    manifest = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    plays = holdout_plays(final_dir, manifest["game_ids"])
    rows = {}
    for one_hot in (False, True):
        X, y = fd_features(plays, era_onehot=one_hot)
        # fd_features filters but keeps row order and only selects start.pos_team_spread (its filters
        # read homeTeamSpread / overUnder), so a copy carrying the game id there yields each kept row's game
        Xg, _ = fd_features(
            plays.with_columns(
                pl.col("_gid").cast(pl.Float64).alias("start.pos_team_spread")
            ),
            era_onehot=one_hot,
        )
        rows[one_hot] = (X, y, Xg["posteam_spread"].to_numpy().astype(np.int64))
    (Xo, yo, go), (Xh, yh, gh) = rows[False], rows[True]
    assert np.array_equal(yo, yh) and np.array_equal(go, gh), (
        "the two encodings kept different rows"
    )
    n_games = len(np.unique(gh))
    record: dict = {
        "prereg": "ClaudeCowork specs/2026-11-25-cfb-fd-model-rebuild-prereg.md",
        "holdout_manifest_sha256": sha256(out_dir / MANIFEST),
        "pbp_sha256": sha256(pbp),
        "shipped_sha256": sha256(shipped),
        "n_plays": len(yh),
        "n_games": n_games,
        "arms": {},
        "paired": {},
        "run_utc": datetime.now(UTC).isoformat(),
    }
    if len(yh) < MIN_PLAYS or n_games < MIN_GAMES:
        record["chosen"] = None
        record["below_floor"] = True
        (out_dir / "fd_rebuild.gate.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(
            "below the pre-registered floor: nothing scored", record["n_plays"], n_games
        )
        return 3
    dist = Xh["distance"].to_numpy()
    event = ((yh - 10) >= dist).astype(float)
    train = pl.read_parquet(pbp)
    models = {"A": xgb.Booster(model_file=str(shipped))}
    for name, (one_hot, seasons) in ARMS.items():
        models[name] = fit(train, one_hot, seasons)
        print("fit", name, flush=True)
    preds, mll = {}, {}
    for name, m in models.items():
        X = Xh if name != "B" else Xo
        cols = FD_FEATURES_ERA_ONEHOT if name != "B" else FD_FEATURES
        probs = m.predict(xgb.DMatrix(X[cols])).reshape(-1, FD_NUM_CLASS)
        preds[name] = p_first_down(probs, dist)
        mll[name] = float(
            -np.mean(np.log(np.clip(probs[np.arange(len(yh)), yh], 1e-15, 1)))
        )
        record["arms"][name] = {
            "brier": float(np.mean((preds[name] - event) ** 2)),
            "cal_err": cal_err(preds[name], event),
            "mlogloss": mll[name],
            "mean_p": float(preds[name].mean()),
        }
    record["event_rate"] = float(event.mean())
    for c in CANDIDATES:
        for b in BASELINES:
            record["paired"][f"{c}-{b}"] = paired(event, preds[c], preds[b], gh)
    record["paired"]["A-B"] = paired(
        event, preds["A"], preds["B"], gh
    )  # reported: which encoding, as shipped
    chosen = decide(record)
    record["chosen"] = chosen
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        models[name].save_model(str(out_dir / f"fd_rebuild_{name}.ubj"))
    (out_dir / "fd_rebuild.gate.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("n_plays", "n_games", "event_rate", "arms", "chosen")
            },
            indent=2,
        )
    )
    return 0 if chosen else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fd_rebuild_gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--final-dir", type=Path, required=True)
    f.add_argument("--out-dir", type=Path, default=Path("../models/fourth_down"))
    r = sub.add_parser("run")
    r.add_argument("--final-dir", type=Path, required=True)
    r.add_argument("--pbp", type=Path, required=True)
    r.add_argument("--shipped", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, default=Path("../models/fourth_down"))
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        return freeze(a.final_dir, a.out_dir)
    return run(a.final_dir, a.pbp, a.shipped, a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
