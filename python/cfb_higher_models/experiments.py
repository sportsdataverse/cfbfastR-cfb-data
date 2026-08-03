"""The whole experiment suite, one command.

Every result in this package was measured against `cfb_team_summaries_weekly`
as published on 2026-08-01 -- which carried NO opponent adjustment
(adj_off_epa correlated 0.9928 with its own raw EPAplay_off; the ridge penalty
was on the glmnet scale). Those numbers are a FLOOR, not a result, and all of
them have to be re-measured once the corrected datasets land.

Re-running a dozen ad-hoc scripts by hand is how a re-measurement quietly
becomes a partial re-measurement. So the suite is one entry point:

    python -m cfb_higher_models experiments --fresh

``--fresh`` is not optional after a republish. The local parquet cache under
`.cache/higher_models/` is keyed only by season range, so it will happily serve
the OLD, unadjusted data to a re-run that believes it is testing the new. That
is the same class of failure as everything else this module documents: a
component reporting success while doing nothing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl

from .backtest import walk_forward
from .data import _CACHE, build_game_frame, diff_features, paired_features
from .market import ceiling
from .train_game import (
    CONTEXT_FEATURES,
    HEADS,
    ablate,
    add_context,
    compare_feature_sets,
    sweep_blend_k,
)


def _fresh_cache() -> None:
    if _CACHE.exists():
        shutil.rmtree(_CACHE)
        print(f"cleared cache {_CACHE}")


def verify_spine(frame: pl.DataFrame) -> dict[str, float]:
    """Confirm the spine's ratings are ACTUALLY opponent-adjusted.

    The whole point of the re-run. If this still reports ~0.99 the republish
    did not land and every number below is a repeat of the old measurement.
    """
    import numpy as np

    out = {}
    for adj, raw in (("adj_off_epa", "EPAplay_off"), ("adj_def_epa", "EPAplay_def")):
        a, r = f"{adj}_home", f"{raw}_home"
        if a in frame.columns and r in frame.columns:
            d = frame.select(a, r).drop_nulls()
            out[adj] = float(np.corrcoef(d[a].to_numpy(), d[r].to_numpy())[0, 1])
    print("SPINE CHECK -- corr(adjusted, raw); ~0.99 means the no-op is still live:")
    for k, v in out.items():
        verdict = "STILL A NO-OP" if v > 0.95 else "adjusted"
        print(f"    {k}: {v:.4f}  [{verdict}]")
    return out


def run_all(
    seasons: list[int] | None = None,
    out_dir: str = "artifacts/higher_models",
    *,
    fresh: bool = False,
    quick: bool = False,
) -> dict:
    seasons = seasons or list(range(2014, 2026))
    if fresh:
        _fresh_cache()

    res: dict = {"seasons": seasons}
    frame = build_game_frame(seasons, enrich=True)
    res["spine_check"] = verify_spine(frame)
    frame, diffs = diff_features(frame, paired_features(frame))
    frame = add_context(frame)
    feats = diffs + list(CONTEXT_FEATURES)
    print(f"\nframe: {frame.height} games, {len(feats)} features\n")

    # 1. heads, all scored on the same out-of-sample games
    print("=== heads ===")
    res["heads"] = {}
    for name, fn in HEADS.items():
        try:
            rep, _ = walk_forward(frame, lambda tr, te, _f=fn: _f(tr, te, feats=feats), name=name)
            print(rep, "\n")
            res["heads"][name] = {"margin": rep.margin, "wp": rep.wp}
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")

    # 2. the ceiling
    print("=== market ceiling ===")
    try:
        rep, n = ceiling(frame, seasons)
        print(f"matched {n}/{frame.height}")
        print(rep, "\n")
        res["market"] = {"margin": rep.margin, "n": n}
    except Exception as e:  # noqa: BLE001
        print(f"  market: FAILED ({type(e).__name__}: {e})\n")

    # 3. what the rest/carryover blocks are worth, holding the frame fixed
    print("=== feature blocks ===")
    blocks = {
        "rest_": [c for c in diffs if c.startswith(("rest_", "bye_"))],
        "prior_": [c for c in diffs if c.startswith("prior_")],
        "blend_": [c for c in diffs if c.startswith("blend_")],
        "rt_": [c for c in diffs if c.startswith("rt_")],
    }
    sets = {"all": feats}
    for name, cols in blocks.items():
        if cols:
            sets[f"minus {name}({len(cols)})"] = [c for c in feats if c not in cols]
    res["blocks"] = compare_feature_sets(frame, sets)

    if not quick:
        print("\n=== family ablation ===")
        res["ablation"] = ablate(frame, diffs)
        print("\n=== blend_k sweep ===")
        res["blend_k"] = sweep_blend_k(seasons)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = Path(out_dir) / "experiments.json"
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {p}")
    return res


def main(
    seasons: list[int] | None = None,
    out_dir: str = "artifacts/higher_models",
    *,
    fresh: bool = False,
    quick: bool = False,
) -> int:
    run_all(seasons, out_dir, fresh=fresh, quick=quick)
    return 0
