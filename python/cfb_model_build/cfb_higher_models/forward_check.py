"""Score the 2026 forward check of the hierarchical model, once, after the CFP final.

The rule is fixed by the promotion pre-registration (2026-11-27). Run it only after the window closes:

    cd python && python -m cfb_model_build.cfb_higher_models.forward_check --season 2026

Window: each game in ``models/hierarchical_forward/<season>_t*.parquet`` whose kickoff is later than its
snapshot's publish time. The publish time is GitHub's committer date of the first ``main`` commit that added
the file. Games without a final score are dropped and listed.

PASS needs all four clauses:
1. wiring parity: sportsdataverse-py ``cfb_hierarchical_ratings`` + ``cfb_predict_games(model="hierarchical")``
   reproduce ``hier_margin`` to 1e-6;
2. coverage: every window game has both arms, and n >= 60 (fewer is INSUFFICIENT);
3. MAE(hier) - MAE(served) <= +0.50 (point estimate; the 2,000-game paired bootstrap CI, seed 0, is reported);
4. Brier(hier) - Brier(served) <= +0.02.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# sportsdataverse (pandas + pyarrow) before polars does any work; see forward_snapshot.
import sportsdataverse.cfb  # noqa: F401
import numpy as np
import polars as pl

from .forward_snapshot import FIRST_SEASON, OUT, _schedule

REPO = "sportsdataverse/cfbfastR-cfb-data"
MIN_GAMES, MAE_MARGIN, BRIER_MARGIN, PARITY_TOL = 60, 0.50, 0.02, 1e-6


def publish_time(path: str) -> str:
    """Committer date (UTC ISO) of the first main commit that added ``path``, from GitHub."""
    out = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{REPO}/commits?path={path}&sha=main&per_page=100",
            "--jq",
            "[.[].commit.committer.date]",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    dates = json.loads(out)
    if not dates:
        raise RuntimeError(f"{path} is not on main")
    return min(dates)


def parity(sched: pl.DataFrame, season: int, t: int, snap: pl.DataFrame) -> float:
    """Largest |served-path hierarchical margin - snapshot margin| over the slot's games."""
    from sportsdataverse.cfb.cfb_game_predict import cfb_predict_games
    from sportsdataverse.cfb.cfb_hierarchical import cfb_hierarchical_ratings

    hier = cfb_hierarchical_ratings(sched, season, t, first_season=FIRST_SEASON)
    games = snap.select(
        "game_id",
        pl.col("home_id").cast(pl.Utf8).alias("home_team_id"),
        pl.col("away_id").cast(pl.Utf8).alias("away_team_id"),
        "neutral_site",
    )
    ratings = hier.select(
        "team_id",
        (pl.col("theta") / 40.0).alias("adj_net"),
        pl.lit(0.0).alias("adj_off_epa"),
        pl.lit(0.0).alias("adj_def_epa"),
        pl.lit(70.0).alias("off_pace"),
        pl.lit(9).alias("games"),
        pl.lit(70.0).alias("prior_off_pace"),
    )  # only exp_total reads these; the margin comes from `hier`
    got = cfb_predict_games(games, ratings, model="hierarchical", hierarchical=hier)
    j = snap.select("game_id", "hier_margin").join(
        got.select("game_id", "exp_margin"), on="game_id"
    )
    return float((j["hier_margin"] - j["exp_margin"]).abs().max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forward_check")
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args(argv)
    season = args.season
    sched = _schedule(list(range(FIRST_SEASON, season + 1)))
    truth = sched.filter(pl.col("season") == season).select(
        pl.col("game_id").cast(pl.Int64),
        (pl.col("home_points") - pl.col("away_points"))
        .cast(pl.Float64)
        .alias("margin"),
    )
    rows, worst, dropped = [], 0.0, []
    for path in sorted(OUT.glob(f"{season}_t*.parquet")):
        snap = pl.read_parquet(path)
        t = int(snap["asof_t"][0])
        pub = publish_time(f"models/hierarchical_forward/{path.name}")
        late = snap.filter(pl.col("kickoff") > pub)
        worst = max(worst, parity(sched, season, t, late)) if late.height else worst
        dropped += snap.filter(pl.col("kickoff") <= pub)["game_id"].to_list()
        rows.append(late.with_columns(pl.lit(pub).alias("published")))
    w = pl.concat(rows).join(truth, on="game_id", how="left")
    unscored = w.filter(pl.col("margin").is_null())["game_id"].to_list()
    w = w.drop_nulls(["margin", "hier_margin", "served_margin"])
    y = w["margin"].to_numpy()
    won = (y > 0).astype(float)
    e_h, e_s = (
        np.abs(w["hier_margin"].to_numpy() - y),
        np.abs(w["served_margin"].to_numpy() - y),
    )
    b_h = (w["hier_home_win_prob"].to_numpy() - won) ** 2
    b_s = (w["served_home_win_prob"].to_numpy() - won) ** 2
    d = e_h - e_s
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
    checks = {
        f"1 parity max |diff| {worst:.2e} <= {PARITY_TOL}": worst <= PARITY_TOL,
        f"2 coverage n={len(y)} >= {MIN_GAMES}": len(y) >= MIN_GAMES,
        f"3 dMAE {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] <= +{MAE_MARGIN}": d.mean()
        <= MAE_MARGIN,
        f"4 dBrier {(b_h - b_s).mean():+.4f} <= +{BRIER_MARGIN}": (b_h - b_s).mean()
        <= BRIER_MARGIN,
    }
    print(
        f"window {len(y)} games; MAE hier {e_h.mean():.3f} served {e_s.mean():.3f}; Brier {b_h.mean():.4f} / {b_s.mean():.4f}"
    )
    print(f"dropped (kickoff before publish): {dropped}; no final score: {unscored}")
    for k, ok in checks.items():
        print(("  ok   " if ok else "  FAIL ") + k)
    verdict = (
        "INSUFFICIENT"
        if len(y) < MIN_GAMES
        else ("PASS" if all(checks.values()) else "FAIL")
    )
    print("RESULT:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
