"""Score the 2026 forward check of the hierarchical model, once, after the CFP final.

The rule is fixed by the promotion pre-registration (2026-11-27, amendments 1-2). Run it only after the
window closes, from a checkout whose ``origin/main`` is fetched:

    cd python && python -m cfb_model_build.cfb_higher_models.forward_check --season 2026

Snapshots (``models/hierarchical_forward/<season>_t<t>[_<part>].parquet``) are read from ``origin/main`` AS OF
the commit that first added each one; a snapshot changed after that commit is refused. Its publish time is
GitHub's push time (the push activity whose range first contains that commit), not a committer date.

Window: FBS games of the season in regular-season slot >= 13 or the postseason, kicking off after T0.
A game counts from its first snapshot published before its kickoff. Window games no snapshot covers in time
are listed, not imputed.

PASS needs all four clauses (exit 0 only on PASS):
1. wiring parity: sportsdataverse-py ``cfb_hierarchical_ratings`` on the snapshot's committed as-of corpus, plus
   ``cfb_predict_games(model="hierarchical")`` with the real ``cfb_ratings_weekly`` frame, reproduce
   ``hier_margin`` and ``hier_home_win_prob`` to 1e-6. Any non-finite difference or dropped game fails.
2. coverage: every scored game has finite hierarchical and served predictions, and n >= 60 (fewer: INSUFFICIENT);
3. MAE(hier) - MAE(served) <= +0.50 (point estimate; a 2,000-resample paired bootstrap CI, seed 0, is reported);
4. Brier(hier) - Brier(served) <= +0.02.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# sportsdataverse (pandas + pyarrow) before polars does any work; see forward_snapshot.
import sportsdataverse.cfb  # noqa: F401
import numpy as np
import polars as pl

from .forward_snapshot import FIRST_SEASON, OUT, _schedule, _slot

REPO = "sportsdataverse/cfbfastR-cfb-data"
T0 = datetime(
    2026, 11, 27, 7, 45, 40, tzinfo=timezone.utc
)  # the pre-registration's commit time
MIN_GAMES, MAE_MARGIN, BRIER_MARGIN, PARITY_TOL = 60, 0.50, 0.02, 1e-6
ROOT = Path(__file__).resolve().parents[3]
REL = OUT.relative_to(ROOT).as_posix()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout


def _ts(x: str) -> datetime:
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def _pushes() -> list[tuple[str, str, datetime]]:
    out = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{REPO}/activity?ref=refs/heads/main&activity_type=push&per_page=100",
            "--jq",
            ".[] | [.before, .after, .timestamp] | @tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    return [
        (b, a, _ts(t)) for b, a, t in (ln.split("\t") for ln in out.splitlines() if ln)
    ]


def _ancestor(sha: str, of: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", sha, of]
        ).returncode
        == 0
    )


def _published(
    path: str, pushes: list[tuple[str, str, datetime]]
) -> tuple[str, datetime]:
    """(first commit adding ``path`` on main, the push time of that commit). Refuses a rewritten file."""
    commits = _git("log", "origin/main", "--format=%H", "--", path).split()
    if not commits:
        raise RuntimeError(f"{path} is not on main")
    if len(commits) > 1:
        raise RuntimeError(
            f"{path} was changed after it was published ({len(commits)} commits)"
        )
    sha = commits[0]
    hits = [t for b, a, t in pushes if _ancestor(sha, a) and not _ancestor(sha, b)]
    if not hits:
        raise RuntimeError(f"no push on main carries {sha[:9]} ({path})")
    return sha, min(hits)


def _read(sha: str, path: str) -> pl.DataFrame:
    raw = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{sha}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    return pl.read_parquet(io.BytesIO(raw))


def parity(
    corpus: pl.DataFrame,
    season: int,
    t: int,
    snap: pl.DataFrame,
    ratings_weekly: pl.DataFrame,
) -> float:
    """Largest |served-path hierarchical value - snapshot value| over the snapshot's games (inf on any gap)."""
    from sportsdataverse.cfb.cfb_game_predict import cfb_predict_games
    from sportsdataverse.cfb.cfb_hierarchical import cfb_hierarchical_ratings

    hier = cfb_hierarchical_ratings(corpus, season, t, first_season=FIRST_SEASON)
    tw = t - 1 if t < 100 else int(ratings_weekly["through_week"].max())
    ratings = ratings_weekly.filter(pl.col("through_week") == tw).drop("through_week")
    key = ratings.schema["team_id"]
    games = snap.select(
        "game_id",
        pl.col("home_id").cast(pl.Int64).cast(key).alias("home_team_id"),
        pl.col("away_id").cast(pl.Int64).cast(key).alias("away_team_id"),
        pl.col("neutral_site").cast(pl.Boolean),
    )
    got = cfb_predict_games(games, ratings, model="hierarchical", hierarchical=hier)
    j = snap.select("game_id", "hier_margin", "hier_home_win_prob").join(
        got.select("game_id", "exp_margin", "home_win_prob"), on="game_id", how="inner"
    )
    if j.height != snap.height:
        return float("inf")
    d = np.concatenate(
        [
            (j["hier_margin"] - j["exp_margin"]).abs().to_numpy(),
            (j["hier_home_win_prob"] - j["home_win_prob"]).abs().to_numpy(),
        ]
    )
    return float(d.max()) if np.isfinite(d).all() else float("inf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forward_check")
    ap.add_argument("--season", type=int, required=True)
    season = ap.parse_args(argv).season
    from sportsdataverse.cfb import load_cfb_ratings_weekly

    rw = load_cfb_ratings_weekly([season])
    rw = rw if isinstance(rw, pl.DataFrame) else pl.from_pandas(rw)
    sched = _schedule(list(range(FIRST_SEASON, season + 1)))
    pushes = _pushes()
    names = sorted(
        n
        for n in _git("ls-tree", "--name-only", "origin/main", f"{REL}/").split()
        if re.fullmatch(rf"{REL}/{season}_t\d+(_\d+)?\.parquet", n)
    )
    rows, worst = [], 0.0
    for name in names:
        sha, pub = _published(name, pushes)
        snap = _read(sha, name)
        corpus_name = name.replace(".parquet", "_corpus.parquet")
        csha, _ = _published(corpus_name, pushes)
        t = int(snap["asof_t"][0])
        corpus = _read(csha, corpus_name)
        leak = corpus.filter((pl.col("season") == season) & (_slot(corpus) >= t)).height
        if leak:
            raise RuntimeError(
                f"{corpus_name} holds {leak} game(s) at or after slot {t}"
            )
        worst = max(worst, parity(corpus, season, t, snap, rw))
        rows.append(
            snap.with_columns(
                pl.lit(pub).alias("published"), pl.lit(name).alias("snapshot")
            )
        )
    allsnap = pl.concat(rows).with_columns(
        pl.col("kickoff").map_elements(_ts, return_dtype=pl.Datetime("us", "UTC"))
    )
    valid = (
        allsnap.filter(
            (pl.col("kickoff") > pl.col("published")) & (pl.col("kickoff") > T0)
        )
        .sort("published")
        .unique("game_id", keep="first")
    )
    window = (
        sched.filter(
            (pl.col("season") == season)
            & (pl.col("fbs_participant") == True)  # noqa: E712
            & (
                ((pl.col("season_type") == "regular") & (_slot(sched) >= 13))
                | (pl.col("season_type") == "postseason")
            )
        )
        .select(
            pl.col("game_id").cast(pl.Int64),
            pl.col("start_date"),
            (pl.col("home_points") - pl.col("away_points"))
            .cast(pl.Float64)
            .alias("margin"),
        )
        .with_columns(
            pl.col("start_date").map_elements(
                _ts, return_dtype=pl.Datetime("us", "UTC")
            )
        )
        .filter(pl.col("start_date") > T0)
    )
    # the final schedule's kickoff must also follow the publish time: a game moved earlier after
    # its snapshot no longer counts
    valid = valid.join(
        window.select("game_id", "start_date"), on="game_id", how="inner"
    ).filter(pl.col("start_date") > pl.col("published"))
    missed = window.join(valid.select("game_id"), on="game_id", how="anti")[
        "game_id"
    ].to_list()
    w = valid.join(window.select("game_id", "margin"), on="game_id", how="left")
    unscored = w.filter(pl.col("margin").is_null())["game_id"].to_list()
    w = w.filter(pl.col("margin").is_not_null())
    preds = [
        "hier_margin",
        "served_margin",
        "hier_home_win_prob",
        "served_home_win_prob",
    ]
    gaps = w.filter(
        pl.any_horizontal(~pl.col(c).is_finite() | pl.col(c).is_null() for c in preds)
    )["game_id"].to_list()
    y = w["margin"].to_numpy()
    won = (y > 0).astype(float)
    e_h, e_s = (
        np.abs(w["hier_margin"].to_numpy() - y),
        np.abs(w["served_margin"].to_numpy() - y),
    )
    b_h = (w["hier_home_win_prob"].to_numpy() - won) ** 2
    b_s = (w["served_home_win_prob"].to_numpy() - won) ** 2
    d, db = e_h - e_s, b_h - b_s
    idx = np.random.default_rng(0).integers(0, max(len(d), 1), size=(2000, len(d)))
    lo, hi = (
        np.quantile(d[idx].mean(axis=1), [0.025, 0.975]) if len(d) else (np.nan, np.nan)
    )
    checks = {
        f"1 parity max |diff| {worst:.2e} <= {PARITY_TOL}": worst <= PARITY_TOL,
        f"2 coverage n={len(y)} >= {MIN_GAMES}, prediction gaps {gaps}": len(y)
        >= MIN_GAMES
        and not gaps,
        f"3 dMAE {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] <= +{MAE_MARGIN}": bool(
            d.mean() <= MAE_MARGIN
        ),
        f"4 dBrier {db.mean():+.4f} <= +{BRIER_MARGIN}": bool(
            db.mean() <= BRIER_MARGIN
        ),
    }
    print(
        json.dumps(
            {
                "snapshots": names,
                "scored": len(y),
                "missed": missed,
                "unscored": unscored,
            }
        )
    )
    print(
        f"MAE hier {e_h.mean():.3f} served {e_s.mean():.3f}; Brier {b_h.mean():.4f} / {b_s.mean():.4f}"
    )
    for k, ok in checks.items():
        print(("  ok   " if ok else "  FAIL ") + k)
    verdict = (
        "INSUFFICIENT"
        if len(y) < MIN_GAMES
        else ("PASS" if all(checks.values()) else "FAIL")
    )
    print("RESULT:", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
