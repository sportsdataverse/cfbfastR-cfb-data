"""Pre-kickoff forward-check snapshot: hierarchical vs served margins for one 2026 slot.

The promotion pre-registration (2026-11-27) fixes the forward check. One snapshot per slot
``t`` (regular week, or 100 + postseason week, as in ``hierarchical.build_corpus``) is
committed and pushed to ``models/hierarchical_forward/<season>_t<t>.parquet`` BEFORE the
slot's first remaining kickoff. Its push time is the publish time, and a game counts only
if that time precedes its kickoff.

- hierarchical arm: ``fit_asof(corpus, season, t, prev, method="eb")``, fit on games with
  ``t' < t`` only. ``prev`` is the previous season's end-of-season posterior from the
  carryover chain that starts at ``FIRST_SEASON`` with zeros, exactly as ``walk_forward``
  builds it.
- served arm: sportsdataverse-py ``cfb_predict_games`` on ``cfb_ratings_weekly`` at
  ``through_week = t - 1`` (postseason: the last regular-season snapshot), the default path.

    cd python && python -m cfb_model_build.cfb_higher_models.forward_snapshot --season 2026 --t 13
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import distribution
from pathlib import Path

# sportsdataverse (pandas + pyarrow) before polars does any work: loading pyarrow after polars
# has run crashed this process on Windows (access violation in pyarrow's import).
import sportsdataverse.cfb  # noqa: F401
import numpy as np
import polars as pl
from scipy.stats import norm

from .hierarchical import build_corpus, fit_asof

FIRST_SEASON = 2014  # walk_forward's chain start (stage 34 default seasons)
#: Win-probability spread for the hierarchical arm: the residual sd of the 2015-2025
#: walk-forward margins (6,201 games, the 2026-11-26 rescore), fixed by the pre-registration.
HIER_MARGIN_SD = 16.15
OUT = Path(__file__).resolve().parents[3] / "models" / "hierarchical_forward"


def _schedule(seasons: list[int]) -> pl.DataFrame:
    from sportsdataverse.cfb import load_cfb_schedule

    s = load_cfb_schedule(seasons)
    return s if isinstance(s, pl.DataFrame) else pl.from_pandas(s)


def _slot(df: pl.DataFrame) -> pl.Expr:
    return (
        pl.when(pl.col("season_type") == "postseason")
        .then(pl.col("week") + 100)
        .otherwise(pl.col("week"))
    )


def hierarchical_margins(
    sched: pl.DataFrame, season: int, t: int, slate: pl.DataFrame
) -> tuple[np.ndarray, dict]:
    """As-of margins for ``slate`` (home_id, away_id, neutral_site) from games before slot ``t``."""
    # The corpus is every completed game BEFORE the slot: earlier seasons, then this season's
    # t' < t. The EM's variance components average over the whole team universe, so a corpus
    # that grew after the snapshot (later games, new opponents) would move theta; this one is
    # recomputable later from the same release.
    done = sched.filter(
        pl.col("home_points").is_not_null()
        & pl.col("away_points").is_not_null()
        & ((pl.col("season") < season) | (_slot(sched) < t))
    )
    corpus = build_corpus(done)
    prev = np.zeros(corpus.n_teams)
    for s in range(FIRST_SEASON, season):
        ts = corpus.games.filter(pl.col("season") == s)["t"]
        prev = fit_asof(corpus, s, int(ts.max()) + 1, prev, method="eb").theta
    fit = fit_asof(corpus, season, t, prev, method="eb")
    tix = {int(x): i for i, x in enumerate(corpus.team_ids)}
    missing = sorted(
        {int(x) for x in pl.concat([slate["home_id"], slate["away_id"]]).to_list()}
        - set(tix)
    )
    if missing:
        raise ValueError(f"teams with no completed game in the corpus: {missing}")
    h = np.array([tix[int(x)] for x in slate["home_id"].to_list()])
    a = np.array([tix[int(x)] for x in slate["away_id"].to_list()])
    nz = slate["neutral_site"].cast(pl.Float64).to_numpy()
    margin = fit.theta[h] - fit.theta[a] + fit.hfa * (1.0 - nz)
    return margin, {
        "hfa": fit.hfa,
        "tau_team": fit.tau_team,
        "sigma_obs": fit.sigma_obs,
    }


def served_margins(season: int, t: int, slate: pl.DataFrame) -> pl.DataFrame:
    """The default sportsdataverse-py pregame path on the as-of ratings snapshot."""
    from sportsdataverse.cfb import load_cfb_ratings_weekly
    from sportsdataverse.cfb.cfb_game_predict import cfb_predict_games

    r = load_cfb_ratings_weekly([season])
    r = r if isinstance(r, pl.DataFrame) else pl.from_pandas(r)
    tw = t - 1 if t < 100 else int(r["through_week"].max())
    ratings = r.filter(pl.col("through_week") == tw).drop("through_week")
    if ratings.is_empty():
        raise ValueError(
            f"no cfb_ratings_weekly {season} snapshot at through_week {tw}"
        )
    key = ratings.schema["team_id"]
    games = slate.select(
        pl.col("game_id"),
        pl.col("home_id").cast(key).alias("home_team_id"),
        pl.col("away_id").cast(key).alias("away_team_id"),
        pl.col("neutral_site").cast(pl.Boolean),
    )
    out = cfb_predict_games(games, ratings)
    return out.select(
        "game_id",
        pl.col("exp_margin").alias("served_margin"),
        pl.col("home_win_prob").alias("served_home_win_prob"),
    )


def snapshot(season: int, t: int) -> pl.DataFrame:
    sched = _schedule(list(range(FIRST_SEASON, season + 1)))
    slate = sched.filter(
        (pl.col("season") == season)
        & pl.col("season_type").is_in(["regular", "postseason"])
        & (pl.col("fbs_participant") == True)  # noqa: E712
        & (_slot(sched) == t)
    ).select(
        pl.col("game_id").cast(pl.Int64),
        pl.col("start_date").alias("kickoff"),
        pl.col("home_id").cast(pl.Int64),
        pl.col("away_id").cast(pl.Int64),
        pl.col("neutral_site").cast(pl.Boolean),
    )
    if slate.is_empty():
        raise ValueError(f"no {season} games in slot t={t}")
    margin, fitinfo = hierarchical_margins(sched, season, t, slate)
    sdv = json.loads(
        distribution("sportsdataverse").read_text("direct_url.json") or "{}"
    )
    out = slate.with_columns(
        pl.Series("hier_margin", margin),
        pl.Series("hier_home_win_prob", norm.cdf(margin / HIER_MARGIN_SD)),
        pl.lit(t, dtype=pl.Int64).alias("asof_t"),
        pl.lit(fitinfo["hfa"]).alias("hier_hfa"),
        pl.lit(sdv.get("vcs_info", {}).get("commit_id")).alias("sdv_commit"),
    ).join(served_margins(season, t, slate), on="game_id", how="left")
    return out.sort("kickoff", "game_id")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forward_snapshot")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument(
        "--t",
        type=int,
        required=True,
        help="slot: regular week, or 100 + postseason week",
    )
    args = ap.parse_args(argv)
    df = snapshot(args.season, args.t)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.season}_t{args.t}.parquet"
    df.write_parquet(path)
    print(
        f"{path}: {df.height} games; served missing {df['served_margin'].null_count()}"
    )
    print(
        df.select("kickoff", "home_id", "away_id", "hier_margin", "served_margin").head(
            8
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
