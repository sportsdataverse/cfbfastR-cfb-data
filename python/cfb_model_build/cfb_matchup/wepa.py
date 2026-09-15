"""WEPA weight search: the scorer and the random search behind the 120 weights.

Port of ``tools/wepa_weight_training.R``. The source scores a candidate weight
vector by (1) turning every play's EPA into ``off_wepa`` / ``def_wepa`` with
the 120 situational weights (``apply_wepa``), (2) for every game of the
corpus, averaging each team's ``off_wepa`` over the plays it ran and
``def_wepa`` over the plays it faced in the games it played EARLIER in that
season (strictly before the game's date), and (3) regressing the home margin
of victory on the four team means; the score is the adjusted R² of that
``lm`` (``sd_err`` is its residual standard error). Candidates are drawn
uniformly per weight; the best adjusted R² wins. The selection is in-sample:
the source never held a season out, and this port reports the held-out score
of the winner alongside its in-sample one rather than pretending otherwise.

The per-game means are the same as-of aggregation as unit F but keyed on the
play's ``pos_team`` / ``def_pos_team`` (as the source's scorer does) rather
than ``offense_play`` / ``defense_play``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from cfb_data_build.matchup_features import FLAG_NAMES, apply_wepa, tag_plays

WEIGHT_NAMES: tuple[str, ...] = tuple(
    f"{side}_{name}_weight" for side in ("off", "def") for name in FLAG_NAMES
)


@dataclass(frozen=True)
class WepaScore:
    adj_r2: float
    sigma: float
    n_games: int


def team_game_wepa(plays_wepa: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Per game, each side's as-of ``off_wepa`` / ``def_wepa`` means and the home margin.

    ``plays_wepa`` needs ``season, play_date, pos_team, def_pos_team, off_wepa,
    def_wepa``; ``games`` needs ``game_id, season, start_date (Date), home_team,
    away_team, home_points, away_points``. A team's mean for game G is over its
    plays dated strictly before G within the same season, so no row reads its
    own game. Missing means stay null (the lm drops those games, as ``na.omit``
    does).
    """
    plays = plays_wepa.filter(pl.col("off_wepa").is_not_null())
    sides = []
    for side, team_col, value in (
        ("off", "pos_team", "off_wepa"),
        ("def", "def_pos_team", "def_wepa"),
    ):
        per_day = (
            plays.group_by(["season", pl.col(team_col).alias("team"), "play_date"])
            .agg(s=pl.col(value).sum(), n=pl.len())
            .sort(["season", "team", "play_date"])
            .with_columns(
                cum_s=pl.col("s").cum_sum().over(["season", "team"]),
                cum_n=pl.col("n").cum_sum().over(["season", "team"]),
            )
        )
        sides.append(
            per_day.rename(
                {"play_date": "date", "cum_s": f"{side}_s", "cum_n": f"{side}_n"}
            ).drop("s", "n")
        )

    def asof(games: pl.DataFrame, team_col: str, prefix: str) -> pl.DataFrame:
        out = games
        for side, table in (("off", sides[0]), ("def", sides[1])):
            # the latest play-day strictly before the game: an asof join backward
            # on the previous calendar day
            j = (
                games.select(
                    "game_id",
                    "season",
                    team=pl.col(team_col),
                    before=pl.col("start_date") - pl.duration(days=1),
                )
                .sort("before")
                .join_asof(
                    table.sort("date"),
                    left_on="before",
                    right_on="date",
                    by=["season", "team"],
                    strategy="backward",
                )
                .select(
                    "game_id",
                    (pl.col(f"{side}_s") / pl.col(f"{side}_n")).alias(
                        f"{prefix}_{side}_wepa"
                    ),
                )
            )
            out = out.join(j, on="game_id", how="left")
        return out

    out = asof(games, "home_team", "home")
    out = asof(out, "away_team", "away")
    return out.with_columns(
        mov_home_team=pl.col("home_points") - pl.col("away_points")
    ).select(
        "game_id",
        "home_off_wepa",
        "home_def_wepa",
        "away_off_wepa",
        "away_def_wepa",
        "mov_home_team",
    )


def score_frame(frame: pl.DataFrame) -> WepaScore:
    """R ``lm(mov_home_team ~ .)`` on the four means: adjusted R² and sigma."""
    data = frame.drop_nulls()
    y = data["mov_home_team"].to_numpy().astype(np.float64)
    x = (
        data.select("home_off_wepa", "home_def_wepa", "away_off_wepa", "away_def_wepa")
        .to_numpy()
        .astype(np.float64)
    )
    n, k = x.shape
    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    sigma = float(np.sqrt(ss_res / (n - k - 1)))
    return WepaScore(float(adj_r2), sigma, int(n))


def score_weights(
    tagged: pl.DataFrame, games: pl.DataFrame, weights: dict[str, float]
) -> WepaScore:
    """Score one weight vector on a tagged play frame (``tag_plays`` output + ``play_date``)."""
    plays = apply_wepa(tagged, weights)
    return score_frame(team_game_wepa(plays, games))


def load_search_table(path: str | Path) -> pl.DataFrame:
    """A candidate table: one row per weight vector (120 weight columns), plus any score columns."""
    df = pl.read_csv(path, null_values=["NA", ""])
    missing = [c for c in WEIGHT_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"search table lacks weights: {missing[:5]} ...")
    return df


def weights_of(row: dict) -> dict[str, float]:
    return {c: float(row[c]) for c in WEIGHT_NAMES}


def random_candidates(
    n: int, *, seed: int, low: float = -1.0, high: float = 1.0
) -> pl.DataFrame:
    """Uniform candidates in ``[low, high]`` per weight (round 1 of the source's search)."""
    rng = np.random.default_rng(seed)
    mat = rng.uniform(low, high, size=(n, len(WEIGHT_NAMES)))
    return pl.DataFrame(mat, schema=list(WEIGHT_NAMES)).with_row_index("candidate")


def search(
    tagged: pl.DataFrame,
    games: pl.DataFrame,
    candidates: pl.DataFrame,
    *,
    holdout: tuple[pl.DataFrame, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Score every candidate; returns the table with ``adj_r2``, ``sd_err``, ``n_games`` appended.

    With ``holdout`` = (tagged, games) of seasons NOT in the search corpus,
    every candidate also gets ``holdout_adj_r2`` -- the number the source never
    computed.
    """
    rows = []
    for row in candidates.iter_rows(named=True):
        w = weights_of(row)
        s = score_weights(tagged, games, w)
        rec = {
            "candidate": row.get("candidate"),
            "adj_r2": s.adj_r2,
            "sd_err": s.sigma,
            "n_games": s.n_games,
        }
        if holdout is not None:
            h = score_weights(holdout[0], holdout[1], w)
            rec["holdout_adj_r2"] = h.adj_r2
        rows.append(rec)
    return candidates.join(pl.DataFrame(rows), on="candidate", how="left")


def tag_for_search(pbp: pl.DataFrame) -> pl.DataFrame:
    """Flags + the columns the scorer keys on, from a deduped pbp frame."""
    tagged = tag_plays(pbp)
    return tagged.with_columns(
        play_date=pl.col("start_date").str.slice(0, 10).str.to_date()
    )


def write_weights(
    weights: dict[str, float], out_dir: Path, *, meta: dict
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wp, mp = out_dir / "wepa_weights.json", out_dir / "wepa_weights_meta.json"
    wp.write_text(json.dumps({k: weights[k] for k in WEIGHT_NAMES}), encoding="utf-8")
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return wp, mp
