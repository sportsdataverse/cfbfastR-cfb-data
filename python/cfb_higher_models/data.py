"""As-of game-frame assembly. THE single place the leakage boundary is defined.

Every higher-order model reads its training frame from here. That is deliberate:
the recurring defect in this layer is a model fit on same-game or full-season
inputs and then published with a *predictive* label (``pregame_wp``'s metrics,
``cfb_game_predict``'s "MAE 3.23" claim -- both explanatory fits). Centralising
the join means the boundary is enforced once, in code, instead of being
re-argued per model.

The rule, in one line:

    a game in week W may only see team features with ``through_week <= W - 1``.

``build_game_frame`` is the only function that performs that join. Nothing
downstream should ever join a team-week frame to a schedule itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

# Weekly substrate: 384 columns keyed (team_id, season, through_week).
# Already carries the opponent-adjusted ratings (adj_off_epa / adj_def_epa /
# net_adj_epa) plus per-unit efficiency, so the spine is a selection problem.
_CACHE = Path(os.getenv("CFB_HM_CACHE", ".cache/higher_models"))

# `_rank` columns are dense ranks of columns we already keep. They are monotone
# transforms carrying no extra signal for a tree model, and they trible the
# feature count. Dropped by default; `keep_ranks=True` if an experiment wants
# the normalised-within-week view (ranks ARE scale-stable across eras, which is
# occasionally the point).
_DROP_SUFFIX = "_rank"


def _as_polars(df) -> pl.DataFrame:
    """Loaders are inconsistent about polars vs pandas; normalise to polars."""
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def load_weekly(seasons: list[int], *, cache: bool = True) -> pl.DataFrame:
    """Team-week as-of features, cached to parquet (the release fetch is slow)."""
    key = _CACHE / f"weekly_{min(seasons)}_{max(seasons)}.parquet"
    if cache and key.exists():
        return pl.read_parquet(key)
    from sportsdataverse.cfb import load_cfb_team_summaries_weekly

    df = _as_polars(load_cfb_team_summaries_weekly(seasons))
    if cache:
        key.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(key)
    return df


def load_ratings(seasons: list[int], *, cache: bool = True) -> pl.DataFrame:
    """`cfb_ratings_weekly` -- a SECOND, genuinely different rating system.

    Do not assume this agrees with the ``adj_off_epa`` / ``adj_def_epa`` /
    ``net_adj_epa`` columns in the weekly summaries. Measured on 2024, joined
    on (team_id, season, through_week), they correlate only 0.64-0.68 and the
    summaries version is ~22% narrower. They are not the same metric wearing
    two names:

      * summaries -> FBS-vs-FBS, pass/rush scrimmage plays, kneels dropped
      * cfb_ratings -> competitiveness (garbage-time) filtered, includes
        special teams, and carries the FEI variants

    Both are defensible; the hazard is that two published datasets ship a
    column literally named ``adj_off_epa`` meaning different things. The
    shipped ``cfb_game_predict`` constants were fit against THIS one
    (``adj_net``), so scoring them on the summaries column measures something
    the surface never does. Prefixed ``rt_`` here so the collision is
    impossible to make by accident.
    """
    key = _CACHE / f"ratings_{min(seasons)}_{max(seasons)}.parquet"
    if cache and key.exists():
        return pl.read_parquet(key)
    from sportsdataverse.cfb import load_cfb_ratings_weekly

    df = _as_polars(load_cfb_ratings_weekly(seasons))
    keys = {"team_id", "season", "through_week"}
    df = df.select(
        *[pl.col(c) for c in ("team_id", "season", "through_week")],
        *[
            pl.col(c).alias(f"rt_{c}")
            for c, dt in zip(df.columns, df.dtypes)
            if c not in keys and dt.is_numeric() and not c.endswith("_rank")
        ],
    )
    if cache:
        key.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(key)
    return df


def load_games(seasons: list[int], *, cache: bool = True) -> pl.DataFrame:
    """Completed games with scores, one row per game."""
    key = _CACHE / f"games_{min(seasons)}_{max(seasons)}.parquet"
    if cache and key.exists():
        return pl.read_parquet(key)
    from sportsdataverse.cfb import load_cfb_schedule

    df = _as_polars(load_cfb_schedule(seasons)).filter(
        pl.col("home_points").is_not_null() & pl.col("away_points").is_not_null()
    )
    if cache:
        key.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(key)
    return df


def feature_columns(weekly: pl.DataFrame, *, keep_ranks: bool = False) -> list[str]:
    """Numeric per-team feature columns, excluding keys and (by default) ranks."""
    keys = {
        "team_id",
        "pos_team",
        "division",
        "conference",
        "season",
        "through_week",
        "fbs_class",
    }
    out = []
    for c, dt in zip(weekly.columns, weekly.dtypes):
        if c in keys or not dt.is_numeric():
            continue
        if not keep_ranks and c.endswith(_DROP_SUFFIX):
            continue
        out.append(c)
    return out


#: Opponent-adjusted ratings are null until the ridge has enough games to
#: solve (measured: 195/2079 rows over 2023-25, 174 of them in weeks 2-3).
#: Those rows are real games, but no rating-based head can score them.
#: Both systems must be present -- the shipped closed form needs ``rt_adj_net``
#: and the spine heads need the summaries detail, so a row missing either is
#: unusable for a like-for-like comparison.
RATING_COLS = ("net_adj_epa", "adj_off_epa", "adj_def_epa", "rt_adj_net")


def build_game_frame(
    seasons: list[int],
    *,
    min_week: int = 2,
    keep_ranks: bool = False,
    require_rating: bool = True,
    cache: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """One row per game with STRICTLY as-of features for both teams.

    Week W is joined to ``through_week == W - 1``. Games in ``min_week`` or
    earlier are dropped -- week 1 has no prior week, and week 2 rests on a
    single game of data (measured MAE 20.98 for weeks 2-4 vs 12.98 for 13+),
    so early weeks are noise that a caller should opt into knowingly.

    Returns columns: game_id, season, week, neutral_site, home/away ids,
    ``{feat}_home`` / ``{feat}_away`` for every feature, and the three targets
    ``margin`` (home - away), ``total``, ``home_won``.

    Raises:
        ValueError: if the join key dtypes disagree, or if the join drops
            everything (which is what a silent key-namespace mismatch looks
            like -- see the home/pos_team namespace bug in cfb_ratings).
    """
    weekly = load_weekly(seasons, cache=cache)
    games = load_games(seasons, cache=cache)
    # Carry BOTH rating systems (see load_ratings): the shipped closed form is
    # only meaningful on rt_adj_net, while the spine's efficiency detail lives
    # in the summaries. A head may use either or both.
    weekly = weekly.join(
        load_ratings(seasons, cache=cache).with_columns(
            pl.col("team_id").cast(pl.Int64),
            pl.col("season").cast(pl.Int64),
            pl.col("through_week").cast(pl.Int64),
        ),
        left_on=[
            pl.col("team_id").cast(pl.Int64),
            pl.col("season").cast(pl.Int64),
            pl.col("through_week").cast(pl.Int64),
        ],
        right_on=["team_id", "season", "through_week"],
        how="left",
    )
    feats = feature_columns(weekly, keep_ranks=keep_ranks)

    # ID discipline: pin one dtype at the boundary and assert agreement rather
    # than papering over with a float->Utf8 cast (a float-origin id stringifies
    # as "123.0" and silently matches nothing).
    w = weekly.select(
        pl.col("team_id").cast(pl.Int64),
        pl.col("season").cast(pl.Int64),
        pl.col("through_week").cast(pl.Int64),
        *[pl.col(c).cast(pl.Float64) for c in feats],
    )
    g = games.select(
        pl.col("game_id").cast(pl.Int64),
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("home_id").cast(pl.Int64),
        pl.col("away_id").cast(pl.Int64),
        pl.col("neutral_site").cast(pl.Boolean),
        pl.col("home_points").cast(pl.Float64),
        pl.col("away_points").cast(pl.Float64),
    ).filter(pl.col("week") >= min_week)

    if g.schema["home_id"] != w.schema["team_id"]:
        raise ValueError(
            f"join key dtype mismatch: home_id={g.schema['home_id']} team_id={w.schema['team_id']}"
        )

    # THE BOUNDARY: week W sees only through_week == W - 1.
    g = g.with_columns((pl.col("week") - 1).alias("_asof"))

    for side in ("home", "away"):
        side_w = w.rename({c: f"{c}_{side}" for c in feats}).rename(
            {"team_id": f"_tid_{side}", "through_week": "_asof"}
        )
        g = g.join(
            side_w,
            left_on=["season", "_asof", f"{side}_id"],
            right_on=["season", "_asof", f"_tid_{side}"],
            how="inner",
        )

    if g.height == 0:
        raise ValueError(
            "as-of join produced zero rows -- check that team_id and "
            "home_id/away_id share an id namespace and that through_week "
            "covers the requested weeks"
        )

    g = g.with_columns(
        (pl.col("home_points") - pl.col("away_points")).alias("margin"),
        (pl.col("home_points") + pl.col("away_points")).alias("total"),
        (pl.col("home_points") > pl.col("away_points")).alias("home_won"),
        # `rated` is kept even when we drop on it, so a caller that opts out
        # can still segment instead of silently mixing rated/unrated rows.
        pl.all_horizontal(
            [
                pl.col(f"{c}_{s}").is_not_null()
                for c in RATING_COLS
                for s in ("home", "away")
            ]
        ).alias("rated"),
    ).drop("_asof")

    if require_rating:
        before = g.height
        g = g.filter(pl.col("rated"))
        if verbose and before != g.height:
            print(
                f"build_game_frame: dropped {before - g.height}/{before} games "
                f"with no opponent-adjusted rating (ridge needs a few games; "
                f"mostly weeks 2-3). Pass require_rating=False to keep them."
            )
    return g


def paired_features(frame: pl.DataFrame) -> list[str]:
    """The ``*_home`` / ``*_away`` feature columns present on a built frame."""
    return [c for c in frame.columns if c.endswith(("_home", "_away"))]


def diff_features(
    frame: pl.DataFrame, feats: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    """Collapse each home/away pair to a single ``{feat}_diff`` column.

    Halves the feature count and bakes in the symmetry a margin model would
    otherwise have to learn twice. Keeps ``neutral_site`` so HFA stays
    estimable. Returns (frame_with_diffs, diff_column_names).
    """
    bases = sorted({c[:-5] for c in feats if c.endswith("_home")})
    both = [b for b in bases if f"{b}_away" in frame.columns]
    diffs = [
        (pl.col(f"{b}_home") - pl.col(f"{b}_away")).alias(f"{b}_diff") for b in both
    ]
    return frame.with_columns(diffs), [f"{b}_diff" for b in both]
