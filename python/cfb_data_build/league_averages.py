"""League baselines: mean / median / sd / n of every published metric, per season x level.

One row per (season, level, entity, category, metric). The statistics describe the
DISTRIBUTION OF ENTITY ROWS -- teams, qualified players, team-games -- not a pooled
rate: ``mean`` is the unweighted mean of those rows, the same population the
``_rank`` / ``_pct`` columns and the ``percentiles`` ladder are cut over, so ``n`` is
that percentile's denominator. A null or non-finite value is skipped and does not
count toward ``n``; a metric with no finite value in a level gets no row.

Pure: the caller passes the frames, the level filters and the qualifier gates
(``team_summaries.LEVELS`` / ``PLAYER_QUALIFIERS``). nfl-data carries the same
module; keep the two in step.
"""

from __future__ import annotations

import polars as pl

SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "level": pl.Utf8,
    "entity": pl.Utf8,
    "category": pl.Utf8,
    "metric": pl.Utf8,
    "mean": pl.Float64,
    "median": pl.Float64,
    "sd": pl.Float64,
    "n": pl.Int64,
    # the per-TEAM-GAME gate (14.0 dropbacks, 6.25 carries, 1.875 targets), so a
    # float; null on team rows
    "qualifier_min": pl.Float64,
}

#: numeric columns that identify a row or count its context rather than measure it
_NOT_METRICS = frozenset(
    {"season", "team_id", "player_id", "games", "team_games", "valid_games"}
)
#: a rank, a percentile (incl. cohort ``_pos_pct`` / ``_conf_pct``) or a sample size
#: is derived from a metric, never a metric of its own
_DERIVED_SUFFIXES = ("_rank", "_pct", "_n")

#: category -> entity, in output order
CATEGORIES: dict[str, str] = {
    "passing": "player",
    "rushing": "player",
    "receiving": "player",
    "team_summaries": "team",
    "team_game": "team",
}


def metric_columns(df: pl.DataFrame) -> list[str]:
    """The numeric columns of ``df`` that are metrics (see the module docstring)."""
    return [
        c
        for c, t in df.schema.items()
        if t.is_numeric()
        and c not in _NOT_METRICS
        and not c.endswith(_DERIVED_SUFFIXES)
    ]


def summarize(
    df: pl.DataFrame,
    *,
    season: int,
    level: str,
    entity: str,
    category: str,
    qualifier_min: float | None = None,
) -> pl.DataFrame:
    """mean / median / sd (ddof=1) / n of every metric column of ``df``."""
    cols = metric_columns(df)
    if df.height == 0 or not cols:
        return pl.DataFrame(schema=SCHEMA)
    long = (
        df.select(pl.col(cols).cast(pl.Float64))
        .unpivot(variable_name="metric", value_name="v")
        # is_finite() is null on a null, and filter keeps only True
        .filter(pl.col("v").is_finite())
    )
    out = long.group_by("metric", maintain_order=True).agg(
        mean=pl.col("v").mean(),
        median=pl.col("v").median(),
        sd=pl.col("v").std(),
        n=pl.len(),
    )
    return (
        out.with_columns(
            season=pl.lit(season),
            level=pl.lit(level),
            entity=pl.lit(entity),
            category=pl.lit(category),
            qualifier_min=pl.lit(qualifier_min, dtype=pl.Float64),
        )
        .select(list(SCHEMA))
        .cast(SCHEMA)
    )


def build_league_averages(
    tables: dict[str, pl.DataFrame],
    season: int,
    *,
    levels: dict[str, pl.Expr | None],
    qualifiers: dict[str, tuple[pl.Expr, float]],
) -> pl.DataFrame:
    """Baselines for every category present in ``tables`` (keys of :data:`CATEGORIES`).

    A player category is filtered to its qualifier gate first. A level whose filter
    names a column the frame lacks (``team_game`` has no ``fbs_class``) is skipped;
    a ``None`` filter is the whole population.
    """
    frames: list[pl.DataFrame] = []
    for category, entity in CATEGORIES.items():
        df = tables.get(category)
        if df is None or df.height == 0:
            continue
        qmin: float | None = None
        if category in qualifiers:
            gate, qmin = qualifiers[category]
            df = df.filter(gate)
        for level, cond in levels.items():
            if cond is None:
                part = df
            elif set(cond.meta.root_names()) <= set(df.columns):
                part = df.filter(cond)
            else:
                continue
            frames.append(
                summarize(
                    part,
                    season=season,
                    level=level,
                    entity=entity,
                    category=category,
                    qualifier_min=qmin,
                )
            )
    return pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=SCHEMA)
