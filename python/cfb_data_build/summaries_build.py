"""Season driver for the team-summaries family (5 tables from released pbp).

Unlike the ``final.json`` datasets, this family reads the RELEASED
``espn_cfb_pbp`` + ``espn_cfb_schedule`` via the sdv-py loaders, preps the
plays frame (:mod:`cfb_data_build.summaries_input`), runs the R-script-15 port
(:mod:`cfb_data_build.team_summaries`), and writes/publishes each table
through the shared :func:`cfb_data_build.io.write_dataset` /
:func:`cfb_data_build.publish.publish_dataset` path.

``through_week`` filters plays to ``week <= W`` before the build — the
builder-level cumulative snapshot contract (program plan P8).
"""

from __future__ import annotations

import polars as pl

from cfb_data_build.config import SUMMARIES_REGISTRY
from cfb_data_build.io import write_dataset
from cfb_data_build.publish import publish_dataset
from cfb_data_build.summaries_input import prepare_plays_input
from cfb_data_build.team_summaries import build_team_summaries


def build_summaries_season(
    season: int,
    *,
    through_week: int | None = None,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
    pbp: pl.DataFrame | None = None,
    schedule: pl.DataFrame | None = None,
) -> dict[str, int]:
    """Build (and optionally publish) the 5-table family for one season.

    Args:
        season: season to build.
        through_week: keep plays with ``week <= through_week`` only
            (``None`` = full season, what the release tags hold).
        base: output root directory.
        publish: upload parquet + rds + csv to each table's release tag.
        dry_run: print publish actions instead of running them.
        pbp: pre-loaded pbp frame (loaded from the release when ``None``).
        schedule: pre-loaded schedule frame (loaded when ``None``).

    Returns:
        Row counts per table key.
    """
    if through_week is not None:
        if publish:
            raise ValueError(
                "through-week snapshots are not publishable; the release tags "
                "hold season-final builds (weekly long-format is a separate "
                "dataset, program plan P8)"
            )
        # keep snapshots away from the canonical season artifacts + manifest
        base = f"{base}/snapshots/through_wk{through_week:02d}"

    if pbp is None or schedule is None:
        from sportsdataverse.cfb import load_cfb_pbp, load_cfb_schedule

        if pbp is None:
            pbp = load_cfb_pbp(seasons=[season])
        if schedule is None:
            schedule = load_cfb_schedule(seasons=[season])

    # A season with no plays yet (published schedule, no kickoff) used to reach
    # build_team_summaries with an empty frame and die on
    # ColumnNotFoundError: game_id -- three times over, since the weekly caller
    # retries. Report it the way build_derived reports an empty derived season.
    if pbp is None or pbp.height == 0:
        print(
            f"[summaries {season}] 0 plays, skipped (season has not started)",
            flush=True,
        )
        return {key: 0 for key in SUMMARIES_REGISTRY}

    plays = prepare_plays_input(pbp, schedule, season)
    if through_week is not None:
        plays = plays.filter(pl.col("week") <= through_week)
    tables = build_team_summaries(plays, season)

    counts: dict[str, int] = {}
    for key, spec in SUMMARIES_REGISTRY.items():
        df = tables[key]
        paths = write_dataset(df, spec.dataset, season, spec.stem, base=base)
        if publish and paths is not None:
            publish_dataset(spec, season, base=base, dry_run=dry_run)
        counts[key] = 0 if df is None else df.height
        print(f"[summaries {season}] {key}: {counts[key]} rows")
    return counts
