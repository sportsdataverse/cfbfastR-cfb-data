"""Season driver for the team-summaries family (6 tables from released pbp).

Unlike the ``final.json`` datasets, this family reads the RELEASED
``espn_cfb_pbp`` + ``espn_cfb_schedule`` via the sdv-py loaders, preps the
plays frame (:mod:`cfb_data_build.summaries_input`), runs the R-script-15 port
(:mod:`cfb_data_build.team_summaries`), and writes/publishes each table
through the shared :func:`cfb_data_build.io.write_dataset` /
:func:`cfb_data_build.publish.publish_dataset` path.

``through_week`` keeps REGULAR-season games with ``week <= W`` (postseason
week numbering restarts at 1, so it never enters a snapshot) — the
builder-level cumulative snapshot contract (program plan P8).

``team_summaries`` also carries the Paper Index luck columns
(:mod:`cfb_data_build.paper_index`): the per-game shares are computed from the
same pbp frame and cut by the same snapshot game ids as the plays.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cfb_data_build.config import SUMMARIES_REGISTRY
from cfb_data_build.io import dataset_stem, write_dataset
from cfb_data_build.paper_index import attach_luck, paper_index_games_table
from cfb_data_build.publish import publish_dataset
from cfb_data_build.summaries_input import prepare_plays_input
from cfb_data_build.team_summaries import build_team_summaries


def _load_season_rosters(season: int, *, base: str = "cfb") -> pl.DataFrame:
    """The season's ``cfb_rosters``: this build's output under ``base`` first, else the release.

    The daily processor builds ``cfb_rosters`` just before the summaries, so the
    local file is the current one; the ``espn_cfb_rosters`` release is the
    fallback for a build that did not. Empty when neither has the season.
    """
    from cfb_data_build.rosters_espn import SPEC

    local = (
        Path(base)
        / SPEC.dataset
        / "parquet"
        / f"{dataset_stem(SPEC.stem, season)}.parquet"
    )
    if local.exists():
        return pl.read_parquet(local, columns=["athlete_id", "position_abbreviation"])
    from sportsdataverse.cfb import load_cfb_rosters

    return load_cfb_rosters(seasons=[season])


def _keep_games(df: pl.DataFrame, game_ids: pl.Series) -> pl.DataFrame:
    """Rows of ``df`` whose ``game_id`` is in ``game_ids``, compared in the frame's own id dtype."""
    if not (game_ids.dtype.is_integer() or game_ids.dtype == pl.Utf8):
        raise TypeError(
            f"schedule game_id is {game_ids.dtype}: ids are integer or string, never float"
        )
    ids = game_ids.cast(df.schema["game_id"])
    return df.filter(pl.col("game_id").is_in(ids.implode()))


def build_summaries_season(
    season: int,
    *,
    through_week: int | None = None,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
    pbp: pl.DataFrame | None = None,
    schedule: pl.DataFrame | None = None,
    rosters: pl.DataFrame | None = None,
) -> dict[str, int]:
    """Build (and optionally publish) the 6-table family for one season.

    Args:
        season: season to build.
        through_week: keep regular-season games with ``week <= through_week``
            only (``None`` = full season incl. postseason, what the release
            tags hold).
        base: output root directory.
        publish: upload parquet + rds + csv to each table's release tag.
        dry_run: print publish actions instead of running them.
        pbp: pre-loaded pbp frame (loaded from the release when ``None``).
        schedule: pre-loaded schedule frame (loaded when ``None``).
        rosters: pre-loaded ``cfb_rosters`` frame for the player tables'
            ``position_group`` (loaded when ``None``, see
            :func:`_load_season_rosters`).

    Returns:
        Row counts per table key.
    """
    season_base = base  # the roster is a season build output, never a snapshot's
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

    if rosters is None:
        rosters = _load_season_rosters(season, base=season_base)
    if rosters.is_empty():
        print(
            f"[summaries {season}] no cfb_rosters for {season}: "
            "position_group and every *_pos_pct are null",
            flush=True,
        )

    plays = prepare_plays_input(pbp, schedule, season)
    # Paper Index shares of every scored game, from the same pbp as the plays
    paper_games = paper_index_games_table(pbp)
    if through_week is not None:
        # REGULAR-season games through week W only. ESPN restarts postseason
        # week numbering at 1, so a bare `week <= W` put every bowl and CFP
        # game into every snapshot (2024 Notre Dame: valid_games 5 at
        # through_week 1). The schedule decides, not pbp `seasonType`, which
        # is null on some real pre-2013 regular-season games.
        snapshot_ids = schedule.filter(
            (pl.col("season_type_id") == 2) & (pl.col("week") <= through_week)
        )["game_id"]
        # ONE id list cuts the plays and the per-game shares, so a snapshot's
        # luck columns and its play aggregates cannot cover different games
        plays = _keep_games(plays, snapshot_ids)
        paper_games = _keep_games(paper_games, snapshot_ids)
    tables = build_team_summaries(
        plays, season, through_week=through_week, rosters=rosters
    )
    # Attached after build_team_summaries, so league_averages and the
    # conference percentiles in there never see the outcome-derived columns.
    tables["team_summaries"] = attach_luck(
        tables["team_summaries"], paper_games, season, through_week=through_week
    )

    counts: dict[str, int] = {}
    for key, spec in SUMMARIES_REGISTRY.items():
        df = tables[key]
        paths = write_dataset(df, spec.dataset, season, spec.stem, base=base)
        if publish and paths is not None:
            publish_dataset(spec, season, base=base, dry_run=dry_run)
        counts[key] = 0 if df is None else df.height
        print(f"[summaries {season}] {key}: {counts[key]} rows")
    return counts
