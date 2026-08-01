"""Per-season build driver -- polars port of ``build_season``.

Enumerate season game ids -> fetch/read each ``final.json`` -> reshape per the
:class:`~cfb_data_build.config.DatasetSpec` -> drift-safe union -> write -> (opt)
publish. Port of ``R/_data_utils.R:184-199``. Reuses
:func:`cfb_data_ingest.fetch.fetch_final` and
:func:`cfb_data_ingest.schedule.season_game_ids` for the network/enumeration
layer (kept isolated, exactly as the R side isolates ``fetch_*``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cfb_data_build import reshapers
from cfb_data_build.config import REGISTRY, DatasetSpec
from cfb_data_build.io import write_dataset
from cfb_data_build.reshape import bind_games, flat_block_frame
from cfb_data_ingest.fetch import fetch_final
from cfb_data_ingest.schedule import SCHEDULE_URL, season_game_ids

# ESPN's advBoxScore blocks put a team ID in `pos_team` / `def_pos_team` -- a
# name-shaped column. We surface the ID as `<col>_id` and fill `<col>` with the
# readable display name, so the column finally means what it is named.
_TEAM_ID_COLS = ("pos_team", "def_pos_team")


def _team_names(schedule_path_or_url: str | Path | None, season: int) -> pl.DataFrame:
    """``(team_id, team_name)`` for one season, unioned over the home and away sides.

    The schedule master stores ``home_id`` / ``away_id`` as **String** while the
    advBoxScore blocks emit ``pos_team`` as **Int64**, so both sides are cast to
    Int64 before they are ever used as a join key (a raw join would silently
    match nothing).
    """
    src = str(schedule_path_or_url) if schedule_path_or_url is not None else SCHEDULE_URL
    lf = pl.scan_parquet(src).filter(pl.col("season") == season)
    sides = [
        lf.select(
            pl.col(f"{side}_id").cast(pl.Int64, strict=False).alias("team_id"),
            pl.col(f"{side}_display_name").cast(pl.Utf8).alias("team_name"),
        )
        for side in ("home", "away")
    ]
    return pl.concat(sides).drop_nulls("team_id").unique(subset=["team_id"]).collect()


def _resolve_team_names(df: pl.DataFrame, season: int, schedule: str | Path | None) -> pl.DataFrame:
    """Split ``pos_team`` / ``def_pos_team`` into ``<col>_id`` + readable ``<col>``.

    No-op when the frame carries neither column or the season has no schedule
    rows, so a dataset without a possession team is untouched. The id column is
    kept adjacent to the name in the original position rather than appended, so
    the column order stays readable.
    """
    present = [c for c in _TEAM_ID_COLS if c in df.columns]
    if not present or df.height == 0:
        return df
    names = _team_names(schedule, season)
    if names.height == 0:
        print(f"  team names: no schedule rows for {season}, leaving {present} as ids")
        return df
    order: list[str] = []
    for col in df.columns:
        order.extend([f"{col}_id", col] if col in present else [col])
    for col in present:
        df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False).alias(f"{col}_id")).drop(col)
        df = df.join(names.rename({"team_id": f"{col}_id", "team_name": col}), on=f"{col}_id", how="left")
        hit = df[col].drop_nulls().len() / df.height
        print(f"  {col}: {hit:.1%} of rows resolved to a team name")
    return df.select(order)


def _resolve_block(game: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Navigate a nested key path to a block; ``None`` if any level is missing."""
    node: Any = game
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node


def build_dataset_frame(spec: DatasetSpec, game: dict[str, Any], *, output: str = "default") -> pl.DataFrame:
    """Reshape one game's payload into the dataset's per-game frame (the ``reshape_fn``).

    ``output`` selects the pbp column tier (see
    :func:`cfb_data_build.pbp.apply_pbp_output_schema`); every other reshaper
    ignores it.

    PUBLISH espn_cfb_pbp WITH ``output="full"``. The ``default`` tier drops
    ``sack_vec`` (via ``PBP_DROP_REDUNDANT``), which ``team_summaries`` and
    ``summaries_input`` both consume -- publishing a default-tier pbp would
    break the summaries rebuild that reads it back.
    """
    if spec.reshaper is not None:
        fn = reshapers.RESHAPERS[spec.reshaper]
        return fn(game, output=output) if spec.reshaper == "pbp" else fn(game)
    if spec.block is not None:
        return flat_block_frame(_resolve_block(game, spec.block), game)
    raise ValueError(f"{spec.dataset}: spec has neither block nor reshaper")


# Released game-ids per season, cached for the PROCESS. The fan-out runs 10
# adv_* datasets x 22 seasons; without this the release parquet (~34 MB) would
# be re-downloaded 220 times (~7.5 GB) instead of 22.
_RELEASE_IDS_CACHE: dict[int, set[int]] = {}


def _union_release_ids(spec: DatasetSpec, season: int, ids: list[int], cache: Path) -> list[int]:
    """Add game ids the CURRENT release has but the schedule master does not list.

    The master is not a superset of what was published: sampled 2004-2024 it
    misses 160 released games (2012: 54, 2008: 49, 2016: 33), while 2004's
    release conversely has FEWER games than the master. Enumerating from the
    master alone would silently DROP those games from the republished dataset,
    so a consumer querying one today would find it gone tomorrow.

    Only ids with a cached ``final.json`` can actually contribute rows; the rest
    are reported so the shortfall is visible rather than silent.
    """
    cached = _RELEASE_IDS_CACHE.get(season)
    if cached is None:
        try:
            import sportsdataverse.cfb as cfb

            released = cfb.load_cfb_pbp([season])
        except Exception as exc:  # noqa: BLE001 - never let this abort a build
            print(f"{spec.dataset} {season}: release id-union skipped ({type(exc).__name__})")
            return ids

        import polars as _pl

        cached = {x for x in released["game_id"].cast(_pl.Int64).to_list() if x is not None}
        _RELEASE_IDS_CACHE[season] = cached
    released_ids = cached

    extra = sorted(released_ids - set(ids))
    if not extra:
        return ids
    have = [g for g in extra if (cache / f"{g}.json").exists()]
    print(
        f"{spec.dataset} {season}: +{len(have)} release-only games recovered "
        f"({len(extra) - len(have)} released ids have no cached final -- still absent)"
    )
    return list(ids) + have


def build_season(
    spec: DatasetSpec,
    season: int,
    *,
    cache_dir: str | Path,
    schedule: str | Path | None = None,
    fetch: bool = True,
    publish: bool = False,
    base: str | Path = "cfb",
    output: str = "default",
    include_release_ids: bool = False,
) -> pl.DataFrame:
    """Build (and optionally publish) one dataset for one season.

    Args:
        spec: the dataset to build.
        season: season year.
        cache_dir: directory of cached ``{game_id}.json`` payloads.
        schedule: schedule master path/URL (``None`` -> the default raw URL).
        fetch: when ``True``, download any missing games into ``cache_dir`` first.
        publish: when ``True`` and the frame is non-empty, upload to the release.
        base: output root (``cfb`` by default).

    Returns:
        The bound, written per-season frame (possibly empty).
    """
    if fetch:
        fetch_final([season], cache_dir, schedule=schedule)
    ids = season_game_ids(schedule, [season])
    if include_release_ids:
        ids = _union_release_ids(spec, season, ids, Path(cache_dir))
    cache = Path(cache_dir)
    frames: list[pl.DataFrame | None] = []
    for gid in ids:
        path = cache / f"{gid}.json"
        if not path.exists():
            continue
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
            frames.append(build_dataset_frame(spec, game, output=output))
        except Exception as exc:  # noqa: BLE001 — one bad game cannot abort the season
            print(f"{spec.dataset} {gid}: {exc}")
    df = bind_games(frames)
    df = _resolve_team_names(df, season, schedule)
    print(f"{spec.dataset} {season}: {df.height} rows from {len(ids)} games")
    write_dataset(df, spec.dataset, season, spec.stem, base=base)
    if publish and df.height > 0:
        from cfb_data_build.publish import publish_dataset

        publish_dataset(spec, season, base=base)
    return df


def build_rosters_season(season: int, *, base: str | Path = "cfb", publish: bool = False) -> pl.DataFrame:
    """Season roster = dedup of the already-built game_rosters parquet (R espn_cfb_08).

    rosters is DERIVED from the whole-season game_rosters output (not per game),
    so build game_rosters for the season first. Reads its parquet, derives one
    row per athlete-team (latest game), writes + optionally publishes.
    """
    spec = REGISTRY["rosters"]
    gr_path = Path(base) / "game_rosters" / "parquet" / f"game_rosters_{season}.parquet"
    if not gr_path.exists():
        print(f"rosters {season}: no game_rosters parquet at {gr_path} (build game_rosters first)")
        return pl.DataFrame()
    gr = pl.read_parquet(gr_path)
    df = reshapers.derive_rosters(gr)
    print(f"rosters {season}: {df.height} athlete-team rows (from {gr.height} game-roster rows)")
    write_dataset(df, spec.dataset, season, spec.stem, base=base)
    if publish and df.height > 0:
        from cfb_data_build.publish import publish_dataset

        publish_dataset(spec, season, base=base)
    return df


def build_dataset(
    dataset: str,
    start_year: int,
    end_year: int,
    *,
    cache_dir: str | Path,
    schedule: str | Path | None = None,
    fetch: bool = True,
    publish: bool = False,
    base: str | Path = "cfb",
    output: str = "default",
    include_release_ids: bool = False,
) -> None:
    """Build a dataset across an inclusive season range (the R script ``main`` loop).

    ``output`` is the pbp column tier; publish espn_cfb_pbp with ``"full"``.
    """
    # rosters is a season-level derive over the game_rosters output, not a
    # per-game build -- route it to its dedicated season builder.
    if dataset == "rosters":
        for season in range(start_year, end_year + 1):
            build_rosters_season(season, base=base, publish=publish)
        return
    spec = REGISTRY[dataset]
    for season in range(start_year, end_year + 1):
        build_season(
            spec,
            season,
            cache_dir=cache_dir,
            schedule=schedule,
            fetch=fetch,
            output=output,
            include_release_ids=include_release_ids,
            publish=publish,
            base=base,
        )
