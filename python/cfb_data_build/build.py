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
from cfb_data_ingest.schedule import season_game_ids


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


def build_dataset_frame(spec: DatasetSpec, game: dict[str, Any]) -> pl.DataFrame:
    """Reshape one game's payload into the dataset's per-game frame (the ``reshape_fn``)."""
    if spec.reshaper is not None:
        return reshapers.RESHAPERS[spec.reshaper](game)
    if spec.block is not None:
        return flat_block_frame(_resolve_block(game, spec.block), game)
    raise ValueError(f"{spec.dataset}: spec has neither block nor reshaper")


def build_season(
    spec: DatasetSpec,
    season: int,
    *,
    cache_dir: str | Path,
    schedule: str | Path | None = None,
    fetch: bool = True,
    publish: bool = False,
    base: str | Path = "cfb",
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
    cache = Path(cache_dir)
    frames: list[pl.DataFrame | None] = []
    for gid in ids:
        path = cache / f"{gid}.json"
        if not path.exists():
            continue
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
            frames.append(build_dataset_frame(spec, game))
        except Exception as exc:  # noqa: BLE001 — one bad game cannot abort the season
            print(f"{spec.dataset} {gid}: {exc}")
    df = bind_games(frames)
    print(f"{spec.dataset} {season}: {df.height} rows from {len(ids)} games")
    write_dataset(df, spec.dataset, season, spec.stem, base=base)
    if publish and df.height > 0:
        from cfb_data_build.publish import publish_dataset

        publish_dataset(spec, season, base=base)
    return df


def build_rosters_season(
    season: int, *, base: str | Path = "cfb", publish: bool = False
) -> pl.DataFrame:
    """Season roster = dedup of the already-built game_rosters parquet (R espn_cfb_08).

    rosters is DERIVED from the whole-season game_rosters output (not per game),
    so build game_rosters for the season first. Reads its parquet, derives one
    row per athlete-team (latest game), writes + optionally publishes.
    """
    spec = REGISTRY["rosters"]
    gr_path = Path(base) / "game_rosters" / "parquet" / f"game_rosters_{season}.parquet"
    if not gr_path.exists():
        print(
            f"rosters {season}: no game_rosters parquet at {gr_path} (build game_rosters first)"
        )
        return pl.DataFrame()
    gr = pl.read_parquet(gr_path)
    df = reshapers.derive_rosters(gr)
    print(
        f"rosters {season}: {df.height} athlete-team rows (from {gr.height} game-roster rows)"
    )
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
) -> None:
    """Build a dataset across an inclusive season range (the R script ``main`` loop)."""
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
            publish=publish,
            base=base,
        )
