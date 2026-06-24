"""Dataset registry -- the per-dataset config that drives the builders.

Mirrors the R creation scripts' ``build_season(season, dataset, stem, tag,
reshape_fn)`` calls. Each dataset is built one of two ways:

* ``block`` set -- a generic flatten of a (possibly nested) ``final.json`` block
  (``flat_block_frame``). Covers play_participants, game_rosters, injuries,
  power_index, and all 8 ``advBoxScore`` sections.
* ``reshaper`` set -- a bespoke per-game reshape registered in
  :mod:`cfb_data_build.reshapers` (pbp conform, team/player box pivots, drives
  field extraction, betting/schedule scalar rows, linescores, rosters derive).

Exactly one of ``block`` / ``reshaper`` is set. Adding a flatten dataset = one
row; adding a bespoke one = one row + one function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """How to build one released dataset from a ``final.json`` payload.

    Attributes:
        dataset: directory name under ``cfb/`` and the manifest key.
        stem: output file stem (``{stem}_{season}.parquet`` etc.).
        tag: the ``sportsdataverse-data`` release tag.
        block: nested key path to a list-of-dicts block for the generic flatten
            (e.g. ``("advBoxScore", "team")``); ``None`` for bespoke datasets.
        reshaper: key into :data:`cfb_data_build.reshapers.RESHAPERS` for a
            bespoke per-game reshape; ``None`` for generic-flatten datasets.
    """

    dataset: str
    stem: str
    tag: str
    block: tuple[str, ...] | None = None
    reshaper: str | None = None


REGISTRY: dict[str, DatasetSpec] = {
    # --- generic flatten (top-level block) -------------------------------
    "play_participants": DatasetSpec(
        "play_participants",
        "play_participants",
        "espn_cfb_play_participants",
        block=("play_participants",),
    ),
    "game_rosters": DatasetSpec(
        "game_rosters",
        "game_rosters",
        "espn_cfb_game_rosters",
        block=("game_rosters",),
    ),
    "injuries": DatasetSpec(
        "injuries",
        "injuries",
        "espn_cfb_injuries",
        block=("injuries",),
    ),
    # --- generic flatten (nested block) ----------------------------------
    "adv_team": DatasetSpec(
        "adv_team",
        "adv_team",
        "espn_cfb_adv_team",
        block=("advBoxScore", "team"),
    ),
    "adv_passing": DatasetSpec(
        "adv_passing",
        "adv_passing",
        "espn_cfb_adv_passing",
        block=("advBoxScore", "pass"),
    ),
    "adv_rushing": DatasetSpec(
        "adv_rushing",
        "adv_rushing",
        "espn_cfb_adv_rushing",
        block=("advBoxScore", "rush"),
    ),
    "adv_receiving": DatasetSpec(
        "adv_receiving",
        "adv_receiving",
        "espn_cfb_adv_receiving",
        block=("advBoxScore", "receiver"),
    ),
    "adv_defensive": DatasetSpec(
        "adv_defensive",
        "adv_defensive",
        "espn_cfb_adv_defensive",
        block=("advBoxScore", "defensive"),
    ),
    "adv_turnover": DatasetSpec(
        "adv_turnover",
        "adv_turnover",
        "espn_cfb_adv_turnover",
        block=("advBoxScore", "turnover"),
    ),
    "adv_drives": DatasetSpec(
        "adv_drives",
        "adv_drives",
        "espn_cfb_adv_drives",
        block=("advBoxScore", "drives"),
    ),
    "adv_situational": DatasetSpec(
        "adv_situational",
        "adv_situational",
        "espn_cfb_adv_situational",
        block=("advBoxScore", "situational"),
    ),
    # --- bespoke per-game reshapers --------------------------------------
    "pbp": DatasetSpec("pbp", "play_by_play", "espn_cfb_pbp", reshaper="pbp"),
    "team_box": DatasetSpec(
        "team_box", "team_box", "espn_cfb_team_box", reshaper="team_box"
    ),
    "player_box": DatasetSpec(
        "player_box", "player_box", "espn_cfb_player_box", reshaper="player_box"
    ),
    "drives": DatasetSpec("drives", "drives", "espn_cfb_drives", reshaper="drives"),
    "betting": DatasetSpec(
        "betting", "betting", "espn_cfb_betting", reshaper="betting"
    ),
    "schedules": DatasetSpec(
        "schedules", "cfb_schedule", "espn_cfb_schedules", reshaper="schedules"
    ),
    "linescores": DatasetSpec(
        "linescores", "linescores", "espn_cfb_linescores", reshaper="linescores"
    ),
    "power_index": DatasetSpec(
        "power_index", "power_index", "espn_cfb_power_index", reshaper="power_index"
    ),
    # rosters is DERIVED from the game_rosters dataset (season dedup); the
    # registered reshaper derives per-game (correct for parity on one game).
    "rosters": DatasetSpec(
        "rosters", "rosters", "espn_cfb_rosters", reshaper="rosters"
    ),
}
