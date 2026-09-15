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
    #: a ``sportsdataverse.football.usage_box`` section computed at BUILD time
    #: from the final's ``plays`` + ``play_participants`` (never read from
    #: final.json, so the installed sdv-py's definitions always win)
    usage_section: str | None = None
    #: sum the per-game usage rows into one season leaderboard
    aggregate: bool = False
    #: a ``sportsdataverse.football.tendencies`` cut over the season's plays:
    #: ``"team"`` (season x team), ``"coach"`` (season x team x head coach) or
    #: ``"careers"`` (every written coach season summed per coach, one file)
    tendencies: str | None = None


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
    "adv_defensive_players": DatasetSpec(
        "adv_defensive_players",
        "adv_defensive_players",
        "espn_cfb_adv_defensive_players",
        block=("advBoxScore", "defensive_players"),
    ),
    "adv_specialists": DatasetSpec(
        "adv_specialists",
        "adv_specialists",
        "espn_cfb_adv_specialists",
        block=("advBoxScore", "specialists"),
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
    # --- usage / situational box (build-time, sportsdataverse.football.usage_box) --
    # per game (shims 30-35) and season leaderboards (40-45); R twin pending.
    "adv_player_usage": DatasetSpec(
        "adv_player_usage",
        "adv_player_usage",
        "espn_cfb_adv_player_usage",
        usage_section="player_usage",
    ),
    "usage_players": DatasetSpec(
        "usage_players",
        "usage_players",
        "espn_cfb_usage_players",
        usage_section="player_usage",
        aggregate=True,
    ),
    "adv_position_group_usage": DatasetSpec(
        "adv_position_group_usage",
        "adv_position_group_usage",
        "espn_cfb_adv_position_group_usage",
        usage_section="position_group_usage",
    ),
    "usage_position_groups": DatasetSpec(
        "usage_position_groups",
        "usage_position_groups",
        "espn_cfb_usage_position_groups",
        usage_section="position_group_usage",
        aggregate=True,
    ),
    "adv_tackles": DatasetSpec(
        "adv_tackles", "adv_tackles", "espn_cfb_adv_tackles", usage_section="tackles"
    ),
    "usage_tackles": DatasetSpec(
        "usage_tackles",
        "usage_tackles",
        "espn_cfb_usage_tackles",
        usage_section="tackles",
        aggregate=True,
    ),
    "adv_position_group_tackles": DatasetSpec(
        "adv_position_group_tackles",
        "adv_position_group_tackles",
        "espn_cfb_adv_position_group_tackles",
        usage_section="position_group_tackles",
    ),
    "usage_position_group_tackles": DatasetSpec(
        "usage_position_group_tackles",
        "usage_position_group_tackles",
        "espn_cfb_usage_position_group_tackles",
        usage_section="position_group_tackles",
        aggregate=True,
    ),
    "adv_team_usage": DatasetSpec(
        "adv_team_usage",
        "adv_team_usage",
        "espn_cfb_adv_team_usage",
        usage_section="team_usage",
    ),
    "usage_teams": DatasetSpec(
        "usage_teams",
        "usage_teams",
        "espn_cfb_usage_teams",
        usage_section="team_usage",
        aggregate=True,
    ),
    "adv_drive_scripting": DatasetSpec(
        "adv_drive_scripting",
        "adv_drive_scripting",
        "espn_cfb_adv_drive_scripting",
        usage_section="drive_scripting",
    ),
    "usage_drive_scripting": DatasetSpec(
        "usage_drive_scripting",
        "usage_drive_scripting",
        "espn_cfb_usage_drive_scripting",
        usage_section="drive_scripting",
        aggregate=True,
    ),
    "adv_st_kickers": DatasetSpec(
        "adv_st_kickers",
        "adv_st_kickers",
        "espn_cfb_adv_st_kickers",
        usage_section="st_kickers",
    ),
    "usage_st_kickers": DatasetSpec(
        "usage_st_kickers",
        "usage_st_kickers",
        "espn_cfb_usage_st_kickers",
        usage_section="st_kickers",
        aggregate=True,
    ),
    "adv_st_punters": DatasetSpec(
        "adv_st_punters",
        "adv_st_punters",
        "espn_cfb_adv_st_punters",
        usage_section="st_punters",
    ),
    "usage_st_punters": DatasetSpec(
        "usage_st_punters",
        "usage_st_punters",
        "espn_cfb_usage_st_punters",
        usage_section="st_punters",
        aggregate=True,
    ),
    "adv_st_returners": DatasetSpec(
        "adv_st_returners",
        "adv_st_returners",
        "espn_cfb_adv_st_returners",
        usage_section="st_returners",
    ),
    "usage_st_returners": DatasetSpec(
        "usage_st_returners",
        "usage_st_returners",
        "espn_cfb_usage_st_returners",
        usage_section="st_returners",
        aggregate=True,
    ),
    "adv_st_blocks": DatasetSpec(
        "adv_st_blocks",
        "adv_st_blocks",
        "espn_cfb_adv_st_blocks",
        usage_section="st_blocks",
    ),
    "usage_st_blocks": DatasetSpec(
        "usage_st_blocks",
        "usage_st_blocks",
        "espn_cfb_usage_st_blocks",
        usage_section="st_blocks",
        aggregate=True,
    ),
    "adv_st_team": DatasetSpec(
        "adv_st_team", "adv_st_team", "espn_cfb_adv_st_team", usage_section="st_team"
    ),
    "usage_st_team": DatasetSpec(
        "usage_st_team",
        "usage_st_team",
        "espn_cfb_usage_st_team",
        usage_section="st_team",
        aggregate=True,
    ),
    # --- team / coach tendencies (build-time, sportsdataverse.football.tendencies) --
    "team_tendencies": DatasetSpec(
        "team_tendencies",
        "team_tendencies",
        "espn_cfb_team_tendencies",
        tendencies="team",
    ),
    "coach_tendencies": DatasetSpec(
        "coach_tendencies",
        "coach_tendencies",
        "espn_cfb_coach_tendencies",
        tendencies="coach",
    ),
    "coach_careers": DatasetSpec(
        "coach_careers", "coach_careers", "espn_cfb_coach_careers", tendencies="careers"
    ),
}

# --- usage / situational box (sportsdataverse.football.usage_box) -----------
# Eleven per-game sections (shims 30-40) and their season leaderboards (50-60).
# Computed at build time from each final's plays + participants; the R chain
# has no twin yet (KNOWN_UNPAIRED in tests/test_r_python_parity.py).
_USAGE_SECTIONS = (
    ("player_usage", "player_usage", "players"),
    ("position_group_usage", "position_group_usage", "position_groups"),
    ("tackles", "tackles", "tackles"),
    ("position_group_tackles", "position_group_tackles", "position_group_tackles"),
    ("team_usage", "team_usage", "teams"),
    ("drive_scripting", "drive_scripting", "drive_scripting"),
    ("st_kickers", "st_kickers", "st_kickers"),
    ("st_punters", "st_punters", "st_punters"),
    ("st_returners", "st_returners", "st_returners"),
    ("st_blocks", "st_blocks", "st_blocks"),
    ("st_team", "st_team", "st_team"),
)
USAGE_ADV_ORDER: list[str] = [f"adv_{k}" for _, k, _ in _USAGE_SECTIONS]
USAGE_LEADERBOARD_ORDER: list[str] = [f"usage_{k}" for _, _, k in _USAGE_SECTIONS]

# Team then coach seasons (shims 61-62), then careers (63), which reads the
# written coach seasons. Coach attribution is per team-season from
# ``data/cfb_coach_seasons.csv`` (cfb_data_build.coaches).
TENDENCIES_ORDER: list[str] = ["team_tendencies", "coach_tendencies", "coach_careers"]

# The team-summaries family builds from the RELEASED espn_cfb_pbp (not
# final.json), so neither ``block`` nor ``reshaper`` is set — these specs only
# feed ``write_dataset`` / ``publish_dataset``. Mirrors the R script 15 tuples.
SUMMARIES_REGISTRY: dict[str, DatasetSpec] = {
    "percentiles": DatasetSpec(
        "percentiles", "cfb_percentiles", "espn_cfb_percentiles"
    ),
    "team_summaries": DatasetSpec(
        "team_summaries", "cfb_team_summaries", "espn_cfb_team_summaries"
    ),
    "passing": DatasetSpec("passing", "cfb_passing", "espn_cfb_passing"),
    "rushing": DatasetSpec("rushing", "cfb_rushing", "espn_cfb_rushing"),
    "receiving": DatasetSpec("receiving", "cfb_receiving", "espn_cfb_receiving"),
}


# --- release sidecar metadata -------------------------------------------------
# Every published tag carries package_function.txt/.json naming the loader a
# consumer reaches the data through -- the half of R's sportsdataverse_save()
# the Python publisher used to drop. Values are NOT invented: where the R
# producer already published a package_function to the tag, that exact string
# is reused, so re-stamping from Python does not change what a consumer sees.
# Python-only tags that never had one name the sdv-py loader instead.
#
# Keyed by tag, not dataset -- several datasets can share one tag. BOTH
# registries publish through the same publish_dataset(), so both are covered;
# the publish tests assert every tag in either one has an entry, so a new
# dataset cannot ship an unnamed tag.
PKG_FUNCTION: dict[str, str] = {
    # no reader in any package: espn_cfb_injuries has no cfbfastR loader and is
    # absent from sdv-py's releases.yaml, so it names its producer stage (the
    # same convention the ncaa_*_rapm tags already carry on their sidecars).
    "espn_cfb_injuries": "R/espn_cfb_14_injuries_creation.R",
    "espn_cfb_passing": "sportsdataverse.cfb.load_cfb_passing()",
    "espn_cfb_percentiles": "sportsdataverse.cfb.load_cfb_percentiles()",
    "espn_cfb_receiving": "sportsdataverse.cfb.load_cfb_receiving()",
    "espn_cfb_rushing": "sportsdataverse.cfb.load_cfb_rushing()",
    "espn_cfb_schedules": "cfbfastR::load_cfb_schedules()",
    "espn_cfb_team_summaries": "sportsdataverse.cfb.load_cfb_team_summaries()",
    "espn_cfb_adv_defensive": "sportsdataverse.cfb.load_cfb_adv_defensive()",
    "espn_cfb_adv_defensive_players": "sportsdataverse.cfb.load_cfb_adv_defensive_players()",
    "espn_cfb_adv_drives": "sportsdataverse.cfb.load_cfb_adv_drives()",
    "espn_cfb_adv_passing": "sportsdataverse.cfb.load_cfb_adv_passing()",
    "espn_cfb_adv_receiving": "sportsdataverse.cfb.load_cfb_adv_receiving()",
    "espn_cfb_adv_rushing": "sportsdataverse.cfb.load_cfb_adv_rushing()",
    "espn_cfb_adv_situational": "sportsdataverse.cfb.load_cfb_adv_situational()",
    "espn_cfb_adv_specialists": "sportsdataverse.cfb.load_cfb_adv_specialists()",
    "espn_cfb_adv_team": "sportsdataverse.cfb.load_cfb_adv_team()",
    "espn_cfb_adv_turnover": "sportsdataverse.cfb.load_cfb_adv_turnover()",
    "espn_cfb_betting": "sportsdataverse.cfb.load_cfb_betting()",
    "espn_cfb_drives": "sportsdataverse.cfb.load_cfb_drives()",
    "espn_cfb_game_rosters": "sportsdataverse.cfb.load_cfb_game_rosters()",
    "espn_cfb_linescores": "sportsdataverse.cfb.load_cfb_linescores()",
    "espn_cfb_pbp": "cfbfastR::load_cfb_pbp()",
    "espn_cfb_play_participants": "sportsdataverse.cfb.load_cfb_play_participants()",
    "espn_cfb_player_box": "sportsdataverse.cfb.load_cfb_player_box()",
    "espn_cfb_power_index": "sportsdataverse.cfb.load_cfb_power_index()",
    "espn_cfb_rosters": "cfbfastR::load_cfb_rosters()",
    "espn_cfb_team_box": "sportsdataverse.cfb.load_cfb_team_box()",
    # usage box + leaderboards: no sdv-py loader yet (tracked follow-up), so
    # each tag names its producer stage, as espn_cfb_injuries does.
    **{
        f"espn_cfb_adv_{k}": f"python/espn_cfb_{30 + i}_adv_{k}_creation.py"
        for i, (_, k, _) in enumerate(_USAGE_SECTIONS)
    },
    **{
        f"espn_cfb_usage_{k}": f"python/espn_cfb_{50 + i}_usage_{k}_creation.py"
        for i, (_, _, k) in enumerate(_USAGE_SECTIONS)
    },
    # tendencies: same convention (no loader yet), shims 61-63
    "espn_cfb_team_tendencies": "python/espn_cfb_61_team_tendencies_creation.py",
    "espn_cfb_coach_tendencies": "python/espn_cfb_62_coach_tendencies_creation.py",
    "espn_cfb_coach_careers": "python/espn_cfb_63_coach_careers_creation.py",
}
