"""``cfb_defense_vs_position``: what each defense allowed to QBs, RBs, WRs and TEs.

One row per ``(season, team_id, position_group)`` for the DEFENSE (stage 66),
from one season of the committed ``cfb/pbp`` plus that season's ``cfb_rosters``
and ``cfb_schedules``. The rows are ``sportsdataverse.defense_vs_position``'s,
passed through unchanged; the producer adds only what it owns:

* ``team_id`` as Int64: an integer parse of the function's text id, never
  through a float. A float id in the pbp, the roster or the schedule raises.
* ``pos_team`` / ``division`` / ``conference``, the names the summaries tables
  carry, read from the unified schedule (every defense has them, FBS or not).
* ``unattributed_target_share``: of the passes thrown against the defense
  (dropbacks that are not sacks), the share on which ESPN names NO receiver --
  about 31% in 2024: most incompletions and every interception. Those throws
  reach the QB row and no WR or TE row, so the WR and TE rates are **on targets
  naming a receiver**, which are mostly completions, and they read high. The
  share is repeated on the defense's WR and TE rows and null on QB and RB.
  Nothing is imputed.
* ``<metric>_pct``: the percentile among FBS qualifiers (``division == "fbs"``
  and ``qualified``, the 3-game floor) within ``(season, position_group)``,
  0-100 and oriented so that HIGHER = BETTER DEFENSE for every metric. A
  non-qualifier, a non-FBS defense, a null metric and a cohort under
  ``MIN_COHORT_TEAMS`` get null, and none of them counts toward ``n``.

Direction of each metric (which end is the better defense):

====================================  ==============  =========================
metric                                better defense  a high ``_pct`` means
====================================  ==============  =========================
``epa_per_play_allowed``              LOWER           allowed less EPA per play
``success_rate_allowed``              LOWER           allowed fewer successes
``explosive_rate_allowed``            LOWER           allowed fewer explosives
``rush_yards_per_carry_allowed`` RB   LOWER           allowed fewer yards/carry
``yards_per_target_allowed`` WR, TE   LOWER           allowed fewer yards/target
``sack_rate_allowed`` QB              HIGHER          sacked the QB more often
====================================  ==============  =========================

``sack_rate_allowed`` is sacks per dropback the defense GOT, so it is the one
reversed column. The percentile is the F5 cohort helper's Weibull position,
``100 * (n + 1 - rank) / (n + 1)`` with rank 1 the best defense.

Floor: ``FLOOR`` (2014). The 2004-2013 rosters list nearly every player with no
position, so those seasons would publish QB rows and little else; they are not
built.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from sportsdataverse.defense_vs_position import OUTPUT_SCHEMA as _T1_SCHEMA

# ponytail: _plays is sdv-py's private population filter; importing it keeps the
# share on exactly the plays the rows count. uv.lock pins the commit; if sdv-py
# renames it this import fails at build time -- ask for a public name then.
from sportsdataverse.defense_vs_position import (
    PBP_COLUMNS,
    _plays,
    defense_vs_position,
)

from cfb_data_build.team_summaries import MIN_COHORT_TEAMS, _attach_cohort_percentiles

#: first season with usable roster positions; earlier seasons are not built
FLOOR = 2014

#: metric -> True when a HIGHER value is the better defense (see the module table)
PCT_METRICS: dict[str, bool] = {
    "epa_per_play_allowed": False,
    "success_rate_allowed": False,
    "explosive_rate_allowed": False,
    "sack_rate_allowed": True,
    "rush_yards_per_carry_allowed": False,
    "yards_per_target_allowed": False,
}

_KEYS = ("season", "team_id")
_TEAM_COLS = ("pos_team", "division", "conference")
_ROSTER_COLS = ("season", "athlete_id", "position", "position_abbreviation")
_SCHEDULE_COLS = tuple(
    f"{side}_{c}"
    for side in ("home", "away")
    for c in ("id", "team", "division", "conference")
)

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64(),
    "team_id": pl.Int64(),
    **{c: pl.Utf8() for c in _TEAM_COLS},
    **{c: t for c, t in _T1_SCHEMA.items() if c not in _KEYS},
    "unattributed_target_share": pl.Float64(),
    **{f"{m}_pct": pl.Float64() for m in PCT_METRICS},
}


def _teams(schedule: pl.DataFrame) -> pl.DataFrame:
    """``team_id -> (pos_team, division, conference)``, one row per team.

    A team's rows agree except on ESPN-only games (cancelled, postponed), which
    carry a long name and no division, so the classified row seen most wins.
    """
    sides = []
    for side in ("home", "away"):
        if not schedule.schema[f"{side}_id"].is_integer():
            raise TypeError(
                f"schedule {side}_id is {schedule.schema[f'{side}_id']}: "
                "team ids join as integers, never through a float"
            )
        sides.append(
            schedule.select(
                team_id=pl.col(f"{side}_id").cast(pl.Int64),
                **{c: pl.col(f"{side}_{c.removeprefix('pos_')}") for c in _TEAM_COLS},
            )
        )
    return (
        pl.concat(sides)
        .drop_nulls("team_id")
        .group_by("team_id", *_TEAM_COLS)
        .len()
        .sort(
            pl.col("division").is_null(),
            "len",
            "pos_team",
            descending=[False, True, False],
        )
        .unique(subset="team_id", keep="first", maintain_order=True)
        .drop("len")
    )


def _unattributed_target_share(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per ``(season, team_id)``: the share of throws faced that name no receiver."""
    throws = _plays(pbp, "cfb").filter(
        (pl.col("dropback") == True) & (pl.col("sack") == False)  # noqa: E712
    )
    return throws.group_by(_KEYS).agg(
        unattributed_target_share=pl.col("receiver_id").is_null().mean()
    )


def _attach_percentiles(df: pl.DataFrame) -> pl.DataFrame:
    """``<metric>_pct`` among FBS qualifiers within ``position_group`` (one season)."""
    eligible = (pl.col("qualified") == True) & (pl.col("division") == "fbs")  # noqa: E712
    ranks = {
        f"{m}_rank": pl.when(eligible)
        .then(pl.col(m))
        .rank(method="average", descending=higher_is_better)
        .over("position_group")
        for m, higher_is_better in PCT_METRICS.items()
    }
    return _attach_cohort_percentiles(
        df.with_columns(**ranks),
        cohort="position_group",
        rank_cols=list(ranks),
        suffix="_pct",
        min_cohort=MIN_COHORT_TEAMS,
    ).drop(list(ranks))


def defense_vs_position_table(
    pbp: pl.DataFrame, rosters: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    """The published table for ONE season's pbp, rosters and unified schedule.

    Raises ``TypeError`` on a float team, game or player id (pbp, roster or
    schedule) rather than joining through ``"213.0"``, and ``ValueError`` when a
    defense in the pbp has no row in the schedule.
    """
    rows = defense_vs_position(pbp, rosters, "cfb")
    if rows.height == 0:
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    # Utf8 -> Int64 is an integer parse: strict, so a non-integer id raises
    as_int = pl.col("team_id").cast(pl.Int64)
    rows = rows.with_columns(as_int)
    share = _unattributed_target_share(pbp).with_columns(as_int)
    teams = _teams(schedule)
    unmatched = sorted(set(rows["team_id"]) - set(teams["team_id"]))
    if unmatched:
        # a null division would drop the defense from the FBS cohort without a
        # word and move every other defense's percentile
        raise ValueError(
            f"{len(unmatched)} defense team_id(s) have no schedule row: {unmatched[:10]}"
        )
    for right in (teams, share):
        assert rows.schema["team_id"] == right.schema["team_id"], (
            f"team_id {rows.schema['team_id']} != {right.schema['team_id']}"
        )
    out = (
        rows.join(teams, on="team_id", how="left", validate="m:1")
        .join(share, on=_KEYS, how="left", validate="m:1")
        .with_columns(
            unattributed_target_share=pl.when(
                pl.col("position_group").is_in(["WR", "TE"])
            ).then(pl.col("unattributed_target_share"))
        )
    )
    out = (
        _attach_percentiles(out)
        .select(list(OUTPUT_SCHEMA))
        .sort(*_KEYS, "position_group")
    )
    assert out.schema == pl.Schema(OUTPUT_SCHEMA), out.schema
    return out


def build_defense_vs_position(season: int, *, base: str = "cfb") -> pl.DataFrame:
    """``cfb_defense_vs_position`` for ``season`` from the tree under ``base``.

    Reads ``pbp``, ``cfb_rosters`` and ``cfb_schedules`` for the season, all of
    them this run's builds (no release fallback, so no network). Returns the
    empty ``OUTPUT_SCHEMA`` frame, and prints why, before ``FLOOR`` or when the
    season's pbp is not built; a missing roster or schedule beside a built pbp
    raises.
    """
    root = Path(base)
    pbp_path = root / "pbp" / "parquet" / f"play_by_play_{season}.parquet"
    if season < FLOOR:
        why = f"before {FLOOR} cfb_rosters carry no positions"
    elif not pbp_path.is_file():
        why = f"no {pbp_path}"
    else:
        why = None
    if why:
        print(f"  defense_vs_position {season}: not built, {why}", flush=True)
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    pbp = pl.read_parquet(pbp_path, columns=list(PBP_COLUMNS["cfb"]))
    stray = sorted(set(pbp["season"].unique()) - {season})
    if stray:
        # the percentile cohort is ONE season's position group
        raise ValueError(f"{pbp_path} holds season(s) {stray}, expected {season}")
    return defense_vs_position_table(
        pbp,
        pl.read_parquet(
            root / "cfb_rosters" / "parquet" / f"cfb_rosters_{season}.parquet",
            columns=list(_ROSTER_COLS),
        ),
        pl.read_parquet(
            root / "cfb_schedules" / "parquet" / f"cfb_schedules_{season}.parquet",
            columns=list(_SCHEDULE_COLS),
        ),
    )
