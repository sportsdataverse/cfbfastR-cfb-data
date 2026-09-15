"""Reference inputs for the matchup line's remaining side-input columns.

Four tables under ``data/`` feed the eleven per-side columns CFBD does not
publish. Two are DERIVED here and rebuilt by ``refresh`` (so they carry every
season their source supports, not just the current one); two are external
values that can only be imported and versioned:

============================ ========= =========================================
table                        seasons   how it is produced
============================ ========= =========================================
cfb_matchup_coordinators     2013-     IMPORTED. Offensive / defensive
                                       coordinator names per school-season; the
                                       public coaching lists they come from are
                                       not an API. Refresh: append the new
                                       season's rows before August.
cfb_matchup_returning_        2015-    IMPORTED. Returning-production shares.
production                             NOT CFBD's ``/player/returning`` -- that
                                       is a different metric (measured against
                                       this table for 2025: r = 0.75 offense,
                                       0.88 defense, no exact agreement), and
                                       the SDV ``cfb_returning_production``
                                       release repeats its offense value in
                                       ``overall_returning``. Do not "fix" this
                                       by swapping the source.
cfb_matchup_roster_talent    2015-     IMPORTED. Rank-decayed roster talent sum
                                       from a recruiting service, distinct from
                                       CFBD ``/talent``'s composite.
cfb_matchup_coach_continuity 2014-     DERIVED from the coordinators table by
                                       :func:`coach_continuity`.
cfb_matchup_qb_starters      2004-     DERIVED from the ESPN play-by-play
                                       release: the passer with the most
                                       attempts (:func:`starters_from_pbp`).
============================ ========= =========================================

Team keys: the coordinators table and the QB table are keyed by the mascot-form
school name (the pbp release's ``pos_team``), and resolve to CFBD's ``school``
through the ``join_name`` that ``matchup_side.tidy_cfbd_teams`` builds. The two
imported value tables are keyed by a bare school name. Both paths go through
:func:`match_team_names`, which reports the spellings it could not resolve
rather than dropping them silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parents[2] / "data"

COORDINATORS_CSV = DATA / "cfb_matchup_coordinators.csv"
RETURNING_CSV = DATA / "cfb_matchup_returning_production.csv"
TALENT_CSV = DATA / "cfb_matchup_roster_talent.csv"
CONTINUITY_CSV = DATA / "cfb_matchup_coach_continuity.csv"
QB_CSV = DATA / "cfb_matchup_qb_starters.csv"

#: Irregular spellings the contracting normalizer cannot reach, seeded from the
#: unmatched report of a build (``match_team_names`` returns it). Keys and
#: values are contracted keys, mapping the reference table's spelling onto
#: CFBD's own school name.
TEAM_ALIASES: dict[str, str] = {
    "appalachianstate": "appstate",
    "bowlinggreen": "bowlinggreenstate",
    "centralflorida": "ucf",
    "connecticut": "uconn",
    "louisianalafayette": "louisiana",
    "louisianamonroe": "ulmonroe",
    "massachusetts": "umass",
    "miamifl": "miami",
    "miamioh": "miamiohio",
    "middletennesseestate": "middletennessee",
    "nevadalasvegas": "unlv",
    "olemiss": "mississippi",
    "southernmississippi": "southernmiss",
    "texassanantonio": "utsa",
    "utsanantonio": "utsa",
}

#: Mascot-form school names in the coordinators table whose CFBD counterpart
#: differs by more than punctuation (same source: the first build's report).
MASCOT_ALIASES: dict[str, str] = {
    "appalachianstatemountaineers": "appstatemountaineers",
    "delawarefightinbluehens": "delawarebluehens",
    "floridainternationalgoldenpanthers": "floridainternationalpanthers",
    "louisianamonroewarhawks": "ulmonroewarhawks",
    "southernmississippigoldeneagles": "southernmissgoldeneagles",
    "umassminutemen": "massachusettsminutemen",
}


def _key(expr: pl.Expr) -> pl.Expr:
    """Contracting match key: lowercase, letters and digits only."""
    return expr.str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")


def _candidates(name: str, *, aliases: dict[str, str] | None = None) -> list[str]:
    """The contracted key plus the systematic shorthands these lists use."""
    table = TEAM_ALIASES if aliases is None else aliases
    k = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    out = [k, table.get(k, k)]
    if k.endswith("st"):
        out += [k[:-2] + "state", k[:-2]]
    if k.startswith("calst"):
        out.append("calstate" + k[5:])
    if k.endswith("state"):
        out.append(k[: -len("state")])
    return list(dict.fromkeys(out))


def match_team_names(
    frame: pl.DataFrame,
    schools: list[str],
    *,
    column: str = "team",
    aliases: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Re-key ``frame`` onto CFBD ``schools``; returns (matched, unmatched names).

    Unmatched rows are DROPPED and returned so the caller can assert a match
    rate instead of silently losing a school.
    """
    lookup: dict[str, str] = {}
    for school in schools:
        for c in _candidates(school, aliases=aliases):
            lookup.setdefault(c, school)
    mapped, missing = [], []
    for name in frame[column].unique().sort().to_list():
        cands = _candidates(name, aliases=aliases)
        rank = next((i for i, c in enumerate(cands) if c in lookup), None)
        if rank is None:
            missing.append(name)
        else:
            mapped.append({column: name, "_school": lookup[cands[rank]], "_rank": rank})
    # two spellings of one school can both resolve (a table that carries both
    # "Connecticut" and "UConn"); keep the one that matched on its own key
    # rather than through an alias, so the join can never fan the line out
    if mapped:
        best = (
            pl.DataFrame(mapped)
            .sort(["_school", "_rank", column])
            .unique(subset=["_school"], keep="first", maintain_order=True)
            .drop("_rank")
        )
        mapped = best.to_dicts()
    if not mapped:
        return frame.head(0).with_columns(pl.lit(None, pl.Utf8).alias("team")), missing
    out = (
        frame.join(pl.DataFrame(mapped), on=column, how="inner")
        .drop(column)
        .rename({"_school": "team"})
    )
    return out, missing


def _norm_coach(expr: pl.Expr) -> pl.Expr:
    """Coordinator name normalized for EQUALITY only; blank becomes null."""
    k = expr.str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")
    return pl.when(expr.is_null() | (k == "")).then(None).otherwise(k)


def coach_continuity(coordinators: pl.DataFrame) -> pl.DataFrame:
    """Per school-season: were last season's coordinators retained?

    Co-coordinators are split on ``/``; a side counts as continuous when ANY of
    this season's names matches ANY of last season's (the comparison is on the
    normalized name, so punctuation drift between sources is not a false
    change). A missing name never matches, and the flag is 0 rather than null
    so a model never sees a null continuity. The first season of the source
    table has no predecessor and is dropped.
    """
    parts = coordinators.with_columns(
        [
            _norm_coach(
                pl.col(f"{side}_coordinators")
                .str.split("/")
                .list.get(i, null_on_oob=True)
            ).alias(f"{abbr}{i + 1}")
            for side, abbr in (("offensive", "oc"), ("defensive", "dc"))
            for i in (0, 1)
        ]
    ).sort(["school_mascot", "season"])
    lagged = parts.with_columns(
        [
            pl.col(c).shift(1).over("school_mascot").alias(f"lag_{c}")
            for c in ("oc1", "oc2", "dc1", "dc2")
        ]
    )
    flags = []
    for abbr in ("oc", "dc"):
        hit = pl.lit(False)  # noqa: FBT003 - polars literal, not a flag argument
        for a in (f"{abbr}1", f"{abbr}2"):
            for b in (f"lag_{abbr}1", f"lag_{abbr}2"):
                hit = hit | (pl.col(a) == pl.col(b)).fill_null(False)  # noqa: FBT003
        flags.append(hit.cast(pl.Int64).alias(f"{abbr}_cont"))
    first = coordinators["season"].min()
    return (
        lagged.with_columns(flags)
        .filter(pl.col("season") > first)
        .select("season", "school_mascot", "oc_cont", "dc_cont")
        .sort(["season", "school_mascot"])
    )


def starters_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per (season, team): the season's starting quarterback, from the pbp release.

    The starter is the passer with the most pass plays. The ESPN play-by-play
    release carries ``passer_player_id`` on essentially every pass play from
    2004 -- further back than CFBD ``/player/usage`` (2013), and in the same
    athlete-id namespace the line already uses.
    """
    return (
        pbp.filter((pl.col("pass") == 1) & pl.col("passer_player_id").is_not_null())
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("pos_team").alias("team"),
            pl.col("game_id"),
            pl.col("passer_player_name").alias("qb_name"),
            pl.col("passer_player_id").cast(pl.Int64).alias("athlete_id"),
        )
        .group_by("season", "team", "athlete_id")
        .agg(
            pl.len().alias("attempts"),
            pl.col("qb_name").drop_nulls().first().alias("qb_name"),
        )
        .sort(
            ["season", "team", "attempts", "athlete_id"],
            descending=[False, False, True, False],
        )
        .unique(subset=["season", "team"], keep="first", maintain_order=True)
        .select("season", "team", "qb_name", "athlete_id")
    )


def qb_game_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per (season, athlete): distinct games the passer threw in."""
    return (
        pbp.filter((pl.col("pass") == 1) & pl.col("passer_player_id").is_not_null())
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("game_id"),
            pl.col("passer_player_id").cast(pl.Int64).alias("athlete_id"),
        )
        .unique()
        .group_by("season", "athlete_id")
        .agg(pl.col("game_id").n_unique().alias("games"))
    )


def qb_starters(starters: pl.DataFrame, appearances: pl.DataFrame) -> pl.DataFrame:
    """The full QB block per (season, team), derived from the starter series.

    * ``returning_qb`` -- the same athlete started for the same team last season
    * ``qb_starter_years`` -- how many earlier seasons this athlete was a
      starter anywhere (0 for a first-time starter)
    * ``qb_games`` -- games ENTERING the season in which the athlete threw an
      incompletion. The delivered line carries one value per team for the whole
      season, so this is a career-to-date count, not an as-of-date one.
    """
    s = starters.sort(["athlete_id", "season"])
    prior_years = (
        s.with_columns(pl.col("season").cum_count().over("athlete_id").alias("n"))
        .with_columns((pl.col("n") - 1).alias("qb_starter_years"))
        .select("season", "team", "athlete_id", "qb_starter_years")
    )
    prev = s.select(
        (pl.col("season") + 1).alias("season"),
        pl.col("team"),
        pl.col("athlete_id").alias("prev_athlete_id"),
    )
    # games entering the season: sum the athlete's appearances in EVERY earlier
    # season, not only the seasons he has an appearance row for
    career = (
        s.select("season", "team", "athlete_id")
        .join(
            appearances.rename({"season": "appeared_in"}), on="athlete_id", how="left"
        )
        .filter(pl.col("appeared_in") < pl.col("season"))
        .group_by("season", "team", "athlete_id")
        .agg(pl.col("games").sum().alias("prior_games"))
    )
    # a starter with no earlier appearance row has entered with zero games
    return (
        s.join(prior_years, on=["season", "team", "athlete_id"], how="left")
        .join(prev, on=["season", "team"], how="left")
        .join(career, on=["season", "team", "athlete_id"], how="left")
        .with_columns(
            returning_qb=(pl.col("athlete_id") == pl.col("prev_athlete_id"))
            .fill_null(False)  # noqa: FBT003
            .cast(pl.Int64),
            qb_games=pl.col("prior_games").fill_null(0).cast(pl.Int64),
        )
        .select(
            "season",
            "team",
            "qb_name",
            "athlete_id",
            "returning_qb",
            "qb_starter_years",
            "qb_games",
        )
        .sort(["season", "team"])
    )


def _read(path: Path, what: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{what}: {path} is missing -- rebuild it with "
            f"`python -m cfb_data_build.matchup_reference refresh`"
        )
    return pl.read_csv(path, null_values=["NA", ""])


def load_coordinators() -> pl.DataFrame:
    return _read(COORDINATORS_CSV, "coordinators")


def load_returning_production() -> pl.DataFrame:
    return _read(RETURNING_CSV, "returning production")


def load_roster_talent() -> pl.DataFrame:
    return _read(TALENT_CSV, "roster talent")


def load_coach_continuity() -> pl.DataFrame:
    return _read(CONTINUITY_CSV, "coach continuity")


def load_qb_starters() -> pl.DataFrame:
    return _read(QB_CSV, "QB starters")


def _one_row_per_team(frame: pl.DataFrame, what: str, season: int) -> None:
    """A reference block must be unique on ``team``: a join is what fills the line."""
    dupes = frame.group_by("team").len().filter(pl.col("len") > 1)["team"].to_list()
    if dupes:
        raise ValueError(f"{what} {season}: more than one row for {sorted(dupes)}")


def season_reference(
    season: int, *, teams: pl.DataFrame, schools: list[str]
) -> pl.DataFrame:
    """The eleven non-CFBD columns for one season, keyed by CFBD ``team``.

    ``teams`` is :func:`~cfb_data_build.matchup_side.tidy_cfbd_teams` output
    (it carries ``join_name``, which resolves the coordinators table's
    mascot-form school). Every column is present even when a table has no rows
    for the season, so the line's schema never depends on the data.
    """
    out = pl.DataFrame({"team": schools}, schema={"team": pl.Utf8})

    cont = load_coach_continuity().filter(pl.col("season") == season)
    cont, _unmatched_mascots = match_team_names(
        cont,
        teams["join_name"].to_list(),
        column="school_mascot",
        aliases=MASCOT_ALIASES,
    )
    cont = (
        cont.rename({"team": "join_name"})
        .join(teams.select("join_name", "team"), on="join_name", how="inner")
        .select("team", "oc_cont", "dc_cont")
    )
    _one_row_per_team(cont, "coach continuity", season)
    out = out.join(cont, on="team", how="left")

    rp, _ = match_team_names(
        load_returning_production().filter(pl.col("season") == season), schools
    )
    rp = rp.with_columns(
        ovr_rtprod=((pl.col("off_rtprod") + pl.col("def_rtprod")) / 2).round(2)
    ).select("team", "off_rtprod", "def_rtprod", "ovr_rtprod")
    _one_row_per_team(rp, "returning production", season)
    out = out.join(rp, on="team", how="left")

    tl, _ = match_team_names(
        load_roster_talent().filter(pl.col("season") == season), schools
    )
    _one_row_per_team(tl, "roster talent", season)
    out = out.join(tl.select("team", "team_talent_weighted"), on="team", how="left")

    # the pbp release keys teams by the mascot form, as the coordinators table does
    qb, _ = match_team_names(
        load_qb_starters().filter(pl.col("season") == season),
        teams["join_name"].to_list(),
        aliases=MASCOT_ALIASES,
    )
    qb = (
        qb.rename({"team": "join_name"})
        .join(teams.select("join_name", "team"), on="join_name", how="inner")
        .drop("join_name")
    )
    _one_row_per_team(qb, "QB starters", season)
    out = out.join(
        qb.select(
            "team",
            "qb_name",
            "athlete_id",
            "returning_qb",
            "qb_starter_years",
            "qb_games",
        ),
        on="team",
        how="left",
    )
    return out


# --- refresh -----------------------------------------------------------------
#: the ESPN play-by-play release's first season (it carries a passer id from
#: the start); the coordinators table's own first season governs the
#: continuity table, whose first season has no predecessor
PBP_FIRST_SEASON = 2004


def build_qb_table(seasons: list[int], *, pbp_loader=None) -> pl.DataFrame:
    """Rebuild the QB starter table for ``seasons`` from the ESPN pbp release."""
    if pbp_loader is None:
        pbp_loader = _load_pbp_columns
    starters, counts = [], []
    for season in seasons:
        frame = pbp_loader(season)
        starters.append(starters_from_pbp(frame))
        counts.append(qb_game_counts(frame))
    return qb_starters(
        pl.concat(starters, how="vertical"), pl.concat(counts, how="vertical")
    )


def _load_pbp_columns(season: int) -> pl.DataFrame:
    """Only the columns the QB derivation reads, from the ESPN pbp release."""
    from sportsdataverse.cfb import load_cfb_pbp

    return load_cfb_pbp([season]).select(
        "season",
        "game_id",
        "pos_team",
        "pass",
        "passer_player_name",
        "passer_player_id",
    )


def main(argv: list[str] | None = None) -> int:
    """``refresh``: rebuild the two derived reference tables in place."""
    import argparse

    p = argparse.ArgumentParser(
        prog="cfb_data_build.matchup_reference",
        description="rebuild data/cfb_matchup_{coach_continuity,qb_starters}.csv",
    )
    sub = p.add_subparsers(dest="command", required=False)
    r = sub.add_parser("refresh", help="rebuild both derived tables (default)")
    r.add_argument("--end-season", type=int, required=True)
    r.add_argument("--start-season", type=int, default=PBP_FIRST_SEASON)
    r.add_argument(
        "--only",
        choices=("continuity", "qb"),
        help="rebuild just one table (qb needs the pbp release)",
    )
    args = p.parse_args(argv)
    if args.command is None:
        p.print_help()
        return 0

    if args.only != "qb":
        cont = coach_continuity(load_coordinators())
        cont.write_csv(CONTINUITY_CSV)
        print(
            f"{CONTINUITY_CSV.name}: {cont.height} rows, "
            f"{cont['season'].min()}-{cont['season'].max()}"
        )
    if args.only != "continuity":
        seasons = list(range(args.start_season, args.end_season + 1))
        qb = build_qb_table(seasons)
        qb.write_csv(QB_CSV)
        print(
            f"{QB_CSV.name}: {qb.height} rows, {qb['season'].min()}-{qb['season'].max()}, "
            f"{qb['athlete_id'].null_count()} without an id"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
