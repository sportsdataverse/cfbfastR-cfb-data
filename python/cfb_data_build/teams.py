"""``cfb_teams`` -- one tidy row per (season, team) with conference + branding.

Reads the season bundles the ``cfbfastR-cfb-raw`` scraper
(``python/scrape_cfb_teams.py``) commits at
``cfb/teams/json/{season}.json`` **over HTTP** from raw.githubusercontent.com,
the same way ``R/_data_utils.R`` reads ``final`` JSON. There is deliberately no
local-filesystem path: the raw repo is the published input, so a build here can
never silently depend on a sibling checkout being fresh.

The bundle is a container of verbatim ESPN payloads; every tidying decision
lives here.

Grain: one row per (season, team_id). ESPN publishes team identity per season,
so a team's conference, colors, and logos are as-of that season -- this is not a
static dimension table and 2013 Maryland is correctly Big-10-less.

Two ESPN facts the compile depends on:

* **Division comes from group membership, not from the team payload.** A team's
  own payload carries no division field; the group whose team list contains it is
  the only signal. The tree is
  ``99 -> {90 -> {80 FBS, 81 FCS}, 35 -> {57 D-II, 58 D-III}, 36 All Star}``,
  plus group 186 (NAIA), a PARENTLESS sibling of 99 that no tree walk reaches.
* **A team's ``groups`` ``$ref`` points at season type 3** while the group
  children live under type 2. Only the group *id* is comparable across the two,
  so the conference join is on the id parsed out of the ref.

Filtering to FBS is ``pl.col("is_fbs")`` -- a non-null Boolean, so a team ESPN
files in no group at all reads False rather than poisoning the mask with a null.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
from typing import Any

import polars as pl

from cfb_data_build.config import DatasetSpec
from cfb_data_build.io import write_dataset

RAW_BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; sportsdataverse/cfb-data)"}
_ID_RE = re.compile(r"/groups/(\d+)")

SPEC = DatasetSpec("cfb_teams", "cfb_teams", "espn_cfb_teams")
SPECS: dict[str, DatasetSpec] = {"teams": SPEC}

#: Released per-season CFBD team_info (``cfbd_team_info(year, only_fbs = FALSE)``,
#: produced by ``cfbfastR-data/add_team_info.R``). Read over HTTP for the same
#: reason the ESPN bundle is: the published artifact is the input, so a build can
#: never silently depend on a sibling checkout being fresh.
TEAM_INFO_BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_team_info"

#: CFBD-only fields backported onto the ESPN row, as ``cfbd name -> output name``.
#: ONLY fields ESPN does not already ship: the 5 shared ones (``abbreviation``,
#: ``color``, ``venue_id``, ``venue_name``) keep ESPN's value, and CFBD's
#: ``conference`` lands as ``cfbd_conference`` so it cannot be mistaken for a
#: member of the ESPN ``conference_*`` family (they disagree -- ESPN resolves a
#: team to its conference GROUP, CFBD to its own conference string).
#:
#: ``classification`` deliberately coexists with ESPN's ``division``/``is_fbs``
#: rather than replacing them: it is a second, independent opinion on the same
#: question, sourced from a different provider, and collapsing the two would
#: destroy the disagreement that makes it worth having.
CFBD_FIELDS: dict[str, str] = {
    "school": "school",
    "mascot": "mascot",
    "alt_name1": "alt_name1",
    "alt_name2": "alt_name2",
    "alt_name3": "alt_name3",
    "conference": "cfbd_conference",
    "classification": "classification",
    "city": "city",
    "state": "state",
    "country_code": "country_code",
    "timezone": "timezone",
    "latitude": "latitude",
    "longitude": "longitude",
    "elevation": "elevation",
    "capacity": "capacity",
    "dome": "dome",
    "grass": "grass",
}

#: Declared so a season with no CFBD match (or no team_info asset) still ships the
#: same column set with the same dtypes. ``elevation`` is Utf8 because that is how
#: CFBD serves it -- not silently re-typed here.
CFBD_SCHEMA: dict[str, pl.DataType] = {
    "school": pl.Utf8,
    "mascot": pl.Utf8,
    "alt_name1": pl.Utf8,
    "alt_name2": pl.Utf8,
    "alt_name3": pl.Utf8,
    "cfbd_conference": pl.Utf8,
    "classification": pl.Utf8,
    "city": pl.Utf8,
    "state": pl.Utf8,
    "country_code": pl.Utf8,
    "timezone": pl.Utf8,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "elevation": pl.Utf8,
    "capacity": pl.Int64,
    "dome": pl.Boolean,
    "grass": pl.Boolean,
}

#: group id -> division label, MOST SPECIFIC FIRST: a team listed under both a
#: leaf and its parent takes the leaf. ``d2_d3`` and ``d1`` are the honest labels
#: for the parent-only case, which is not an edge case -- ESPN files 107 of 2023's
#: 800 teams directly under group 35, and in 2001 groups 57/58 were empty outright
#: so all 195 non-D-I teams landed there.
DIVISIONS = {
    "80": "fbs",
    "81": "fcs",
    "57": "d2",
    "58": "d3",
    "36": "all_star",
    "186": "naia",
    "35": "d2_d3",
    "90": "d1",
}

#: The classification tree itself. A team's ``groups`` ref can point straight at
#: one of these (every group-35 team, and D-II teams like Colorado Mesa whose ref
#: is group 57), and none of them is a conference -- so the walk-up both STOPS
#: here and refuses to report the node it stopped on as a conference.
STRUCTURAL_GROUPS = {99, 90, 80, 81, 35, 57, 58, 36, 186}

#: Declared so an empty season still ships the documented column set rather than
#: a zero-column frame (and so every season's parquet has one schema).
SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "team_id": pl.Int64,
    "uid": pl.Utf8,
    "guid": pl.Utf8,
    "slug": pl.Utf8,
    "abbreviation": pl.Utf8,
    "display_name": pl.Utf8,
    "short_display_name": pl.Utf8,
    "name": pl.Utf8,
    "nickname": pl.Utf8,
    "location": pl.Utf8,
    "color": pl.Utf8,
    "alternate_color": pl.Utf8,
    "is_active": pl.Boolean,
    "is_all_star": pl.Boolean,
    "is_exhibition": pl.Boolean,
    "division": pl.Utf8,
    "is_fbs": pl.Boolean,
    "team_group_id": pl.Int64,
    "team_group_name": pl.Utf8,
    "conference_id": pl.Int64,
    "conference_name": pl.Utf8,
    "conference_short_name": pl.Utf8,
    "conference_abbreviation": pl.Utf8,
    "conference_midsize_name": pl.Utf8,
    "conference_slug": pl.Utf8,
    "conference_is_conference": pl.Boolean,
    "conference_parent_id": pl.Int64,
    "team_logo": pl.Utf8,
    "team_logo_dark": pl.Utf8,
    "conference_logo": pl.Utf8,
    "venue_id": pl.Int64,
    "venue_name": pl.Utf8,
    "venue_city": pl.Utf8,
    "venue_state": pl.Utf8,
    "venue_indoor": pl.Boolean,
    "venue_grass": pl.Boolean,
}


def bundle_url(season: int, raw_base: str = RAW_BASE) -> str:
    return f"{raw_base}/teams/json/{season}.json"


def load_bundle(season: int, *, raw_base: str = RAW_BASE, timeout: int = 60) -> dict:
    """Fetch one season bundle. A missing season raises rather than returning {}."""
    req = urllib.request.Request(bundle_url(season, raw_base), headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _int(value: Any) -> int | None:
    """Int from an ESPN id. ESPN ships ids as numeric strings; never via float.

    Going through float would stringify a joined id as "123.0" and silently
    break every downstream join, so a non-integral value becomes null instead.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _logo(payload: dict, *rels: str) -> str | None:
    """First logo href whose ``rel`` list contains every requested tag."""
    for logo in (payload or {}).get("logos") or []:
        rel = set(logo.get("rel") or [])
        if all(r in rel for r in rels):
            href = logo.get("href")
            if href:
                return str(href)
    return None


def _conference_logo(conf: dict) -> str | None:
    """Conference logos carry no ``rel``, so take the first href."""
    for logo in (conf or {}).get("logos") or []:
        href = logo.get("href")
        if href:
            return str(href)
    return None


def _division_map(bundle: dict) -> dict[str, str]:
    """team id -> division label; the most specific containing group wins.

    ``DIVISIONS`` is ordered leaf-first, so ``setdefault`` resolves a team listed
    under both a leaf and its parent (432 of them in 2023) to the leaf.
    """
    out: dict[str, str] = {}
    groups = bundle.get("divisions") or {}
    for gid, label in DIVISIONS.items():
        for tid in groups.get(gid) or []:
            out.setdefault(str(tid), label)
    return out


def _group_id(ref: str | None) -> int | None:
    """Group id out of a core-v2 ``$ref``.

    The ref's SEASON TYPE differs by source (a team's own groups ref is type 3,
    the 80/81 children are type 2) so only the id is comparable -- never the URL.
    """
    m = _ID_RE.search(ref or "")
    return int(m.group(1)) if m else None


def _resolve_conference(gid: int | None, conferences: dict, *, max_hops: int = 5) -> dict:
    """Walk a team's group up to the CONFERENCE-level group.

    ESPN assigns a team to its DIVISION WITHIN a conference whenever one exists
    -- 2023 Auburn's groups ref is group 7, ``"SEC - West"`` (``isConference:
    false``), not group 8, ``"Southeastern Conference"``. Taking the immediate
    group as the conference silently publishes "SEC - West" as a conference name
    for 14 of 16 SEC teams, and splits every divisioned conference in two. The
    conference is the ancestor whose own parent is a STRUCTURAL group.

    Returns ``{}`` when the walk lands ON a structural group: the D-II/D-III teams
    ESPN files straight under group 35 (and D-II teams whose ref is group 57) have
    no conference at all, and publishing "Division II/III" as their conference
    name would be a fabrication.
    """
    seen: set[int] = set()
    cur = gid
    group: dict = {}
    for _ in range(max_hops):
        if cur is None or cur in seen:
            break
        seen.add(cur)
        nxt = conferences.get(str(cur))
        if not nxt:
            break
        group = nxt
        parent = _group_id((group.get("parent") or {}).get("$ref"))
        if parent is None or parent in STRUCTURAL_GROUPS:
            break
        cur = parent
    if _int(group.get("id")) in STRUCTURAL_GROUPS:
        return {}
    return group


def _team_row(season: int, tid: str, team: dict, division: str | None, conferences: dict) -> dict:
    group_id = _group_id(((team or {}).get("groups") or {}).get("$ref"))
    group = conferences.get(str(group_id)) or {}
    conf = _resolve_conference(group_id, conferences)
    venue = (team or {}).get("venue") or {}
    address = venue.get("address") or {}
    logo = _logo(team, "full", "default")
    # ESPN files bowl all-star and exhibition squads INSIDE the FBS/FCS groups
    # (2023 group 80 carries 12 of them: "SOUTH All-Stars", "Kai Hulakai",
    # "Team Gaither", "TBA", ...), so a naive count of group-80 rows overstates
    # FBS by a dozen programs. ESPN's own isAllStar flag catches only 4 of those
    # 12, so it is not usable as the filter. What separates them cleanly is
    # having neither a conference group nor a logo: across 2001-2026 that rule
    # selects 154 rows and leaves the 10 real-but-groupless programs
    # (Northeastern, Hofstra -- both dropped football in 2009) correctly in.
    # Guarded on the payload existing: a team whose payload failed to fetch also
    # has no conference and no logo, and must not be mislabelled an exhibition.
    #
    # SCOPED TO DIVISION I. Outside FBS/FCS, "no conference and no logo" describes
    # several hundred perfectly real D-II/D-III/NAIA programs per season, so
    # applying the rule there would relabel most of the newly-captured universe as
    # exhibitions. ESPN only files all-star squads inside 80/81 (plus the dedicated
    # group 36, empty on every season captured), so restricting it there keeps the
    # flag's meaning AND leaves every pre-expansion FBS/FCS row unchanged.
    exhibition = division == "all_star" or (division in ("fbs", "fcs") and bool(team) and not conf and logo is None)
    return {
        "season": season,
        "team_id": _int(team.get("id") or tid),
        "uid": team.get("uid"),
        "guid": team.get("guid"),
        "slug": team.get("slug"),
        "abbreviation": team.get("abbreviation"),
        "display_name": team.get("displayName"),
        "short_display_name": team.get("shortDisplayName"),
        "name": team.get("name"),
        "nickname": team.get("nickname"),
        "location": team.get("location"),
        "color": team.get("color"),
        "alternate_color": team.get("alternateColor"),
        "is_active": team.get("isActive"),
        "is_all_star": team.get("isAllStar"),
        "is_exhibition": exhibition,
        "division": division,
        # Built in Python, not as a polars expression: `pl.col("division") == "fbs"`
        # is NULL wherever `division` is null, and a null in a filter mask drops the
        # row silently instead of answering the question. `is_fbs` is always a real
        # Boolean, so `filter(pl.col("is_fbs"))` and `filter(~pl.col("is_fbs"))`
        # partition the season exactly.
        "is_fbs": division == "fbs",
        "team_group_id": group_id,
        "team_group_name": group.get("name"),
        "conference_id": _int(conf.get("id")),
        "conference_name": conf.get("name"),
        "conference_short_name": conf.get("shortName"),
        "conference_abbreviation": conf.get("abbreviation"),
        "conference_midsize_name": conf.get("midsizeName"),
        "conference_slug": conf.get("slug"),
        "conference_is_conference": conf.get("isConference"),
        "conference_parent_id": _group_id((conf.get("parent") or {}).get("$ref")),
        "team_logo": logo,
        "team_logo_dark": _logo(team, "full", "dark"),
        "conference_logo": _conference_logo(conf),
        "venue_id": _int(venue.get("id")),
        "venue_name": venue.get("fullName"),
        "venue_city": address.get("city"),
        "venue_state": address.get("state"),
        "venue_indoor": venue.get("indoor"),
        "venue_grass": venue.get("grass"),
    }


def team_info_url(season: int, base: str = TEAM_INFO_BASE) -> str:
    return f"{base}/cfb_team_info_{season}.parquet"


def load_team_info(season: int, *, base: str = TEAM_INFO_BASE, timeout: int = 60) -> pl.DataFrame:
    """Fetch one season of CFBD team_info. A missing season raises.

    ``team_id`` is widened Int32 -> Int64 HERE, at the boundary, so the join in
    :func:`enrich_cfbd` compares like for like. Widening an integer is lossless;
    the id is never routed through float or stringified to make a join line up.
    """
    req = urllib.request.Request(team_info_url(season, base), headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        df = pl.read_parquet(io.BytesIO(r.read()))
    return df.with_columns(pl.col("team_id").cast(pl.Int64))


def enrich_cfbd(df: pl.DataFrame, team_info: pl.DataFrame | None) -> pl.DataFrame:
    """Left-join the CFBD-only fields onto the ESPN rows. Pure -- no network.

    LEFT, never inner: ESPN's universe is far larger than CFBD's (857 vs 672 in
    2023, the difference almost entirely D-II/D-III/NAIA), so an inner join would
    silently delete two thirds of the dataset. An ESPN team with no CFBD row keeps
    nulls in the added columns and stays.

    Nothing existing is touched -- no ESPN column is overwritten or renamed, and
    ``division`` / ``is_fbs`` / ``is_exhibition`` keep their meaning exactly.
    """
    cols = list(CFBD_SCHEMA)
    if team_info is None or team_info.height == 0:
        return df.with_columns([pl.lit(None, dtype=CFBD_SCHEMA[c]).alias(c) for c in cols])
    right = team_info.select(
        [pl.col("team_id")]
        + [
            pl.col(src).cast(CFBD_SCHEMA[dst]).alias(dst)
            for src, dst in CFBD_FIELDS.items()
            if src in team_info.columns
        ]
    )
    missing = [c for c in cols if c not in right.columns]
    if missing:
        right = right.with_columns([pl.lit(None, dtype=CFBD_SCHEMA[c]).alias(c) for c in missing])
    # A join is only as correct as the dtype agreement on its key, and a duplicate
    # key on the right silently multiplies rows -- assert both rather than discover
    # it as a row-count drift three datasets downstream.
    if df.schema["team_id"] != right.schema["team_id"]:
        raise ValueError(f"team_id dtype mismatch: espn={df.schema['team_id']} cfbd={right.schema['team_id']}")
    if right.height != right["team_id"].n_unique():
        raise ValueError("cfbd team_info has duplicate team_id rows")
    out = df.join(right.select(["team_id"] + cols), on="team_id", how="left")
    if out.height != df.height:
        raise ValueError(f"join changed row count: {df.height} -> {out.height}")
    return out


def compile_teams(bundle: dict, team_info: pl.DataFrame | None = None) -> pl.DataFrame:
    """Tidy one season bundle. Pure -- no network, no disk."""
    season = _int(bundle.get("season"))
    conferences = bundle.get("conferences") or {}
    divisions = _division_map(bundle)
    teams = bundle.get("teams") or {}
    # Driven by EVERY captured group list, not by the captured team payloads: a
    # team whose payload failed to fetch must still show up (as a null-filled row)
    # rather than vanish from the season it played in. Group 99 (the root, and the
    # whole season universe) carries no division label, so its ids are unioned in
    # from the raw lists rather than through `divisions`.
    listed = {str(t) for lst in (bundle.get("divisions") or {}).values() for t in lst or []}
    ids = sorted(listed | set(divisions) | set(teams), key=lambda t: _int(t) or 0)
    rows = [_team_row(season, tid, teams.get(tid) or {}, divisions.get(tid), conferences) for tid in ids]
    if not rows:
        return enrich_cfbd(pl.DataFrame(schema=SCHEMA), None)
    df = pl.from_dicts(rows, schema=SCHEMA)
    return enrich_cfbd(df.sort("team_id"), team_info)


def build_teams(
    season: int,
    *,
    raw_base: str = RAW_BASE,
    team_info_base: str = TEAM_INFO_BASE,
    **_: Any,
) -> pl.DataFrame:
    return compile_teams(
        load_bundle(season, raw_base=raw_base),
        load_team_info(season, base=team_info_base),
    )


def build(
    start_year: int,
    end_year: int,
    *,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
    raw_base: str = RAW_BASE,
) -> list[tuple[int, str]]:
    """Build (and optionally publish) cfb_teams across a season range.

    Every season is isolated: a failure is recorded and the sweep continues, so
    one bad season cannot abort a backfill. Returns the ``(season, error)`` list.
    """
    failures: list[tuple[int, str]] = []
    for season in range(start_year, end_year + 1):
        try:
            df = build_teams(season, raw_base=raw_base)
            if df.height == 0:
                print(f"  {SPEC.dataset} {season}: 0 rows, skipped", flush=True)
                failures.append((season, "empty"))
                continue
            counts = df.group_by("division").len().sort("division").rows() if "division" in df.columns else []
            # ESPN real-FBS vs CFBD FBS is the cross-provider check worth seeing on
            # every build: the two disagree by at most a handful of programs, so a
            # sudden gap means the join (or a division rule) moved.
            matched = int(df["school"].is_not_null().sum())
            real_fbs = int(
                df.filter((pl.col("is_fbs") == True) & (pl.col("is_exhibition") == False)).height  # noqa: E712
            )
            cfbd_fbs = int(df.filter(pl.col("classification") == "fbs").height)
            print(
                f"  {SPEC.dataset} {season}: {df.height} rows, {df.width} cols, "
                f"is_fbs={int(df['is_fbs'].sum())}, "
                f"cfbd_matched={matched} ({matched / df.height:.1%}), "
                f"real_fbs={real_fbs} vs cfbd_fbs={cfbd_fbs}, "
                + ", ".join(f"{k or 'none'}={v}" for k, v in counts),
                flush=True,
            )
            if dry_run:
                continue
            write_dataset(df, SPEC.dataset, season, SPEC.stem, base=base)
            if publish:
                from cfb_data_build.publish import publish_dataset

                publish_dataset(SPEC, season, base=base)
        except Exception as exc:  # noqa: BLE001 - one season must not kill the sweep
            print(f"  FAILED {season}: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            failures.append((season, type(exc).__name__))
    return failures
