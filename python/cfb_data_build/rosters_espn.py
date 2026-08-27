"""``cfb_rosters`` -- the ESPN-native season roster, with a REAL resolved position.

Why this module exists rather than a ``REGISTRY`` row:

* **Source.** Every other dataset here reshapes ``cfb/json/final/{id}.json`` (55 GB
  across 2004-2025). A season roster needs only the per-game roster block, which the
  raw repo also persists standalone at ``cfb/game_rosters/json/{id}.json`` -- 6.2 GB,
  a ~9x smaller HTTP read for identical output. Compiling from the standalone block
  is the whole efficiency argument.
* **Position.** ESPN ships roster rows with ``position_href`` (a URL) and no position
  field at all. The released ``espn_cfb_rosters`` 2023-2025 assets shipped that href
  verbatim, so the dataset had no usable position column. Here the id is parsed out of
  the href and joined to the league position reference
  (``cfb/reference/positions.json``, 74 entries) to yield ``position`` /
  ``position_abbreviation`` / ``position_name`` / ``position_leaf`` /
  ``position_parent_id``.
* **Division.** ESPN team payloads carry no division field;
  ``cfb/teams/json/{season}.json`` ``divisions`` (group 80 = fbs, 81 = fcs) is the
  only authoritative source.
* **Hometown / recruiting.** ESPN publishes BIRTH place (``birth_place_*``) and no
  recruiting link at all. CollegeFootballData's season roster carries HOME town/geo
  and ``recruit_ids``; those seven fields are LEFT-joined on ``athlete_id`` under a
  ``cfbd_`` prefix (:data:`CFBD_COLS`). Hometown is NOT birthplace -- the two are
  distinct facts and both are shipped.

Everything is read over HTTP from ``RAW_BASE`` (raw.githubusercontent.com) -- the
same contract as ``R/_data_utils.R``. ``raw_base`` is overridable for offline
testing only; the shipped default is the HTTP URL.

Grain: one row per ``(season, team_id, athlete_id)``. When an athlete appears on
several game rosters in a season the LAST one (by week, then game id) supplies the
attribute values -- the same "most recent wins" rule the R producer used -- and
``games_rostered`` counts the appearances.

Usage::

    python -m cfb_data_build --dataset cfb_rosters -s 2004 -e 2025
    python -m cfb_data_build --dataset cfb_rosters -s 2024 -e 2024 --publish
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import polars as pl

from cfb_data_build.config import DatasetSpec
from cfb_data_build.io import gzip_csv, write_dataset
from cfb_data_build.teams import DIVISIONS as _TEAM_DIVISIONS
from cfb_data_ingest import RAW_BASE
from cfb_data_ingest.schedule import season_game_ids

SPEC = DatasetSpec(
    dataset="cfb_rosters",
    stem="cfb_rosters",
    tag="espn_cfb_rosters",
    reshaper="cfb_rosters",
)

#: Per-GAME circumstance -- meaningless once collapsed to a season roster. Mirrors
#: R's ``GAME_ROSTER_GAME_COLS`` plus the per-game statistics URL. Documented rather
#: than used directly: :data:`ESPN_COLS` is the positive list.
GAME_ONLY_COLS = (
    "game_id",
    "week",
    "starter",
    "did_not_play",
    "order",
    "home_away",
    "winner",
    "valid",
    "statistics_href",
)

#: The FULL union of ESPN athlete/team fields observed across 2004-2025 (2.96M
#: athlete-game rows, 77 distinct keys, minus :data:`GAME_ONLY_COLS`). Pinned as a
#: constant so EVERY season writes the same schema -- a key ESPN only started
#: sending in 2005 (``hand_*``, ``citizenship``, ``nickname``) or 2008
#: (``jersey_right``, ``display_name``) is null-filled in earlier seasons rather
#: than shifting the column set season to season.
ESPN_COLS: tuple[str, ...] = (
    "athlete_id",
    "athlete_uid",
    "athlete_guid",
    "athlete_type",
    "first_name",
    "middle_name",
    "last_name",
    "full_name",
    "display_name",
    "athlete_display_name",
    "short_name",
    "nickname",
    "slug",
    "jersey",
    "jersey_right",
    "weight",
    "display_weight",
    "height",
    "display_height",
    "age",
    "date_of_birth",
    "hand_type",
    "hand_abbreviation",
    "hand_display_value",
    "linked",
    "active",
    "alternate_ids_sdr",
    "birth_place_city",
    "birth_place_state",
    "birth_place_country",
    "birth_country_alternate_id",
    "birth_country_abbreviation",
    "citizenship",
    "flag_href",
    "flag_alt",
    "flag_rel",
    "headshot_href",
    "headshot_alt",
    "experience_years",
    "experience_display_value",
    "experience_abbreviation",
    "status_id",
    "status_name",
    "status_type",
    "status_abbreviation",
    "draft_display_text",
    "draft_round",
    "draft_year",
    "draft_selection",
    "draft_team_href",
    "team_guid",
    "team_uid",
    "team_slug",
    "team_location",
    "team_name",
    "team_nickname",
    "team_abbreviation",
    "team_display_name",
    "team_short_display_name",
    "team_color",
    "team_alternate_color",
    "team_alternate_ids_sdr",
    "is_active",
    "is_all_star",
    "logo_href",
    "logo_dark_href",
    "athlete_href",
    "position_href",
)

#: Where the CollegeFootballData season roster lives. A *different* repo from
#: :data:`RAW_BASE` -- cfbfastR-data, not cfbfastR-cfb-raw.
CFBD_ROSTER_BASE = (
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/rosters/parquet"
)

#: The CFBD roster fields ESPN has NO equivalent for, mapped source -> output name.
#: Prefixed ``cfbd_`` so provenance is unambiguous and so ``home_*`` can never be
#: mistaken for ESPN's ``birth_place_*`` (hometown != birthplace).
#:
#: Deliberately EXCLUDED: ``headshot_url`` (ESPN ships ``headshot_href``), ``team``
#: (ESPN ships the whole team block), ``year`` (0% populated on older seasons;
#: ESPN's ``experience_*`` covers class standing).
CFBD_COLS: dict[str, str] = {
    "recruit_ids": "cfbd_recruit_ids",
    "home_city": "cfbd_home_city",
    "home_state": "cfbd_home_state",
    "home_country": "cfbd_home_country",
    "home_latitude": "cfbd_home_latitude",
    "home_longitude": "cfbd_home_longitude",
    "home_county_fips": "cfbd_home_county_fips",
}

#: Final column ORDER. Identity, then the resolved position, then the ESPN union,
#: then the CFBD-only enrichment.
OUTPUT_COLS: tuple[str, ...] = (
    "season",
    "team_id",
    "athlete_id",
    "division",
    "position_id",
    "position",
    "position_abbreviation",
    "position_name",
    "position_leaf",
    "position_parent_id",
    "games_rostered",
    *(c for c in ESPN_COLS if c != "athlete_id"),
    *CFBD_COLS.values(),
)

#: Int64 everywhere these appear -- they are join keys. Never cast via float
#: (a float-origin id stringifies as "123.0" and silently breaks a join).
ID_COLS = ("season", "team_id", "athlete_id", "position_id", "position_parent_id")

_POSITION_ID_RE = re.compile(r"/positions/([^/?]+)")


#: Reuse the teams builder's map so the two datasets speak one division
#: vocabulary. Ordering is load-bearing: it is MOST SPECIFIC FIRST, so a team
#: listed under both a leaf and its parent takes the leaf.
DIVISION_BY_GROUP = dict(_TEAM_DIVISIONS)


# --------------------------------------------------------------------------- IO


def _download(url: str) -> str | None:
    """GET ``url``, returning the body or ``None`` for anything non-200."""
    from sportsdataverse.dl_utils import download  # pooled session + retry/backoff

    resp = download(url)
    if getattr(resp, "status_code", 200) != 200 or not getattr(resp, "text", ""):
        return None
    return resp.text


def _download_bytes(url: str) -> bytes | None:
    """GET ``url`` for a BINARY body (parquet), returning ``None`` for non-200."""
    from sportsdataverse.dl_utils import download

    resp = download(url)
    if getattr(resp, "status_code", 200) != 200:
        return None
    return getattr(resp, "content", None) or None


def _read_json(src: str, *, downloader: Callable[[str], str | None] | None = None):
    """Read a JSON document from an HTTP URL (or a filesystem path when testing)."""
    if "://" in src:
        body = (downloader or _download)(src)
        return json.loads(body) if body else None
    p = Path(src)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_positions(
    raw_base: str = RAW_BASE, *, downloader: Callable[[str], str | None] | None = None
) -> pl.DataFrame:
    """The 74-row league position reference as a joinable frame.

    Source: ``cfb/reference/positions.json`` in cfbfastR-cfb-raw, keyed by position
    id as a STRING; cast to Int64 here so the join key matches ``position_id``.
    """
    doc = _read_json(f"{raw_base}/reference/positions.json", downloader=downloader)
    if not doc or not doc.get("positions"):
        raise RuntimeError(
            "position reference unavailable at "
            f"{raw_base}/reference/positions.json -- refusing to ship href-only "
            "positions (that is the defect this dataset exists to fix)"
        )
    rows = []
    for pid, payload in doc["positions"].items():
        parent = (payload.get("parent") or {}).get("$ref")
        rows.append(
            {
                "position_id": int(pid),
                "position": payload.get("displayName"),
                "position_name": payload.get("name"),
                "position_abbreviation": payload.get("abbreviation"),
                "position_leaf": bool(payload.get("leaf")),
                "position_parent_id": _position_id(parent),
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("position_id").cast(pl.Int64),
        pl.col("position_parent_id").cast(pl.Int64),
    )


def load_divisions(
    season: int,
    raw_base: str = RAW_BASE,
    *,
    downloader: Callable[[str], str | None] | None = None,
) -> pl.DataFrame:
    """``(team_id, division)`` for one season from the raw season team bundle.

    The bundle's ``divisions`` map is the ONLY authoritative division source --
    the team payload itself carries no division field. It holds the whole ESPN
    group tree, structural nodes included, so the group ids are resolved through
    the teams builder's most-specific-first map rather than the bundle's own key
    order. Returns an empty frame when the bundle is absent so the build degrades
    to a null ``division`` rather than failing.
    """
    doc = _read_json(f"{raw_base}/teams/json/{season}.json", downloader=downloader)
    groups = (doc or {}).get("divisions") or {}

    # Walk DIVISION_BY_GROUP in its own most-specific-first order and take the
    # first hit per team. Do NOT iterate the bundle's own key order with
    # unique(keep="first"): the bundle carries the structural nodes too (99 has
    # all 800 of 2023's teams, 90 has 272) and they map to no division, so
    # whichever happened to come first would blank every real classification.
    assigned: dict[int, str] = {}
    for group_id, division in DIVISION_BY_GROUP.items():
        for tid in groups.get(str(group_id)) or []:
            assigned.setdefault(int(tid), division)

    if not assigned:
        return pl.DataFrame(schema={"team_id": pl.Int64, "division": pl.Utf8})
    return pl.DataFrame(
        {
            "team_id": list(assigned.keys()),
            "division": list(assigned.values()),
        },
        schema={"team_id": pl.Int64, "division": pl.Utf8},
    )


def load_cfbd_rosters(
    season: int,
    cfbd_base: str = CFBD_ROSTER_BASE,
    *,
    fetcher: Callable[[str], bytes | None] | None = None,
) -> pl.DataFrame:
    """The CFBD-only roster fields for one season, keyed by ``athlete_id`` (Int64).

    CFBD ships ``athlete_id`` as a STRING and ``recruit_ids`` as ``List(Int32)``.
    Both are normalized here, at the boundary: the id is cast straight from Utf8 to
    Int64 (never via float -- a float-origin id stringifies as ``"123.0"``), and the
    recruit list is JSON-encoded to a string so parquet, rds and csv all carry one
    dtype (the same rule :mod:`cfb_data_build.reshape` applies to every list cell).

    **Deduped to one row per athlete.** CFBD emits an athlete once per team, so a
    mid-season transfer appears twice (197 such rows in 2021, the worst season). The
    seven columns taken here are athlete-level facts and are byte-identical across an
    athlete's rows -- verified 2004-2025, where a full-row ``unique()`` collapses the
    duplicates to exactly the distinct-id count -- so the dedup is lossless. The
    post-condition is asserted rather than assumed: a future season where the values
    genuinely disagree raises instead of silently fanning out the ESPN row set.

    Returns an empty (but correctly typed) frame when the season has no CFBD asset,
    so the build degrades to null enrichment rather than failing.
    """
    schema = {"athlete_id": pl.Int64, **{v: pl.Utf8 for v in CFBD_COLS.values()}}
    body = (fetcher or _download_bytes)(f"{cfbd_base}/cfb_rosters_{season}.parquet")
    if not body:
        return pl.DataFrame(schema=schema)

    import io

    df = pl.read_parquet(io.BytesIO(body))
    have = [c for c in CFBD_COLS if c in df.columns]
    df = df.select(["athlete_id", *have]).with_columns(
        pl.col("athlete_id").cast(pl.Utf8).cast(pl.Int64, strict=False)
    )
    if "recruit_ids" in have:
        df = df.with_columns(
            pl.col("recruit_ids")
            .cast(pl.List(pl.Int64))
            .list.eval(pl.element().cast(pl.Utf8))
            .list.join(",")
            .map_elements(
                lambda v: None if v is None or v == "" else f"[{v}]",
                return_dtype=pl.Utf8,
            )
        )
    df = df.with_columns(
        [pl.col(c).cast(pl.Utf8) for c in have if c != "recruit_ids"]
    ).rename({c: CFBD_COLS[c] for c in have})

    df = df.filter(pl.col("athlete_id").is_not_null()).unique()
    if df.height != df.get_column("athlete_id").n_unique():
        raise RuntimeError(
            f"cfbd rosters {season}: athlete_id is not unique after dedup -- the "
            "CFBD-only fields disagree between an athlete's rows, so a left join "
            "would multiply the ESPN row set"
        )
    missing = [c for c in schema if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing])
    return df.select(list(schema))


def _game_roster_url(raw_base: str, game_id: int) -> str:
    return f"{raw_base}/game_rosters/json/{game_id}.json"


def fetch_game_rosters(
    game_ids: Iterable[int],
    raw_base: str = RAW_BASE,
    *,
    cache_dir: str | Path | None,
    workers: int = 8,
    downloader: Callable[[str], str | None] | None = None,
) -> list[dict[str, Any]]:
    """Read every game's roster payload, resuming from ``cache_dir`` when set.

    Fail-soft per game: a missing capture (16 of 19,586 master ids have none) drops
    that game rather than aborting the season. Concurrency stays modest --
    raw.githubusercontent tolerates it, but a per-game payload averages 300 KB.
    """
    ids = list(game_ids)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def one(gid: int) -> dict[str, Any] | None:
        dest = cache / f"{gid}.json" if cache else None
        if dest is not None and dest.exists():
            try:
                return json.loads(dest.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 -- corrupt cache entry: re-fetch
                pass
        try:
            doc = _read_json(_game_roster_url(raw_base, gid), downloader=downloader)
        except Exception:  # noqa: BLE001 -- one bad game cannot abort the season
            return None
        if doc is not None and dest is not None:
            dest.write_text(json.dumps(doc), encoding="utf-8")
        return doc

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [d for d in pool.map(one, ids) if d]


# ---------------------------------------------------------------------- reshape


def _position_id(href: str | None) -> int | None:
    """Position id parsed out of a ``.../positions/{id}?lang=..`` href."""
    if not href:
        return None
    m = _POSITION_ID_RE.search(href)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def roster_rows(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten ``{game_id, season, week, data:[athlete,...]}`` docs to athlete-game rows."""
    out: list[dict[str, Any]] = []
    for doc in payloads:
        season, week, gid = doc.get("season"), doc.get("week"), doc.get("game_id")
        for row in doc.get("data") or []:
            r = dict(row)
            r["season"], r["week"], r["game_id"] = season, week, gid
            out.append(r)
    return out


def empty_frame() -> pl.DataFrame:
    """Zero-row frame carrying the documented schema (stable for empty seasons)."""
    return pl.DataFrame(
        schema={
            c: (pl.Int64 if c in ID_COLS or c == "games_rostered" else pl.Utf8)
            for c in OUTPUT_COLS
        }
    )


def derive_rosters(
    rows: list[dict[str, Any]],
    positions: pl.DataFrame,
    divisions: pl.DataFrame,
    cfbd: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Collapse athlete-game rows to one row per ``(season, team_id, athlete_id)``.

    Last appearance (by week, then game id) supplies the attributes; ``games_rostered``
    counts them. Position is resolved from ``position_href``; ``division`` is joined
    from the season team bundle; the CFBD-only hometown/recruiting fields are LEFT
    joined from ``cfbd`` on ``athlete_id``.

    The CFBD join is deliberately LEFT: ESPN owns the row set. CFBD carries athletes
    ESPN omits (mostly FCS) and those are NOT unioned in here -- see ``DATASETS.md``
    for the coverage accounting.
    """
    if not rows:
        return empty_frame()

    df = pl.from_dicts(rows, infer_schema_length=None)
    # Pin id dtypes at the boundary, before any join touches them.
    df = df.with_columns(
        pl.col("season").cast(pl.Int64, strict=False),
        pl.col("team_id").cast(pl.Int64, strict=False),
        pl.col("athlete_id").cast(pl.Int64, strict=False),
        pl.col("week").cast(pl.Int64, strict=False),
        pl.col("game_id").cast(pl.Int64, strict=False),
    )
    # jersey arrives space-padded from ESPN ("88 ").
    if "jersey" in df.columns:
        df = df.with_columns(pl.col("jersey").cast(pl.Utf8).str.strip_chars())

    counts = df.group_by(["team_id", "athlete_id"]).agg(
        pl.len().alias("games_rostered")
    )
    latest = (
        df.sort(["week", "game_id"], nulls_last=False)
        .group_by(["team_id", "athlete_id"])
        .last()
    )
    df = latest.join(counts, on=["team_id", "athlete_id"], how="left")

    df = df.with_columns(
        pl.col("position_href")
        .map_elements(_position_id, return_dtype=pl.Int64)
        .alias("position_id")
    )
    assert positions.schema["position_id"] == df.schema["position_id"], (
        "position_id dtype disagreement between roster rows and the position reference"
    )
    df = df.join(positions, on="position_id", how="left")

    assert divisions.schema["team_id"] == df.schema["team_id"], (
        "team_id dtype disagreement between roster rows and the division map"
    )
    df = df.join(divisions, on="team_id", how="left")

    if cfbd is not None and cfbd.height:
        assert cfbd.schema["athlete_id"] == df.schema["athlete_id"] == pl.Int64, (
            "athlete_id dtype disagreement between roster rows and the CFBD roster"
        )
        before = df.height
        df = df.join(cfbd, on="athlete_id", how="left")
        assert df.height == before, (
            f"CFBD left join changed the ESPN row set ({before} -> {df.height})"
        )

    # Null-fill any column ESPN did not send this season, then pin order + dtypes.
    missing = [c for c in OUTPUT_COLS if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing])
    df = df.select(list(OUTPUT_COLS))
    df = df.with_columns(
        [pl.col(c).cast(pl.Int64, strict=False) for c in (*ID_COLS, "games_rostered")]
    )
    return df.sort(["team_id", "position_id", "athlete_id"], nulls_last=True)


# ------------------------------------------------------------------------ build


#: Shared with every other builder that ships a gzipped season csv.
_gzip_csv = gzip_csv


def build_season(
    season: int,
    *,
    raw_base: str = RAW_BASE,
    cfbd_base: str = CFBD_ROSTER_BASE,
    schedule: str | Path | None = None,
    cache_dir: str | Path | None = ".cache/cfb_game_rosters",
    workers: int = 8,
    base: str | Path = "cfb",
    publish: bool = False,
    write: bool = True,
) -> pl.DataFrame:
    """Compile, write and optionally publish one season of ``cfb_rosters``."""
    positions = load_positions(raw_base)
    divisions = load_divisions(season, raw_base)
    cfbd = load_cfbd_rosters(season, cfbd_base)
    ids = season_game_ids(schedule, [season])
    payloads = fetch_game_rosters(ids, raw_base, cache_dir=cache_dir, workers=workers)
    df = derive_rosters(roster_rows(payloads), positions, divisions, cfbd)
    n = max(1, df.height)
    n_pos = df.filter(pl.col("position").is_not_null()).height
    n_div = df.filter(pl.col("division").is_not_null()).height
    n_rec = df.filter(pl.col("cfbd_recruit_ids").is_not_null()).height
    print(
        f"cfb_rosters {season}: {df.height} rows from {len(payloads)}/{len(ids)} games"
        f" | position {100 * n_pos / n:.1f}% | division {100 * n_div / n:.1f}%"
        f" | cfbd_recruit_ids {100 * n_rec / n:.1f}% of {cfbd.height} cfbd athletes",
        flush=True,
    )
    if df.height == 0:
        # A supported season that compiles to nothing means every roster fetch
        # failed. Returning quietly would skip write_dataset, record no failure,
        # and let the CLI report success with no artifact for the season.
        raise RuntimeError(f"cfb_rosters {season}: compiled zero roster rows")
    if write:
        paths = write_dataset(df, SPEC.dataset, season, SPEC.stem, base=base)
        _gzip_csv(paths)
        if publish:
            from cfb_data_build.publish import publish_dataset

            publish_dataset(SPEC, season, base=base)
    return df


def build(start_year: int, end_year: int, **kwargs: Any) -> list[tuple[int, str]]:
    """Build a season range; returns ``(season, error)`` for each failure."""
    failures: list[tuple[int, str]] = []
    for season in range(start_year, end_year + 1):
        try:
            build_season(season, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- one season cannot abort the range
            failures.append((season, repr(exc)))
            print(f"cfb_rosters {season}: FAILED {exc!r}", flush=True)
    return failures
