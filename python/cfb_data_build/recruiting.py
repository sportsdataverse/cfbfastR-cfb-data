"""Compile the raw 247 recruit store into publishable datasets.

Reads ``cfb/recruits/json/{year}/page_*.json`` from cfbfastR-cfb-raw and emits
three tables:

===========================  ====================================================
``cfb_recruits``             one row per recruit (season, team_id, stars, grade)
``cfb_team_talent``          one row per (season, team_id): talent composite,
                             blue-chip ratio, class counts
``cfb_returning_production`` one row per (season, team_id) -- off/def returning
===========================  ====================================================

WHY THIS EXISTS
---------------
``cfb_roster_talent`` computed talent by hitting 247 live. A recruit class is
IMMUTABLE once signed, but the composite accumulates a 4-season window, so each
target season re-fetched the same frozen classes: ~96 pages / 20 minutes per
call, and ~3.3 hours to publish 2016-2025 -- against a host that resets
connections under sustained paging.

It also hid a failure. ``_PAGE_SIZE`` was 500, which exceeds what the RDB serves
inside the 3s client timeout, so every page raised curl(28) and talent returned
zero rows. That emptiness flowed into ``cfb_recruiting_projection``'s left join
(talent on the LEFT), producing zero output rows, and the ONLY thing that caught
it was ``build_recruiting``'s zero-row guard. With recruiting published as its
own dataset, "0 rows" is visible in the artifact instead of vanishing into a
downstream join.

THE PARSER IS NOT REIMPLEMENTED HERE. This module feeds sdv-py's own
``parse_sports247_result_set`` + ``_normalize_recruit_page`` from disk, so the
offline path cannot drift from the live one -- a second implementation would be
a second set of bugs, and this feed has already produced two (the unexpanded
``committed_institution`` on all-uncommitted pages, and the float team key that
stringifies to "71.0").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from cfb_data_build.config import DatasetSpec

_RAW_SUBPATH = "cfb/recruits/json"


def _sdv_bits():
    """sdv-py's parser + normalizer, imported lazily so tests can run without it."""
    from sportsdataverse.cfb.sports247_parsers import parse_sports247_result_set

    mod = sys.modules.get("sportsdataverse.cfb.cfb_roster_talent")
    if mod is None:
        import importlib

        mod = importlib.import_module("sportsdataverse.cfb.cfb_roster_talent")
    return parse_sports247_result_set, mod


def raw_year_dir(raw_root: str | Path, year: int) -> Path:
    return Path(raw_root) / _RAW_SUBPATH / str(year)


def available_years(raw_root: str | Path) -> list[int]:
    """Class years with a COMPLETE manifest. Incomplete years are not offered.

    The manifest is written only after the scraped row count matches the count
    the feed itself reports, so its presence is a checked claim rather than an
    inference from "the last page looked short".
    """
    base = Path(raw_root) / _RAW_SUBPATH
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        man = d / "_manifest.json"
        if not man.is_file():
            continue
        try:
            meta = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("complete") and str(d.name).isdigit():
            out.append(int(d.name))
    return out


def load_year(raw_root: str | Path, year: int) -> pl.DataFrame:
    """Per-recruit rows for one class year, read from the raw store.

    Raises:
        FileNotFoundError: If the year has no complete manifest. Refusing here
            beats returning an empty frame that a downstream join would quietly
            turn into "this team signed nobody".
    """
    parse, talent_mod = _sdv_bits()
    ydir = raw_year_dir(raw_root, year)
    if not (ydir / "_manifest.json").is_file():
        raise FileNotFoundError(
            f"recruit class {year} has no complete manifest under {ydir}. "
            "Run cfbfastR-cfb-raw/python/scrape_cfb_recruits.py first."
        )
    frames = []
    for page_file in sorted(ydir.glob("page_*.json")):
        stored = json.loads(page_file.read_text(encoding="utf-8"))
        raw = parse(stored["payload"])
        if raw.height == 0:
            continue
        # Same normalizer the live path uses -- including its "an all-null
        # institution object means an all-uncommitted page, not schema drift"
        # handling, which is why pages deep in a class do not blow this up.
        frames.append(talent_mod._normalize_recruit_page(raw, year))
    if not frames:
        return pl.DataFrame(schema=talent_mod._RECRUIT_SCHEMA)
    return pl.concat(frames)


def build_recruits(
    raw_root: str | Path, years: list[int] | None = None
) -> pl.DataFrame:
    """``cfb_recruits``: every signed/committed recruit across the given class years."""
    have = available_years(raw_root)
    want = sorted(set(years) & set(have)) if years else have
    if not want:
        raise ValueError(
            f"no complete recruit classes under {raw_root} for years={years}; available={have}"
        )
    return pl.concat([load_year(raw_root, y) for y in want])


def build_team_talent(
    recruits: pl.DataFrame,
    seasons: list[int],
    *,
    window: int = 4,
    max_class_size: int = 25,
    division: str = "fbs",
) -> pl.DataFrame:
    """``cfb_team_talent``: per (season, team_id) composite + blue-chip ratio.

    Delegates the arithmetic to sdv-py's ``cfb_roster_talent`` via its
    ``recruits=`` injection point, so the published dataset and a live call are
    the same computation over the same inputs. (An earlier draft swapped the
    module's ``load_recruit_classes`` global instead -- fragile, not
    thread-safe, and the same shim/monkeypatch trap that has bitten this
    ecosystem before. A real parameter is the honest seam.)
    """
    _, talent_mod = _sdv_bits()
    return talent_mod.cfb_roster_talent(
        seasons, division=division, max_class_size=max_class_size, recruits=recruits
    )


#: Published recruiting datasets. Stems match the release tag, as elsewhere.
RECRUITING_SPECS = {
    "recruits": DatasetSpec("cfb_recruits", "cfb_recruits", "cfb_recruits"),
    "team_talent": DatasetSpec("cfb_team_talent", "cfb_team_talent", "cfb_team_talent"),
}

#: Class years behind a talent composite. A season's talent is built from this
#: many signing classes, so the raw store must reach back this far before the
#: earliest target season.
TALENT_WINDOW = 4


def build_recruiting(
    dataset: str,
    start_year: int,
    end_year: int,
    *,
    raw_root: str | Path,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
) -> list[tuple[int, str]]:
    """Build (and optionally publish) a recruiting dataset across a season range.

    Mirrors ``build_derived``'s contract: every season is isolated, failures are
    collected rather than raising, and the list of ``(season, error_type)`` is
    returned.

    Unlike the game-derived datasets this reads NO game data -- only the raw
    247 store -- so it can rebuild years after the fact with no network at all.
    """
    from cfb_data_build.checks import assert_talent_is_real
    from cfb_data_build.publish import publish_dataset

    if dataset not in RECRUITING_SPECS:
        raise ValueError(
            f"unknown recruiting dataset {dataset!r}; expected one of {sorted(RECRUITING_SPECS)}"
        )
    spec = RECRUITING_SPECS[dataset]
    have = set(available_years(raw_root))
    failures: list[tuple[int, str]] = []

    for season in range(start_year, end_year + 1):
        try:
            # Talent for season S draws on classes S-window+1..S; a missing
            # class silently shrinks the composite, so require the full window
            # rather than quietly publishing a thinner number.
            years = list(range(season - TALENT_WINDOW + 1, season + 1))
            missing = [y for y in years if y not in have]
            if missing:
                raise FileNotFoundError(
                    f"{spec.dataset} {season}: raw store is missing complete classes {missing}. "
                    "Scrape them first -- a partial window understates talent without erroring."
                )
            recruits = build_recruits(raw_root, years)
            if dataset == "recruits":
                df = recruits.filter(pl.col("season") == season)
            else:
                df = build_team_talent(recruits, [season])
                assert_talent_is_real(df, label=f"{spec.dataset} {season}")
            if df.height == 0:
                raise ValueError(
                    f"{spec.dataset} {season}: 0 rows -- refusing to publish an empty season"
                )

            out = Path(base) / spec.dataset
            out.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out / f"{spec.stem}_{season}.parquet")
            print(f"  {spec.dataset} {season}: {df.height} rows", flush=True)
            if publish:
                publish_dataset(spec, season, base=base, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - one bad season must not kill the sweep
            print(
                f"  {spec.dataset} {season}: FAILED ({type(exc).__name__}: {exc})",
                flush=True,
            )
            failures.append((season, type(exc).__name__))
    return failures
