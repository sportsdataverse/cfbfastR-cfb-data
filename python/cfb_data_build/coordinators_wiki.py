"""Head coach and coordinator names per FBS school-season, from Wikipedia.

No API publishes offensive / defensive coordinators before the imported
coordinator table starts (2013). Wikipedia's team-season articles do: nearly
every FBS school-season since before 2002 has one, and its infobox
(``Infobox NCAA team season`` or ``Infobox college sports team season`` --
same parameters) carries ``head_coach`` / ``hc_year`` / ``off_coach`` /
``def_coach`` plus co-coordinator slots ``cooff_coach1..2`` and
``codef_coach1..2``.

Reproducibility is by REVISION, not by page. Every row records the revision id
it was read from, and ``backfill --pinned`` re-reads exactly those revisions
through the MediaWiki API (``revids=``), so a rebuild returns the same names
even after the articles are edited. A fresh run (without ``--pinned``) reads
the current revisions and records their ids for the next pinned rebuild.

The universe of school-seasons is CFBD ``/coaches`` for the season (the FBS
programs that year); the article title is ``{year} {school} {mascot} football
team`` in CFBD spelling, then the era's Wikipedia spelling from
:data:`TITLE_STEMS`, then the MediaWiki search as a last resort. A school-season
with no resolvable article is reported, never guessed.

Values are facts (who held a job), recorded with the article title and revision
for attribution.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import polars as pl

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "SportsDataverse-cfbfastR-cfb-data/1.0 "
    "(https://github.com/sportsdataverse/cfbfastR-cfb-data)"
)
#: MediaWiki caps a revisions query at 50 titles (or revision ids)
BATCH = 50
INFOBOX = re.compile(
    r"\{\{\s*Infobox\s+(?:NCAA\s+team\s+season|college\s+sports\s+team\s+season)\b",
    re.IGNORECASE,
)

#: The Wikipedia title stem ("school mascot") a CFBD school used, newest first.
#: Only schools whose CFBD spelling does not resolve directly need an entry.
TITLE_STEMS: dict[str, tuple[str, ...]] = {
    "Arkansas State": ("Arkansas State Red Wolves", "Arkansas State Indians"),
    "Florida International": (
        "FIU Panthers",
        "FIU Golden Panthers",
        "Florida International Golden Panthers",
    ),
    "Hawai'i": ("Hawaii Rainbow Warriors", "Hawaii Warriors"),
    "Louisiana": (
        "Louisiana Ragin' Cajuns",
        "Louisiana–Lafayette Ragin' Cajuns",
        "Louisiana-Lafayette Ragin' Cajuns",
    ),
    "Massachusetts": ("UMass Minutemen", "Massachusetts Minutemen"),
    "Miami (OH)": ("Miami RedHawks",),
    "San José State": ("San Jose State Spartans",),
    "App State": ("Appalachian State Mountaineers",),
    "UL Monroe": (
        "Louisiana–Monroe Warhawks",
        "Louisiana-Monroe Warhawks",
        "Louisiana–Monroe Indians",  # the mascot until 2006
        "Louisiana-Monroe Indians",
    ),
}


def _api(params: dict[str, str], *, attempts: int = 6) -> dict:
    """One MediaWiki API call: JSON, polite ``maxlag``, retried with backoff."""
    query = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    url = API + "?" + urllib.parse.urlencode(query)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https host
                payload = json.load(resp)
        except Exception as exc:  # noqa: BLE001 - network: retry every transport error
            last = exc
            time.sleep(2 * (attempt + 1))
            continue
        error = payload.get("error") or {}
        if error.get("code") == "maxlag":
            time.sleep(5 * (attempt + 1))
            continue
        if error:
            raise RuntimeError(f"MediaWiki error: {error}")
        return payload
    raise RuntimeError(f"MediaWiki request failed after {attempts} attempts: {last}")


@dataclass(frozen=True)
class Page:
    """One resolved article revision."""

    title: str
    revid: int
    text: str


def _pages_from(payload: dict) -> dict[str, Page]:
    out: dict[str, Page] = {}
    for p in payload.get("query", {}).get("pages", []):
        if p.get("missing") or not p.get("revisions"):
            continue
        rev = p["revisions"][0]
        out[p["title"]] = Page(
            p["title"], int(rev["revid"]), rev["slots"]["main"]["content"]
        )
    return out


def fetch_titles(titles: list[str]) -> dict[str, Page]:
    """Requested title -> the article it resolves to (redirects followed)."""
    resolved: dict[str, Page] = {}
    for i in range(0, len(titles), BATCH):
        chunk = titles[i : i + BATCH]
        payload = _api(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "ids|content",
                "rvslots": "main",
                "redirects": "1",
                "titles": "|".join(chunk),
            }
        )
        pages = _pages_from(payload)
        hops = {x["from"]: x["to"] for x in payload["query"].get("normalized", [])}
        for x in payload["query"].get("redirects", []):
            hops[x["from"]] = x["to"]
        for requested in chunk:
            final = requested
            for _ in range(4):  # normalized -> redirect -> redirect
                if final not in hops:
                    break
                final = hops[final]
            if final in pages:
                resolved[requested] = pages[final]
        time.sleep(1)
    return resolved


def fetch_revisions(revids: list[int]) -> dict[int, Page]:
    """Exact revisions by id -- the pinned, reproducible read."""
    out: dict[int, Page] = {}
    for i in range(0, len(revids), BATCH):
        chunk = revids[i : i + BATCH]
        payload = _api(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "ids|content",
                "rvslots": "main",
                "revids": "|".join(str(r) for r in chunk),
            }
        )
        for page in _pages_from(payload).values():
            out[page.revid] = page
        time.sleep(1)
    return out


def search_title(season: int, school: str) -> str | None:
    """Last resort: the MediaWiki search's first ``{season} ... football team`` hit."""
    payload = _api(
        {
            "action": "query",
            "list": "search",
            "srsearch": f'"{season}" {school} "football team"',
            "srlimit": "5",
            "srnamespace": "0",
        }
    )
    pattern = re.compile(rf"^{season} .+ football team$")
    for hit in payload.get("query", {}).get("search", []):
        if pattern.match(hit["title"]) and _contract(school) in _contract(hit["title"]):
            return hit["title"]
    return None


def _contract(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# --- parsing -----------------------------------------------------------------


def infobox_params(text: str) -> dict[str, str]:
    """The team-season infobox's top-level ``name = value`` pairs (raw wikitext).

    Split at depth 0 only, so a pipe inside ``[[A|B]]`` or a nested
    ``{{template|..}}`` never ends a parameter.
    """
    m = INFOBOX.search(text)
    if not m:
        return {}
    i, depth, start = m.start(), 0, m.start()
    body_end = len(text)
    while i < len(text) - 1:
        pair = text[i : i + 2]
        if pair in ("{{", "[["):
            depth += 1
            i += 2
            continue
        if pair in ("}}", "]]"):
            depth -= 1
            i += 2
            if depth == 0:
                body_end = i - 2
                break
            continue
        i += 1
    body = text[start + 2 : body_end]
    parts, buf, depth = [], [], 0
    j = 0
    while j < len(body):
        pair = body[j : j + 2]
        if pair in ("{{", "[["):
            depth += 1
            buf.append(pair)
            j += 2
            continue
        if pair in ("}}", "]]"):
            depth -= 1
            buf.append(pair)
            j += 2
            continue
        if body[j] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            j += 1
            continue
        buf.append(body[j])
        j += 1
    parts.append("".join(buf))
    out: dict[str, str] = {}
    for part in parts[1:]:  # parts[0] is the template name
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip().lower()] = value.strip()
    return out


_SORTNAME = re.compile(r"\{\{\s*sortname\s*\|\s*([^|}]*)\|\s*([^|}]*)[^}]*\}\}", re.I)


def clean_value(raw: str) -> list[str]:
    """Wikitext cell -> the people named in it, display names only.

    Handles links (``[[Target|Name]]``), ``{{sortname|First|Last}}``, line
    breaks and ``/`` separating co-holders, references, comments, HTML
    entities and bare formatting.
    """
    if not raw:
        return []
    s = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", "", s, flags=re.I)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.I | re.S)
    s = _SORTNAME.sub(lambda m: f"{m.group(1).strip()} {m.group(2).strip()}", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)  # other templates carry no name
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = unescape(s).replace("\xa0", " ")
    s = re.sub(r"'{2,}", "", s)
    # annotations like "(2nd season / first as OC)" or "(interim)" describe the
    # person, never name one -- and their slashes must not split a name
    s = re.sub(r"\s*\([^()]*\)", "", s)
    names = []
    # a comma splits co-holders, except before a generational suffix
    for piece in re.split(r"\n|/|;|,(?!\s*(?:Jr|Sr|II|III|IV)\b)| and ", s):
        name = re.sub(r"\s+", " ", piece).strip(" -–—:,")
        if name and not re.fullmatch(r"(?i)(none|n/?a|vacant|tbd|tba|-|–)", name):
            names.append(name)
    return list(dict.fromkeys(names))


def _ordinal(raw: str) -> int | None:
    m = re.search(r"(\d+)", raw or "")
    return int(m.group(1)) if m else None


def coaches_from_page(text: str, season: int) -> dict[str, object] | None:
    """Head coach, first season and both coordinator lists from one article."""
    params = infobox_params(text)
    if not params:
        return None
    head = clean_value(params.get("head_coach", ""))
    offense = []
    defense = []
    for key in ("off_coach", "cooff_coach1", "cooff_coach2", "cooff_coach3"):
        offense += clean_value(params.get(key, ""))
    for key in ("def_coach", "codef_coach1", "codef_coach2", "codef_coach3"):
        defense += clean_value(params.get(key, ""))
    tenure = _ordinal(params.get("hc_year", ""))
    return {
        "head_coach": head[0] if head else None,
        "first_season": season - tenure + 1 if tenure else None,
        "offensive_coordinators": " / ".join(dict.fromkeys(offense)) or None,
        "defensive_coordinators": " / ".join(dict.fromkeys(defense)) or None,
    }


# --- building ----------------------------------------------------------------

COLUMNS = {
    "season": pl.Int64,
    "school_mascot": pl.Utf8,
    "head_coach": pl.Utf8,
    "first_season": pl.Int64,
    "offensive_coordinators": pl.Utf8,
    "defensive_coordinators": pl.Utf8,
    "source_title": pl.Utf8,
    "source_revid": pl.Int64,
}


def title_candidates(season: int, school: str, mascot: str) -> list[str]:
    stems = [f"{school} {mascot}".strip(), *TITLE_STEMS.get(school, ())]
    return [f"{season} {stem} football team" for stem in dict.fromkeys(stems)]


def build_season(
    season: int, schools: dict[str, str], *, use_search: bool = True
) -> tuple[pl.DataFrame, list[str]]:
    """Rows for one season from the CURRENT revisions; returns (rows, unresolved).

    ``schools`` maps each CFBD school playing FBS that season to its mascot.
    """
    candidates = {s: title_candidates(season, s, m) for s, m in schools.items()}
    wanted = sorted({t for ts in candidates.values() for t in ts})
    found = fetch_titles(wanted)
    rows, unresolved = [], []
    for school, mascot in sorted(schools.items()):
        page = next((found[t] for t in candidates[school] if t in found), None)
        if page is None and use_search:
            hit = search_title(season, school)
            if hit:
                page = fetch_titles([hit]).get(hit)
        parsed = coaches_from_page(page.text, season) if page else None
        if parsed is None:
            unresolved.append(school)
            continue
        rows.append(
            {
                "season": season,
                "school_mascot": f"{school} {mascot}".strip(),
                **parsed,
                "source_title": page.title,
                "source_revid": page.revid,
            }
        )
    return pl.DataFrame(rows, schema=COLUMNS), unresolved


def rebuild_pinned(pinned: pl.DataFrame) -> pl.DataFrame:
    """Re-read exactly the recorded revisions: the reproducible rebuild."""
    ids = pinned.filter(pl.col("source_revid").is_not_null())
    pages = fetch_revisions(ids["source_revid"].to_list())
    rows = []
    for rec in ids.iter_rows(named=True):
        page = pages.get(rec["source_revid"])
        parsed = coaches_from_page(page.text, rec["season"]) if page else None
        if parsed is None:
            raise RuntimeError(
                f"pinned revision {rec['source_revid']} ({rec['source_title']}) "
                "no longer parses -- the parser changed, not the source"
            )
        rows.append(
            {
                "season": rec["season"],
                "school_mascot": rec["school_mascot"],
                **parsed,
                "source_title": page.title,
                "source_revid": page.revid,
            }
        )
    return pl.DataFrame(rows, schema=COLUMNS)


# --- command line --------------------------------------------------------------

TABLE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / ("cfb_matchup_coordinators_wikipedia.csv")
)


def load_table() -> pl.DataFrame:
    """The committed, revision-pinned Wikipedia coordinator table."""
    return pl.read_csv(TABLE, schema_overrides=COLUMNS)


def fbs_schools(season: int) -> dict[str, str]:
    """CFBD school -> mascot for every program with a head coach that FBS season."""
    from cfb_data_build.matchup_side import _cfbd

    mascots = {t["school"]: t.get("mascot") or "" for t in _cfbd("/teams")}
    coaches = _cfbd(f"/coaches?minYear={season}&maxYear={season}")
    return {
        s["school"]: mascots.get(s["school"], "")
        for c in coaches
        for s in c.get("seasons") or []
        if s.get("year") == season
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="cfb_data_build.coordinators_wiki",
        description=f"(re)build data/{TABLE.name} from Wikipedia team-season articles",
    )
    sub = p.add_subparsers(dest="command", required=False)
    b = sub.add_parser(
        "backfill", help="read the CURRENT revisions and record their ids"
    )
    b.add_argument("--start-season", type=int, required=True)
    b.add_argument("--end-season", type=int, required=True)
    sub.add_parser(
        "pinned",
        help="re-read exactly the revisions recorded in the table (reproducible)",
    )
    args = p.parse_args(argv)
    if args.command is None:
        p.print_help()
        return 0

    if args.command == "pinned":
        rebuilt = rebuild_pinned(load_table())
        rebuilt.sort(["season", "school_mascot"]).write_csv(TABLE)
        print(f"{TABLE.name}: rebuilt {rebuilt.height} rows from pinned revisions")
        return 0

    frames = []
    for season in range(args.start_season, args.end_season + 1):
        rows, missing = build_season(season, fbs_schools(season))
        frames.append(rows)
        print(
            f"{season}: {rows.height} school-seasons, unresolved {missing}", flush=True
        )
    new = pl.concat(frames, how="vertical")
    if TABLE.exists():
        # replace at the (season, school_mascot) key, never a whole season: a school
        # whose article did not resolve this run (a transient MediaWiki miss lands
        # in `unresolved`, not in `rows`) keeps the row it already has
        keep = load_table().join(
            new.select("season", "school_mascot"),
            on=["season", "school_mascot"],
            how="anti",
        )
        new = pl.concat([keep, new], how="vertical")
    new.sort(["season", "school_mascot"]).write_csv(TABLE)
    print(
        f"{TABLE.name}: {new.height} rows, {new['season'].min()}-{new['season'].max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
