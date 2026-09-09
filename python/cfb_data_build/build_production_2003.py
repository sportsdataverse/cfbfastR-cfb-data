"""Build the 2003 offensive production table used to seed returning production 2004.

ESPN's player box starts in 2004, so `cfb_returning_production(2004)` has no
season-2003 production to draw on and raises SeasonNotFoundError. CFBD's
play-by-play DOES start in 2003 (2002 returns nothing), and its play text is
name-tagged by team, so 2003 offensive production is recoverable by parsing it.

Names are resolved to ESPN athlete ids against the 2004 ESPN roster, which is the
id space `_returning_from_frames` joins on. A 2003 producer who is NOT on the 2004
roster keeps a synthetic id: they must still land in the DENOMINATOR (they were
production that did not return), and must never collide with a real athlete id.

2003 is immutable, so this runs once and the parquet is vendored.
"""

from __future__ import annotations
import collections
import json
import os
import re
import unicodedata
import urllib.request
from pathlib import Path
import polars as pl
from sportsdataverse.cfb import load_cfb_rosters


def _key() -> str:
    """CFBD key from the environment, falling back to ~/.Renviron.

    Only R reads .Renviron at startup, so a Python run on the same box does not
    inherit CFBD_API_KEY unless it is also an OS env var -- hence the fallback.
    """
    key = os.environ.get("CFBD_API_KEY")
    if key:
        return key
    for path in (
        os.path.expanduser("~/.Renviron"),
        os.path.expanduser("~/Documents/.Renviron"),
    ):
        if os.path.exists(path):
            for line in open(path):
                if line.startswith("CFBD_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("CFBD_API_KEY not set (env or ~/.Renviron)")


H = {"Authorization": f"Bearer {_key()}", "User-Agent": "Mozilla/5.0"}


def _get(url):
    return json.load(
        urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=180)
    )


def norm(n: str) -> str:
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace("'", "").replace("-", " ")
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


# CFBD school -> ESPN team_location, for the names that differ
ALIASES = {
    "San José State": "San Jose State",
    "UConn": "Connecticut",
    "UL Monroe": "Louisiana Monroe",
    "Sam Houston": "Sam Houston State",
    "Jacksonville State": "Jacksonville State",
    "Massachusetts": "UMass",
    "UT San Antonio": "UTSA",
    "Hawai'i": "Hawai'i",
    "Miami": "Miami",
}
NAME = r"([A-Z][A-Za-z'\-\.]+(?: [A-Z][A-Za-z'\-\.]+)+) \([A-Z0-9&\-]{2,8}\)"


def production_2003() -> tuple[dict[tuple[str, str], float], list[str]]:
    """Parse every 2003 week, reporting which requests failed.

    Failures are collected rather than swallowed: this artifact is vendored and
    committed, so one transient CFBD failure across the 18 requests would become
    permanently missing production for that week. `MIN_TEAM_YARDS` does not
    catch it either -- a uniformly short build lowers every team together and
    stays above the floor.
    """
    prod: dict[tuple[str, str], float] = collections.defaultdict(float)
    failed: list[str] = []
    for stype, weeks in (("regular", range(1, 17)), ("postseason", range(1, 3))):
        for wk in weeks:
            try:
                plays = _get(
                    f"https://api.collegefootballdata.com/plays?year=2003&week={wk}&seasonType={stype}"
                )
            except Exception as exc:  # noqa: BLE001 - reported below, never swallowed
                print(f"  wk {stype}/{wk}: {exc}")
                failed.append(f"{stype}/{wk}")
                continue
            for p in plays:
                text = p.get("playText") or ""
                off = p.get("offense")
                yds = abs(p.get("yardsGained") or 0)
                m = re.match(r"\s*" + NAME + r"\s+rushed", text)
                if m:
                    prod[(off, norm(m.group(1)))] += yds
                    continue
                m = re.match(r"\s*" + NAME + r"\s+pass", text)
                if m:
                    prod[(off, norm(m.group(1)))] += yds
                    m2 = re.search(r"complete to " + NAME, text)
                    if m2:
                        prod[(off, norm(m2.group(1)))] += yds
            print(f"  {stype} wk{wk}: {len(plays)} plays", flush=True)
    return prod, failed


def main() -> None:
    prod, failed = production_2003()
    if failed:
        raise SystemExit(f"refusing to vendor a partial table; weeks failed: {failed}")
    print(f"\n2003 producers: {len(prod)} across {len({k[0] for k in prod})} schools")

    r = load_cfb_rosters(2004)
    if not isinstance(r, pl.DataFrame):
        r = pl.from_pandas(r)
    r = r.filter(pl.col("full_name").is_not_null())
    roster: dict[str, dict[str, int]] = collections.defaultdict(dict)
    loc2tid: dict[str, int] = {}
    for loc, fn, tid in r.select("team_location", "full_name", "team_id").iter_rows():
        roster[loc][norm(fn)] = tid and int(tid)
        loc2tid[loc] = int(tid)
    ath: dict[tuple[str, str], int] = {}
    for loc, fn, aid in r.select(
        "team_location", "full_name", "athlete_id"
    ).iter_rows():
        ath[(loc, norm(fn))] = int(aid)

    rows, unmapped, matched = [], collections.Counter(), 0
    for (school, nm), yds in prod.items():
        loc = ALIASES.get(school, school)
        if loc not in loc2tid:
            unmapped[school] += 1
            continue
        aid = ath.get((loc, nm))
        rows.append(
            {
                "season": 2003,
                "team_id": str(loc2tid[loc]),
                "player_id": str(aid) if aid is not None else f"cfbd2003:{loc}:{nm}",
                "player_name": nm,
                "unit": "offense",
                "prod_weight": float(yds),
                "position": None,
            }
        )
        matched += aid is not None
    # Teams CFBD only saw in a game or two produce a fraction off a near-empty
    # denominator: five FCS schools (The Citadel, Murray State, Indiana State,
    # Furman, Sacramento State) carried 224-522 yards against a 5,158 median and
    # came out at 0.00-0.05. A 1000-yard floor drops exactly those and leaves
    # Pittsburgh's legitimate 0.042 (Rutherford AND Fitzgerald both departed).
    MIN_TEAM_YARDS = 1000.0
    df = pl.DataFrame(
        rows,
        schema={
            "season": pl.Int64,
            "team_id": pl.Utf8,
            "player_id": pl.Utf8,
            "player_name": pl.Utf8,
            "unit": pl.Utf8,
            "prod_weight": pl.Float64,
            "position": pl.Utf8,
        },
    )
    # A player whose every 2003 play gained exactly zero carries no production.
    # They cancel out of the ret/tot fraction but would still be counted in
    # `n_returning`, publishing a contributor who contributed nothing.
    df = df.filter(pl.col("prod_weight") > 0)
    keep = (
        df.group_by("team_id")
        .agg(pl.col("prod_weight").sum().alias("y"))
        .filter(pl.col("y") >= MIN_TEAM_YARDS)["team_id"]
        .to_list()
    )
    df = df.filter(pl.col("team_id").is_in(keep))
    print(
        f"rows: {df.height}  resolved to ESPN athlete_id: {matched} ({100 * matched / max(df.height, 1):.1f}%)"
    )
    print(f"schools unmapped: {sorted(unmapped)}")
    print(
        f"teams: {df['team_id'].n_unique()}   total yards: {df['prod_weight'].sum():,.0f}"
    )
    # Anchored on this file, not the cwd: a rebuild launched from python/ would
    # otherwise write python/data/, leave the vendored copy stale, and let the
    # contract tests keep passing against the old artifact (CodeRabbit on #76).
    out = Path(__file__).resolve().parents[2] / "data" / "cfb_production_2003.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
