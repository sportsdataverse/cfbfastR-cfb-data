"""``cfb_schedules`` -- the unified per-season schedule the loaders read.

Two schedule datasets existed and they were crossed: ``espn_cfb_schedules``
(built here from ``final.json``, FBS-scoped, read by nobody) and
``cfb_schedules`` (read by sdv-py ``load_cfb_schedule`` and by nothing that
produced it). This module is the producer the ``cfb_schedules`` tag never had.

**Everything in this dataset is ESPN data.** Both inputs are ESPN: one is this
repo's own ESPN-native ``schedules`` artifact, the other is the
CollegeFootballData ``/games`` endpoint -- and CollegeFootballData is a
*redistributor* of ESPN, not an independent source. ``game_id`` is an ESPN id on
both paths, which is exactly why they join on it. The two feeds are therefore
read for COVERAGE, not for provenance: the redistributed feed reaches every
division and every season type, the native artifact reaches the games ESPN
scheduled but never played. No column is attributed to CollegeFootballData
anywhere in this dataset's documentation, because no column originates there.

Grain: one row per ``game_id``.

**Rows** are the redistributed superset -- every division (``fbs``/``fcs``/
``ii``/``iii``) and every season type, including 2020's ``spring_*`` COVID
season -- UNIONed with the games only the native ESPN artifact carries. Those
extra rows are real: in 2020 they are 121 COVID postponements/cancellations the
redistributed feed drops because they were never played (56
``STATUS_POSTPONED`` + 65 ``STATUS_CANCELED``) plus one offseason all-star game.
A scheduled-then-cancelled game is a schedule fact, so it ships with ``status``
telling you what happened: ``status`` in (``STATUS_POSTPONED``,
``STATUS_CANCELED``) plus ``season_type == "offseason"`` is what identifies
those 122 rows.

**Columns** are the union of both feeds MINUS the seven modeling columns
(``*_pregame_elo``, ``*_postgame_elo``, ``*_post_win_prob``,
``excitement_index``) -- those are model output rather than ESPN schedule facts,
and belong to the modeling datasets.

Where the two feeds overlap semantically the redistributed value is taken first
and the native artifact is coalesced in as the fallback, so no near-duplicate
column pair ships:

===========================  ==========================  =======================
unified column               redistributed key           native ESPN fallback
===========================  ==========================  =======================
``home_points``              ``homePoints``              ``home_score``
``away_points``              ``awayPoints``              ``away_score``
``completed``                ``completed``               ``status == FINAL``
``start_date``               ``startDate``               ``game_date``
``conference_game``          ``conferenceGame``          ``conference_competition``
===========================  ==========================  =======================

``game_date`` is dropped as a column because it is ``start_date`` at coarser
string precision (``2023-08-26T18:30Z`` vs ``2023-08-26T18:30:00.000Z``) --
identical instants on all 911 shared 2023 rows. ``conference_competition`` is
KEPT alongside ``conference_game`` because the two measurably disagree (12 of
911 shared 2023 rows): one is derived from conference membership, the other
flags the competition itself.

**Season type ships twice, because ESPN publishes it twice.** ``season_type_id``
is ESPN's integer and ``season_type`` its label; the two are a strict 1:1
mapping (see ``_SEASON_TYPES``, taken verbatim from ESPN's own
``/seasons/{year}/types`` resource).

**FBS filterability.** ``home_division``/``away_division`` are kept and two
booleans are derived. The division values are ``fbs``/``fcs``/``ii``/``iii``
**and null** (2023: 21 null home, 61 null away) -- a null is a genuinely
non-FBS team outside ESPN's group-80/81 universe, not missing data, so it must
evaluate FALSE and never null. That is what ``.eq_missing()`` gives and what a
bare ``== "fbs"`` does not.

``playoff_*`` columns are snake_cased here (``playoff_round_name``, not the
``playoff_roundName`` cfbfastR's flatten emitted) per the repo column
convention.

Inputs, both over HTTP, never a local checkout:

* CollegeFootballData ``/games?year=&seasonType=both`` (ESPN data,
  redistributed) -- needs ``CFBD_API_KEY``.
* ESPN: this repo's own ``schedules`` artifact, local first (the daily driver
  builds it minutes earlier) then the published ``espn_cfb_schedules`` release.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

from cfb_data_build.config import DatasetSpec
from cfb_data_build.io import gzip_csv, write_dataset

CFBD_GAMES = "https://api.collegefootballdata.com/games"
ESPN_RELEASE = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
    "/espn_cfb_schedules/cfb_schedule_{season}.parquet"
)
_UA = {"User-Agent": "Mozilla/5.0 (compatible; sportsdataverse/cfb-data)"}

SPEC = DatasetSpec("cfb_schedules", "cfb_schedules", "cfb_schedules")
SPECS: dict[str, DatasetSpec] = {"cfb_schedules": SPEC}

#: The seven modeling columns this dataset deliberately does not ship --
#: model output, not ESPN schedule facts.
EXCLUDED_CFBD = (
    "home_pregame_elo",
    "away_pregame_elo",
    "home_postgame_elo",
    "away_postgame_elo",
    "home_post_win_prob",
    "away_post_win_prob",
    "excitement_index",
)

#: CFBD payload key -> unified column. Anything not listed is dropped
#: (``*LineScores`` has its own dataset; the elo/win-prob/excitement keys are
#: EXCLUDED_CFBD).
_CFBD_FIELDS = {
    "id": "game_id",
    "season": "season",
    "week": "week",
    "seasonType": "season_type",
    "startDate": "start_date",
    "startTimeTBD": "start_time_tbd",
    "completed": "completed",
    "neutralSite": "neutral_site",
    "conferenceGame": "conference_game",
    "attendance": "attendance",
    "venueId": "venue_id",
    "venue": "venue",
    "homeId": "home_id",
    "homeTeam": "home_team",
    "homeClassification": "home_division",
    "homeConference": "home_conference",
    "homePoints": "home_points",
    "awayId": "away_id",
    "awayTeam": "away_team",
    "awayClassification": "away_division",
    "awayConference": "away_conference",
    "awayPoints": "away_points",
    "highlights": "highlights",
    "notes": "notes",
}

#: Nested ``playoff`` object key -> unified column.
_PLAYOFF_FIELDS = {
    "competition": "playoff_competition",
    "format": "playoff_format",
    "round": "playoff_round",
    "roundName": "playoff_round_name",
    "bracketSlot": "playoff_bracket_slot",
    "homeSeed": "playoff_home_seed",
    "awaySeed": "playoff_away_seed",
    "bowlName": "playoff_bowl_name",
}

#: ESPN's canonical season types, verbatim from
#: ``sports.core.api.espn.com/v2/.../seasons/{year}/types``: id -> snake_cased
#: name. 5/6 ("Spring Regular Season" / "Spring Postseason") exist only in
#: 2020. The redistributed feed publishes the SAME labels -- one more reason
#: to read it as ESPN: both sides of the union speak this one vocabulary.
_SEASON_TYPES = {
    1: "preseason",
    2: "regular",
    3: "postseason",
    4: "offseason",
    5: "spring_regular",
    6: "spring_postseason",
}
_SEASON_TYPE_IDS = {name: ident for ident, name in _SEASON_TYPES.items()}

#: Declared so every season's parquet carries one identical schema even when a
#: source is empty, and so ids are pinned to ONE dtype (Int64) at the boundary
#: rather than inheriting CFBD's Int32 on one path and ESPN's Int64 on the other.
SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64,
    "season": pl.Int64,
    "week": pl.Int64,
    "season_type": pl.Utf8,
    "season_type_id": pl.Int64,
    "start_date": pl.Utf8,
    "start_time_tbd": pl.Boolean,
    "completed": pl.Boolean,
    "neutral_site": pl.Boolean,
    "conference_game": pl.Boolean,
    "conference_competition": pl.Boolean,
    "attendance": pl.Int64,
    "venue_id": pl.Int64,
    "venue": pl.Utf8,
    "status": pl.Utf8,
    "home_id": pl.Int64,
    "home_team": pl.Utf8,
    "home_abbreviation": pl.Utf8,
    "home_division": pl.Utf8,
    "home_conference": pl.Utf8,
    "home_points": pl.Int64,
    "home_winner": pl.Boolean,
    "away_id": pl.Int64,
    "away_team": pl.Utf8,
    "away_abbreviation": pl.Utf8,
    "away_division": pl.Utf8,
    "away_conference": pl.Utf8,
    "away_points": pl.Int64,
    "away_winner": pl.Boolean,
    "fbs_game": pl.Boolean,
    "fbs_participant": pl.Boolean,
    "highlights": pl.Utf8,
    "notes": pl.Utf8,
    "playoff_competition": pl.Utf8,
    "playoff_format": pl.Utf8,
    "playoff_round": pl.Utf8,
    "playoff_round_name": pl.Utf8,
    "playoff_bracket_slot": pl.Utf8,
    "playoff_home_seed": pl.Int64,
    "playoff_away_seed": pl.Int64,
    "playoff_bowl_name": pl.Utf8,
}

#: Columns whose ESPN value is only a FALLBACK for the CFBD one (never a
#: second shipped column).
_COALESCE = {
    "week": "week",
    "start_date": "game_date",
    "neutral_site": "neutral_site",
    "conference_game": "conference_competition",
    "attendance": "attendance",
    "venue": "venue",
    "home_id": "home_id",
    "home_team": "home_team",
    "home_points": "home_score",
    "away_id": "away_id",
    "away_team": "away_team",
    "away_points": "away_score",
}


def _get(url: str, *, headers: dict[str, str] | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={**_UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https hosts
        return resp.read()


def fetch_cfbd_games(season: int, *, api_key: str | None = None, url: str = CFBD_GAMES) -> list[dict]:
    """Every CFBD game for a season, regular + postseason in one call."""
    key = api_key or os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set -- cfb_schedules reads the CFBD /games endpoint. "
            "Export it (CI: the repo secret) before building this dataset."
        )
    raw = _get(
        f"{url}?year={season}&seasonType=both",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    payload = json.loads(raw)
    return payload if isinstance(payload, list) else []


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cfbd_row(game: dict) -> dict[str, Any]:
    row: dict[str, Any] = {out: game.get(src) for src, out in _CFBD_FIELDS.items()}
    playoff = game.get("playoff") or {}
    for src, out in _PLAYOFF_FIELDS.items():
        row[out] = playoff.get(src)
    for col in (
        "game_id",
        "season",
        "week",
        "attendance",
        "venue_id",
        "home_id",
        "away_id",
        "home_points",
        "away_points",
        "playoff_home_seed",
        "playoff_away_seed",
    ):
        row[col] = _int(row.get(col))
    row["season_type"] = None if row["season_type"] is None else str(row["season_type"])
    return row


def tidy_cfbd(games: list[dict]) -> pl.DataFrame:
    """Tidy the CFBD ``/games`` payload. Pure -- no network, no disk."""
    cols = [c for c in SCHEMA if c in set(_CFBD_FIELDS.values()) | set(_PLAYOFF_FIELDS.values())]
    schema = {c: SCHEMA[c] for c in cols}
    if not games:
        return pl.DataFrame(schema=schema)
    return pl.from_dicts([_cfbd_row(g) for g in games], schema=schema)


def load_espn(season: int, *, base: str | Path = "cfb", release: str = ESPN_RELEASE) -> pl.DataFrame:
    """The ESPN-native schedule for a season: local artifact first, release second.

    The daily driver builds ``schedules`` minutes before this stage, so the local
    parquet is the freshest copy; a standalone or backfill run has no local copy
    and falls back to the published ``espn_cfb_schedules`` asset. A season ESPN
    never covered (pre-2004) yields an empty frame rather than an error.
    """
    local = Path(base) / "schedules" / "parquet" / f"cfb_schedule_{season}.parquet"
    if local.exists():
        return pl.read_parquet(local)
    try:
        return pl.read_parquet(io.BytesIO(_get(release.format(season=season))))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return pl.DataFrame()
        raise


def _espn_normalized(espn: pl.DataFrame) -> pl.DataFrame:
    """ESPN frame with pinned id dtypes and the season_type vocabulary mapped."""
    if espn.height == 0:
        return espn
    out = espn.with_columns(
        [
            pl.col(c).cast(pl.Int64)
            for c in (
                "game_id",
                "season",
                "week",
                "home_id",
                "away_id",
                "home_score",
                "away_score",
                "attendance",
            )
            if c in espn.columns
        ]
    )
    if "season_type" in out.columns:
        out = out.with_columns(
            pl.col("season_type")
            .cast(pl.Int64)
            .replace_strict(_SEASON_TYPES, default=None, return_dtype=pl.Utf8)
            .alias("espn_season_type")
        ).drop("season_type")
    return out


def unify(cfbd: pl.DataFrame, espn: pl.DataFrame, season: int) -> pl.DataFrame:
    """Join CFBD (canonical) with ESPN (enrichment + fallback), then union the
    ESPN-only rows. Pure -- no network, no disk."""
    espn = _espn_normalized(espn)
    if espn.height == 0:
        espn = pl.DataFrame(schema={"game_id": pl.Int64})
    assert cfbd.schema["game_id"] == espn.schema["game_id"], (
        f"game_id dtype disagrees across sources: {cfbd.schema['game_id']} vs {espn.schema['game_id']}"
    )
    espn_only_ids = set(espn["game_id"].to_list()) - set(cfbd["game_id"].to_list())

    df = cfbd.join(espn, on="game_id", how="left", suffix="_espn")
    if espn_only_ids:
        extra = espn.filter(pl.col("game_id").is_in(list(espn_only_ids)))
        df = pl.concat([df, extra], how="diagonal_relaxed")

    def espn_col(name: str) -> pl.Expr:
        """The ESPN column, wherever the join put it (suffixed on collision)."""
        for cand in (f"{name}_espn", name):
            if cand in df.columns:
                return pl.col(cand)
        return pl.lit(None)

    df = df.with_columns(
        [pl.coalesce(pl.col(dst), espn_col(src)).alias(dst) for dst, src in _COALESCE.items()]
    ).with_columns(
        season=pl.lit(season, dtype=pl.Int64),
        season_type=pl.coalesce(pl.col("season_type"), espn_col("espn_season_type")),
        status=espn_col("status"),
        conference_competition=espn_col("conference_competition"),
        home_abbreviation=espn_col("home_abbreviation"),
        away_abbreviation=espn_col("away_abbreviation"),
        completed=pl.coalesce(pl.col("completed"), espn_col("status") == "STATUS_FINAL"),
    ).with_columns(
        # Derived from the label, so the pair is 1:1 by construction; verified
        # against ESPN's own integer on every joined row (they never disagree).
        season_type_id=pl.col("season_type").replace_strict(
            _SEASON_TYPE_IDS, default=None, return_dtype=pl.Int64
        ),
    )

    # A winner ESPN did not state is recoverable from a completed game's points;
    # without this 3/4 of the rows would carry a null winner.
    def winner(mine: str, theirs: str, espn_flag: str) -> pl.Expr:
        return pl.coalesce(
            espn_col(espn_flag),
            pl.when(
                (pl.col("completed") == True)  # noqa: E712 - explicit per repo convention
                & pl.col(mine).is_not_null()
                & pl.col(theirs).is_not_null()
            )
            .then(pl.col(mine) > pl.col(theirs))
            .otherwise(None),
        )

    df = df.with_columns(
        home_winner=winner("home_points", "away_points", "home_winner"),
        away_winner=winner("away_points", "home_points", "away_winner"),
        # eq_missing, not `== "fbs"`: division is null for teams outside ESPN's
        # group-80/81 universe and a null must read FALSE, never null.
        fbs_game=pl.col("home_division").eq_missing("fbs") & pl.col("away_division").eq_missing("fbs"),
        fbs_participant=pl.col("home_division").eq_missing("fbs") | pl.col("away_division").eq_missing("fbs"),
    )

    for col, dtype in SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return df.select([pl.col(c).cast(t) for c, t in SCHEMA.items()]).sort("game_id")


def build_schedules(
    season: int,
    *,
    base: str | Path = "cfb",
    api_key: str | None = None,
    **_: Any,
) -> pl.DataFrame:
    return unify(
        tidy_cfbd(fetch_cfbd_games(season, api_key=api_key)),
        load_espn(season, base=base),
        season,
    )


def build(
    start_year: int,
    end_year: int,
    *,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
) -> list[tuple[int, str]]:
    """Build (and optionally publish) cfb_schedules across a season range.

    Every season is isolated: a failure is recorded and the sweep continues, so
    one bad season cannot abort a backfill. Returns the ``(season, error)``
    failures.
    """
    failures: list[tuple[int, str]] = []
    for season in range(start_year, end_year + 1):
        try:
            df = build_schedules(season, base=base)
            print(
                f"  {SPEC.dataset} {season}: {df.height} rows, {df.width} cols, "
                f"fbs_game={df['fbs_game'].sum()}, fbs_participant={df['fbs_participant'].sum()}, "
                f"unplayed={df.filter(pl.col('status').is_in(['STATUS_POSTPONED', 'STATUS_CANCELED'])).height}",
                flush=True,
            )
            if dry_run:
                continue
            # The tag already ships `.csv.gz`, matching the sibling ESPN tags.
            gzip_csv(write_dataset(df, SPEC.dataset, season, SPEC.stem, base=base))
            if publish:
                from cfb_data_build.publish import publish_dataset

                publish_dataset(SPEC, season, base=base)
        except Exception as exc:  # noqa: BLE001 - one season must not kill the sweep
            print(
                f"  {SPEC.dataset} {season}: FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
            failures.append((season, f"{type(exc).__name__}: {exc}"))
    return failures
