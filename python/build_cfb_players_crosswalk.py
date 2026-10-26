"""ESPN athlete <-> CBS / Fox / Yahoo player crosswalk per CFB season, from the parsed alt history.

The CFB sibling of hoopR-nba-data's ``--fill-alt`` and nfl-raw's ``nfl_espn_03_alt_players.py``.
For each season it matches every provider with a parsed season in the private
``cfbfastR-cfb-alt-data`` checkout (written by that repo's ``scripts/parse_alt.sh``) to ESPN
through sdv-py's ``sportsdataverse._crosswalk_alt.match_season``:

* ESPN games come from ``cfb/schedules/parquet/cfb_schedule_{season}.parquet`` (kickoff UTC ->
  US-Eastern date) and ESPN athletes from ``cfb/game_rosters/parquet/game_rosters_{season}.parquet``.
* Teams use the provider's known ESPN team map where sdv-py ships one (CBS), else pair by
  (date, side) co-occurrence; games join on (Eastern date, team pair); players match inside each
  (game, team) on normalized name with a jersey tie-break, voted across the season.

Writes ``{--out}/cfb_players_crosswalk_{season}.parquet`` (default ``cfb/crosswalk/parquet``): one
row per (ESPN team, ESPN athlete) matched by at least one provider. A rebuilt season that would
hold fewer ids in any provider column than the file already in ``--out`` is refused (never drop a
leg). ``--upload`` merges a ``players`` row per season into ``cfb_crosswalk_in_data_repo.csv`` and
publishes to the ``cfb_crosswalk`` tag through ``build_cfb_crosswalk.upload_assets`` (only the
parquets whose sha256 differs from the release's).

Runs on this repo's ``.venv`` (its sdv-py pin ships ``_crosswalk_alt``); driven by
``scripts/cron_alt_fill.sh``::

    .venv/bin/python python/build_cfb_players_crosswalk.py -s 2016 -e 2026 \\
        [--alt-root ../cfbfastR-cfb-alt-data] [--out DIR] [--upload]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from sportsdataverse._common_crosswalk_basketball import normalize_name, to_eastern
from sportsdataverse._crosswalk_alt import known_teams, match_season, read_season

ROOT = Path(__file__).resolve().parent.parent
CFB = ROOT / "cfb"
OUT = CFB / "crosswalk" / "parquet"
log = logging.getLogger("cfb_players_crosswalk")

#: Provider -> (id column, name column) in the output; providers absent from the alt root stay null.
LEGS = {
    "cbs": ("cbs_player_id", "cbs_player_name"),
    "fox": ("fox_athlete_id", "fox_player_name"),
    "yahoo": ("yahoo_player_id", "yahoo_player_name"),
}
LEGS_IDS = [cols[0] for cols in LEGS.values()]
KEYS = ["espn_team_id", "espn_athlete_id"]
SCHEMA = {
    "season": pl.Int32,
    "espn_team_id": pl.Int32,
    "espn_athlete_id": pl.Utf8,
    "espn_player_name": pl.Utf8,
    **{c: pl.Utf8 for cols in LEGS.values() for c in cols},
}


def _join(left: pl.DataFrame, right: pl.DataFrame, how: str) -> pl.DataFrame:
    """Join on ``KEYS`` after asserting both sides carry the same key dtypes."""
    for k in KEYS:
        assert left.schema[k] == right.schema[k], (
            f"{k}: {left.schema[k]} vs {right.schema[k]}"
        )
    return left.join(right, on=KEYS, how=how, coalesce=True)


def espn_side(season: int, cfb: Path = CFB) -> tuple[pl.DataFrame, pl.DataFrame]:
    """ESPN games (Eastern date) and game-roster athletes for one season."""
    schedule = pl.read_parquet(
        cfb / "schedules" / "parquet" / f"cfb_schedule_{season}.parquet"
    )
    games = schedule.select(
        pl.col("game_id").cast(pl.Int64).cast(pl.Utf8).alias("espn_game_id"),
        pl.col("game_date")
        .map_elements(to_eastern, return_dtype=pl.Date)
        .alias("game_date"),
        pl.col("home_id").cast(pl.Int32).alias("home_espn_team_id"),
        pl.col("away_id").cast(pl.Int32).alias("away_espn_team_id"),
    ).unique("espn_game_id", keep="first", maintain_order=True)
    rosters = (
        pl.read_parquet(
            cfb / "game_rosters" / "parquet" / f"game_rosters_{season}.parquet"
        )
        .select(
            pl.col("game_id").cast(pl.Int64).cast(pl.Utf8).alias("espn_game_id"),
            pl.col("team_id").cast(pl.Int32).alias("espn_team_id"),
            pl.col("athlete_id").cast(pl.Int64).cast(pl.Utf8).alias("espn_athlete_id"),
            pl.col("athlete_display_name").cast(pl.Utf8).alias("espn_full_name"),
            pl.col("jersey").cast(pl.Utf8).alias("espn_jersey"),
        )
        .filter(
            pl.col("espn_team_id").is_not_null()
            & pl.col("espn_athlete_id").is_not_null()
        )
    )
    return games, rosters


def _norm(col: str) -> pl.Expr:
    return pl.col(col).map_elements(normalize_name, return_dtype=pl.Utf8)


def drop_ambiguous(leg: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """Where one provider id voted onto 2+ athletes of a team, keep only the exact-name ones.

    ESPN sometimes carries one player under two athlete ids (both names agree: both kept), but the
    vote also lands a provider id on a teammate sharing a surname ("Will Smith" -> "Josh Smith").
    A group with no exact-name athlete is left as voted.
    """
    id_col, name_col = leg.columns[2], leg.columns[3]
    leg = _join(leg, names, "left").with_columns(
        (_norm("espn_player_name") == _norm(name_col)).fill_null(False).alias("_exact")
    )
    group = ["espn_team_id", id_col]
    keep = (
        (pl.len().over(group) == 1)
        | pl.col("_exact")
        | ~pl.col("_exact").any().over(group)
    )
    return leg.filter(keep).select(*KEYS, id_col, name_col)


def combine(
    season: int, games: pl.DataFrame, rosters: pl.DataFrame, legs: list[pl.DataFrame]
) -> pl.DataFrame:
    """Full-join the per-provider legs on (team, athlete), add the ESPN name, conform to ``SCHEMA``."""
    # The athlete's name as ESPN showed it in his latest game for the team.
    assert rosters.schema["espn_game_id"] == games.schema["espn_game_id"], (
        "espn_game_id dtype mismatch"
    )
    names = (
        rosters.join(
            games.select("espn_game_id", "game_date"), on="espn_game_id", how="left"
        )
        .sort("game_date", nulls_last=False, maintain_order=True)
        .unique(KEYS, keep="last", maintain_order=True)
        .select(*KEYS, pl.col("espn_full_name").alias("espn_player_name"))
    )
    legs = [drop_ambiguous(leg, names) for leg in legs]
    out = legs[0]
    for leg in legs[1:]:
        out = _join(out, leg, "full")
    out = _join(out, names, "left").with_columns(
        pl.lit(season, dtype=pl.Int32).alias("season")
    )
    out = out.with_columns(
        pl.lit(None, dtype=t).alias(c)
        for c, t in SCHEMA.items()
        if c not in out.columns
    ).select([pl.col(c).cast(t) for c, t in SCHEMA.items()])
    dupes = out.height - out.select(KEYS).n_unique()
    assert dupes == 0, (
        f"cfb {season}: {dupes} duplicate (espn_team_id, espn_athlete_id) rows"
    )
    return out.sort(KEYS)


def team_map(provider: str, season: int, cfb: Path = CFB) -> pl.DataFrame | None:
    """Fixed ESPN -> provider team pairs: sdv-py's CBS map, else this repo's ``cfb_teams_crosswalk``.

    Co-occurrence alone cannot pair CFB teams (every team plays every Saturday), so Fox and Yahoo
    take their ids from the season's published team crosswalk. Ids listed twice on either side are
    dropped rather than guessed.
    """
    if provider == "cbs":
        return known_teams(provider, "cfb")
    path = cfb / "crosswalk" / "parquet" / f"cfb_teams_crosswalk_{season}.parquet"
    if not path.exists():
        return None
    col = f"{provider}_team_id"
    pairs = (
        pl.read_parquet(path, columns=["espn_team_id", col])
        .drop_nulls()
        .select(
            pl.col("espn_team_id").cast(pl.Int32),
            pl.col(col).cast(pl.Utf8).alias("team_id"),
        )
        .unique()
    )
    return pairs.filter(
        ~pl.col("espn_team_id").is_duplicated() & ~pl.col("team_id").is_duplicated()
    )


def build_season(
    season: int, alt_root: Path, cfb: Path = CFB
) -> tuple[pl.DataFrame | None, dict]:
    """One season's crosswalk (``None`` when no provider has a parsed season) and per-provider stats."""
    games, rosters = espn_side(season, cfb)
    legs, stats = [], {}
    for provider, (id_col, name_col) in LEGS.items():
        alt = read_season(alt_root, provider, "cfb", season)
        if alt is None:
            continue
        m = match_season(
            games, rosters, alt, known_teams=team_map(provider, season, cfb)
        )
        stats[provider] = m["stats"]
        log.info("cfb %s %s: %s", season, provider, m["stats"])
        legs.append(
            m["players"].select(
                pl.col("espn_team_id").cast(pl.Int32),
                pl.col("espn_athlete_id").cast(pl.Utf8),
                pl.col("player_id").alias(id_col),
                pl.col("player_name").alias(name_col),
            )
        )
    return (combine(season, games, rosters, legs) if legs else None), stats


def shrunk(old: pl.DataFrame, new: pl.DataFrame) -> dict[str, tuple[int, int]]:
    """Provider id columns holding fewer non-null ids in ``new`` than in ``old``: {col: (old, new)}."""
    counts = {
        c: (old[c].drop_nulls().len(), new[c].drop_nulls().len())
        for c in LEGS_IDS
        if c in old.columns
    }
    return {c: v for c, v in counts.items() if v[1] < v[0]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-s", "--start", type=int, required=True)
    ap.add_argument("-e", "--end", type=int, required=True)
    ap.add_argument(
        "--alt-root", type=Path, default=ROOT.parent / "cfbfastR-cfb-alt-data"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="output dir (default: cfb/crosswalk/parquet)",
    )
    ap.add_argument(
        "--upload",
        action="store_true",
        help="publish changed seasons to the release tag",
    )
    ap.add_argument("--release-tag", default="cfb_crosswalk")
    ap.add_argument("--repo", default="sportsdataverse/sportsdataverse-data")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    a.out.mkdir(parents=True, exist_ok=True)
    rc, produced, manifest = 0, [], []
    for season in range(a.start, a.end + 1):
        out, _ = build_season(season, a.alt_root)
        if out is None or out.height == 0:
            log.info(
                "cfb %s: no parsed provider season with matched players; skipped",
                season,
            )
            continue
        path = a.out / f"cfb_players_crosswalk_{season}.parquet"
        if path.exists():
            lost = shrunk(pl.read_parquet(path), out)
            if lost:
                log.error(
                    "cfb %s: rebuild would drop provider ids %s; refused", season, lost
                )
                rc = 1
                continue
        out.write_parquet(path, compression="zstd")
        produced.append(path)
        log.info(
            "cfb %s: %d rows, %s -> %s",
            season,
            out.height,
            {c: out[c].drop_nulls().len() for c in LEGS_IDS},
            path,
        )
        if a.upload:
            from build_cfb_crosswalk import _row

            manifest.append(
                _row("players", season, out, "build_cfb_players_crosswalk.py")
            )
    if a.upload and produced:
        from build_cfb_crosswalk import update_manifest, upload_assets

        upload_assets(
            produced + [update_manifest(manifest, a.out, True, a.release_tag, a.repo)],
            a.release_tag,
            a.repo,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
