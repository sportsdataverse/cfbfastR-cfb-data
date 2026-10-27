"""G-CFB-1: the pre-upload gate for REPROCESS run #2 (sdv-py quality program).

Runs on a rebuilt ``cfb/pbp`` season file (and the season's cfb-raw finals for the game
count) and must read 0 defects before that season's datasets publish (ClaudeCowork
``plans/2026-10-08-sdv-py-quality-refactor/plan-v2/REPROCESS.md``, run #2). Each check
counts violating rows or games; a season passes when every count is 0.

    uv run python -m cfb_data_build.gate --base ../cfb \\
        --finals ../../cfbfastR-cfb-raw/cfb/json/final -s 2026

Exit 0 when every season passes, 1 otherwise; one line per season and check either way.
``--publish`` uploads a PASSING season's pbp (the file just checked, no rebuild); a failing
season uploads nothing. ``scripts/daily_cfb_processor.sh`` runs it this way under CFB_GATE=1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

#: Kickoff touchback spots (yards to the end zone): 80 through 2011, 75 from 2012 (#778).
_TOUCHBACK_CUT = 2012
#: A standalone extra-point row is an xp_attempt before the 2014 play-type change (#771).
_XP_TEXT = (
    r"(?i)^(extra point|pat|kick attempt)\b|\bextra point (good|missed|blocked|no good)"
)


def _checks(pbp: pl.DataFrame, season: int) -> dict[str, int]:
    rush = pl.col("rush") == True  # noqa: E712
    yds = pl.col("yds_rushed")
    ko = pl.col("kickoff_play") == True  # noqa: E712
    out: dict[str, int] = {}

    # firstHalfKickoffTeamId is one value per game (the published 2024 read 497/944 inverted)
    out["first_half_kickoff_team_varies"] = (
        pbp.group_by("game_id")
        .agg(pl.col("firstHalfKickoffTeamId").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    # the receive-2H-kickoff flag only means something in the first half
    out["receives_2h_kickoff_after_half"] = pbp.filter(
        (pl.col("period") > 2) & (pl.col("start.pos_team_receives_2H_kickoff") == True)  # noqa: E712
    ).height
    # kickoff touchback spot by era: 80 through 2011, 75 from 2012
    want_tb = 80 if season < _TOUCHBACK_CUT else 75
    out["kickoff_touchback_spot"] = pbp.filter(
        ko & (pl.col("start.yardsToEndzone.touchback") != want_tb).fill_null(True)
    ).height
    # a 2012+ kickoff touchback returns to the 25
    if season >= _TOUCHBACK_CUT:
        out["touchback_return_not_25"] = pbp.filter(
            ko
            & (pl.col("kickoff_tb") == True)
            & (pl.col("yds_kickoff_return") != 25).fill_null(True)  # noqa: E712
        ).height
    # pre-2014 standalone XP rows are xp_attempt
    if season < 2014:
        out["pre2014_xp_not_flagged"] = pbp.filter(
            pl.col("type.text").fill_null("").str.contains(_XP_TEXT)
            & (pl.col("xp_attempt") != True).fill_null(True)  # noqa: E712
        ).height
    # the three line-yards parts sum to the carry on every gain (R843-2)
    parts = (
        pl.col("line_yards") + pl.col("second_level_yards") + pl.col("open_field_yards")
    )
    out["line_yards_parts"] = pbp.filter(
        rush & (yds >= 0) & ((parts - yds).abs() > 1e-6).fill_null(True)
    ).height
    return out


def _adv_team_checks(adv_team: pl.DataFrame | None) -> dict[str, int]:
    """#740: every team row of the season's adv_team (advBoxScore.team) carries yards_per_dropback."""
    if adv_team is None:
        return {"adv_team_missing": 1}
    if adv_team.height == 0:
        return {"adv_team_missing": 1}
    if "yards_per_dropback" not in adv_team.columns:
        return {"yards_per_dropback_missing": adv_team.height}
    return {"yards_per_dropback_missing": adv_team["yards_per_dropback"].null_count()}


def _finals_with_plays(finals: Path, season: int) -> tuple[int, int] | None:
    """``(finals with plays, completed games with no readable final)`` for the season.

    An empty-but-current final has no plays. A completed game whose final is absent or
    unreadable would drop out of the pbp build AND this count alike, so it is counted apart.

    The finals directory is flat (one ``{game_id}.json`` per game, every season), so the
    season's ids come from cfb-raw's ``cfb/cfb_schedule_master.parquet`` beside it; ``None``
    when that file is absent (a ``.cache/cfb_final`` download cache has no master), which
    the caller counts as a failure.
    """
    master = finals.parent.parent / "cfb_schedule_master.parquet"
    if not master.exists():
        return None
    games = pl.read_parquet(master, columns=["season", "id", "status_type_completed"]).filter(
        pl.col("season") == season
    )
    n = lost = 0
    for gid, completed in games.select("id", pl.col("status_type_completed").fill_null(False)).iter_rows():
        try:
            if json.loads((finals / f"{gid}.json").read_text()).get("plays"):
                n += 1
        except (OSError, ValueError):
            lost += bool(completed)
    return n, lost


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="../cfb", help="cfb dataset root: <base>/pbp, <base>/adv_team")
    ap.add_argument(
        "--finals", help="cfb-raw cfb/json/final: also check the season's game count"
    )
    ap.add_argument("-s", "--start", type=int, required=True)
    ap.add_argument("-e", "--end", type=int)
    ap.add_argument("--publish", action="store_true", help="upload each PASSING season's pbp")
    args = ap.parse_args(argv)
    ok = True
    for season in range(args.start, (args.end or args.start) + 1):
        f = Path(args.base) / "pbp" / "parquet" / f"play_by_play_{season}.parquet"
        if not f.exists():
            print(f"{season}: MISSING {f}")
            ok = False
            continue
        pbp = pl.read_parquet(f)
        counts = _checks(pbp, season)
        at = Path(args.base) / "adv_team" / "parquet" / f"adv_team_{season}.parquet"
        counts |= _adv_team_checks(pl.read_parquet(at) if at.exists() else None)
        if args.finals:
            have = pbp["game_id"].n_unique()
            finals = _finals_with_plays(Path(args.finals), season)
            if finals is None:  # no master to count against: fail, never publish unchecked
                counts["finals_master_missing"] = 1
            else:
                counts["games_short_of_finals"] = max(finals[0] - have, 0)
                counts["completed_games_without_final"] = finals[1]
        bad = {k: v for k, v in counts.items() if v}
        ok &= not bad
        print(
            f"{season}: {'PASS' if not bad else 'FAIL'} "
            + " ".join(f"{k}={v}" for k, v in counts.items())
        )
        if args.publish and not bad:
            from cfb_data_build.config import REGISTRY
            from cfb_data_build.publish import publish_dataset

            publish_dataset(REGISTRY["pbp"], season, base=args.base)
    print("G-CFB-1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
