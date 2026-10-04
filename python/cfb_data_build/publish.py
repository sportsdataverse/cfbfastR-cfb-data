"""Release publishing -- generalizes the cfb_model_publish gh-release pattern.

R's ``publish_dataset`` (``R/_data_utils.R:170-181``) uploads each format to the
``sportsdataverse-data`` release under the dataset's tag, creating the release
if absent. ``cfb_model_publish.artifacts.upload_artifacts`` is model-discovery
specific, so we reuse its low-level gh helpers (``_gh_release_exists`` for the
create-if-missing guard, ``_gh_runner`` for the ``gh`` invocation) with an
explicit dataset file list. ``runner`` / ``exists_check`` are injectable for
hermetic tests (same convention as ``upload_artifacts``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from cfb_model_build.cfb_model_publish.artifacts import (
    _gh_release_exists,
    _gh_runner,
    ensure_release,
)
from sportsdataverse.paper_index import HOLDOUT_SEASONS, TRAIN_SEASONS
from sportsdataverse.release import upload_release_sidecars

from cfb_data_build.config import PKG_FUNCTION, DatasetSpec
from cfb_data_build.io import dataset_stem

# Mirror R PUBLISH_REPOS (``R/_data_utils.R:5``).
PUBLISH_REPOS: list[str] = ["sportsdataverse/sportsdataverse-data"]

# the Paper Index fit spans, so the release note names the years sdv-py does
_PI_TRAIN, _PI_HOLDOUT = TRAIN_SEASONS["cfb"], HOLDOUT_SEASONS["cfb"]


def _dataset_files(
    spec: DatasetSpec, season: int | None, base: str | Path
) -> list[Path]:
    """The on-disk release files for one dataset+season (parquet + rds + csv).

    All three released formats ship to the tag — the release is the distribution
    channel (rds/csv are not committed to this repo). The csv is listed in both
    plain and gzipped form and filtered on existence, so a builder that gzips its
    csv (``cfb_rosters``) ships ``.csv.gz`` — matching what the sibling ESPN tags
    already publish — while every other dataset keeps shipping plain ``.csv``.
    """
    root = Path(base) / spec.dataset
    name = dataset_stem(spec.stem, season)
    candidates = [
        root / "parquet" / f"{name}.parquet",
        root / "rds" / f"{name}.rds",
        root / "csv" / f"{name}.csv",
        root / "csv" / f"{name}.csv.gz",
        # the QA season summary (error-free share + drift findings) is part of
        # the espn_cfb_qa asset, not a build artefact
        root / "parquet" / f"{name}_summary.json",
    ]
    return [f for f in candidates if f.exists()]


#: Per-tag release notes. The weekly datasets carry a LEAKAGE WARNING because
#: `through_week == W` is INCLUSIVE of week W -- verified empirically at 97.0%
#: against 58.7% for the exclusive reading. A consumer who filters
#: `through_week == W` to project week W is handed that week's results,
#: including the game being projected. The obvious usage is the wrong one, so
#: the note has to say so.
_WEEKLY_ASOF_WARNING = """
**As-of semantics -- read before using this for projections.**

`through_week == W` is **inclusive of week W**: the snapshot contains games
PLAYED in week W. To project week W, use the `through_week == W - 1` row.
Filtering `through_week == W` and predicting week W leaks that week's results.

Verified empirically (2024, delta between consecutive snapshots vs games
actually played): 97.0% consistent with the inclusive reading, 58.7% with the
exclusive one.
"""

RELEASE_NOTES: dict[str, str] = {
    "cfb_team_summaries_weekly": (
        "College Football team summaries as of the END OF EACH REGULAR-SEASON "
        "WEEK. LONG FORMAT: one asset per season carrying a `through_week` "
        "column with every week's cumulative snapshot stacked. The current "
        "season carries only the weeks played so far.\n" + _WEEKLY_ASOF_WARNING
    ),
    "cfb_ratings_weekly": (
        "College Football opponent-adjusted team ratings as of the END OF EACH "
        "REGULAR-SEASON WEEK. LONG FORMAT: one asset per season carrying a "
        "`through_week` column with every week's cumulative snapshot stacked. "
        "The ridge is refit on everything up to week W, so this is NOT "
        "derivable by summing per-game rows. The current season carries only "
        "the weeks played so far.\n" + _WEEKLY_ASOF_WARNING
    ),
    "cfb_matchup_features": (
        "College Football per-team, per-GAME matchup features (EPA / success / "
        "weighted EPA / scoring-opportunity rate over expected / rush rate over "
        "expected / pace), computed from the team's plays dated STRICTLY BEFORE "
        "the game. Keyed (season, game_id, team_id, team); a team's first game "
        "of the season is null. As-of by construction: the row for game G never "
        "contains G, so no `through_week` arithmetic is needed."
    ),
    "cfb_matchup_line": (
        "College Football per-game matchup line (FBS vs FBS, bowls dropped, CFP "
        "kept): home / away as-of features, the prior season's full-season "
        "`prev_*` block, pace, CFBD pregame ELO with opponent-ELO rolls, "
        "consensus betting lines and game meta -- 268 columns. Regular-season "
        "week-1 (and any later opener's) features are the prior season's "
        "full-season values. Weather (CFBD /games/weather), team / venue meta "
        "(/teams), talent (/talent) and the head coach with tenure (/coaches) "
        "are filled from CFBD; roster talent, returning production, coach "
        "continuity and the QB block come from the versioned reference tables "
        "under data/ (the QB identity is the realized usage leader, post-hoc "
        "for the season it describes)."
    ),
    "cfb_rolling_windows": (
        "College Football rolling event-count windows: per player (dropbacks, "
        "targets, carries) and team (plays), EPA/play and success rate over the "
        "last N events vs the previous N, the season start and the career "
        "baseline (2004+ history), with delta ranks and sample sizes. One asset "
        "per season, as of the season's last game."
    ),
    "cfb_metric_curves": (
        "College Football rate curves along a continuous axis, per SEASON, for "
        "the league, every team and every credited player (`entity_type`): "
        "`fg_pct_by_distance` (FG% by kick distance, 5-yard buckets 15-65 then "
        "65-80), `fourth_conv_by_ytg` (4th-down conversion by yards to go), "
        "`success_by_down_distance` (EPA success by down x distance; `down` is "
        "the second axis, 4 x 5 league rows), `cmp_pct_by_air_yards` and "
        "`epa_by_air_yards` (completion% / EPA success by air-yards bucket). "
        "Each populated bucket (`x_lo` inclusive, `x_hi` exclusive) carries "
        "attempts, successes, rate = successes / attempts and the mean EPA per "
        "attempt; an empty bucket has no row. Regular season + postseason; ids "
        "are ESPN (`id_source`). SPANS: FG, 4th-down and down x distance curves "
        "2004+ (`yds_fg` is present on 98.6-100% of FG attempts in every season "
        "2004-2013 -- 2004 100.0, 2005 98.6, 2006 100.0, 2007 99.8, 2008 99.7, "
        "2009 100.0, 2010 100.0, 2011 99.8, 2012 99.8, 2013 99.8 -- and 99.1%+ "
        "since 2014). AIR-YARDS curves 2025+ ONLY: `air_yards` is on 41% of "
        "2025 pass attempts and 96% of 2026's; every earlier season carries at "
        "most 32 stray air-yards plays, so the producer drops the two air-yards "
        "metrics before 2025 rather than publish a one-attempt curve."
    ),
    "cfb_defense_vs_position": (
        "College Football defense vs position: ONE ROW PER DEFENSE PER POSITION "
        "GROUP per season (`season`, `team_id`, `position_group` in QB / RB / WR "
        "/ TE) with the EPA per play, success rate and explosive rate that "
        "defense allowed to the group, plus QB dropbacks and sack rate, RB "
        "carries and yards per carry, WR / TE targets and yards per target "
        "(`sportsdataverse.defense_vs_position`). Every dropback is the QB's; a "
        "carry goes to the rusher's roster group and a target to the "
        "receiver's, so a completion counts for QB and for its receiver's "
        "group. Regular season + postseason, every game in the pbp (FBS and "
        "FCS defenses; `division` says which). `qualified` = 3 or more games. "
        "PERCENTILES: each `<metric>_pct` is among FBS qualifiers within the "
        "season's position group, 0-100, and HIGHER IS ALWAYS THE BETTER "
        "DEFENSE -- allowing less ranks higher, except `sack_rate_allowed` "
        "(sacks per dropback the defense got), where more ranks higher. "
        "Non-qualifiers and non-FBS defenses carry null percentiles. WR / TE "
        "CAVEAT: in 2024 ESPN names a receiver on only about 69% of throws "
        "(few incompletions, no interceptions), so the WR and TE rates are on "
        "targets naming a receiver, mostly completions, and read high; "
        "`unattributed_target_share` on the WR and TE rows is the share of "
        "throws against that defense with no receiver named (FBS median 0.12 "
        "in 2014, 0.33 in 2024). Nothing is imputed. SPAN: 2014+ only -- "
        "earlier rosters carry no positions."
    ),
    "cfb_paper_index_games": (
        "College Football Paper Index per game: ONE ROW PER TEAM PER SCORED "
        "GAME (`game_id`, `team_id`, both Int64) with `paper_share`, the "
        "team's deserved-win probability from eight performance margins "
        "(success rate, explosive rate, explosiveness, scoring-opportunity "
        "conversion, points per opportunity, starting field position, havoc, "
        "turnovers), `opp_share` (the two sum to 1), `won`, and the eight "
        "`<margin>_margin` columns, team minus opponent "
        "(`sportsdataverse.paper_index.paper_index_games`, Game on Paper's "
        "Paper Index). A game is scored when it is completed, has a winner "
        "and both sides ran at least 20 scrimmage snaps; regular season and "
        "postseason, FBS and FCS. Summed over a season the shares are the "
        "`deserved_wins` on espn_cfb_team_summaries / "
        "cfb_team_summaries_weekly, and wins minus that sum is `luck_wins`. "
        "IN-SAMPLE LABEL: `paper_index_span` is `train` for "
        f"{_PI_TRAIN[0]}-{_PI_TRAIN[1]} (the "
        "season's shares were part of the weight fit, so they are in-sample), "
        f"`holdout` for {_PI_HOLDOUT[0]}-{_PI_HOLDOUT[1]} (scored out of sample at fit time) and "
        "`out_of_span` for every other season (never seen by the fit, never "
        "evaluated). NOT A FEATURE: the shares are fitted on who won, so "
        "they describe results and must not enter a predictive model."
    ),
    "cfb_team_opponent_splits": (
        "College Football by-opponent team-game splits: ONE ROW PER TEAM PER "
        "GAME with season, season_type, week, game_id, team_id, opponent_id, "
        "opponent, is_home, points_for, points_against, plays, epa_per_play and "
        "success_rate. A projection of espn_cfb_adv_team_gamelog plus "
        "espn_cfb_adv_situational's EPA success rate; every game is kept, FCS "
        "opponents and bowls included. `epa_per_play` has 0.01 resolution (the "
        "upstream gamelog rounds it). `season_type` includes 4 (all-star games) "
        "and 5 (the spring 2020-21 games). `is_home` is the listed home side, "
        "even at neutral sites."
    ),
    "cfb_poll_analytics": (
        "College Football weekly poll history: ONE ROW PER TEAM PER POLL PER "
        "WEEK for the AP (`ap`), AFCA Coaches (`coaches`) and CFP committee "
        "(`cfp`) polls, 2004+, from ESPN's core-v2 rankings. ESPN's week W poll "
        "is the one released ENTERING week W (week 1 = preseason); the "
        "postseason's week 1 is the final poll and sequences after the regular "
        "season's last published week. `prev_rank` is the rank in that poll's "
        "previous published week; `move = prev_rank - rank` (up is positive); "
        "`entered` is ranked now and not last week (never in a poll's first "
        "week); an `exited` row (rank null) is emitted for a team ranked last "
        "week and not this week; `weeks_ranked` is cumulative and carried on "
        "exit rows. Only the 25 ranked teams -- 'others receiving votes' are "
        "not captured. Every run rebuilds the season from ESPN: an idempotent "
        "refetch (ESPN keeps poll history), not an append-only capture like "
        "cfb_fpi_weekly."
    ),
    "cfb_poll_week_summary": (
        "College Football per poll-week movement summary over "
        "cfb_poll_analytics: `entries` / `exits` counts, `chaos` = sum of "
        "|prev_rank - rank| and `volatility` = population sd of (prev_rank - "
        "rank), both over the union of teams ranked in either week with an "
        "unranked side counted as 26, and both null in a poll's first week."
    ),
}


def release_notes(tag: str) -> str:
    """Release body for a tag, falling back to the generic one-liner."""
    return RELEASE_NOTES.get(tag, f"{tag} (CFB dataset, Python-built).")


def _stamp(tag: str, run: Callable[[list[str]], object], repo: str) -> None:
    """Re-stamp a tag's timestamp / package_function sidecars after an upload.

    R's sportsdataverse_save() attaches these to every published tag; the Python
    publisher dropped them, which left the tag carrying a timestamp.json frozen
    at the last R run while the data kept moving. Runs LAST so the stamp reflects
    the finished upload, and only when something actually uploaded -- a stamp on
    a no-op run would claim data moved when it did not. Goes through the same
    injected ``run`` as the data assets so tests stay offline.
    """
    upload_release_sidecars(
        tag, runner=run, pkg_function=PKG_FUNCTION.get(tag), repo=repo
    )


def publish_dataset(
    spec: DatasetSpec,
    season: int | None,
    *,
    base: str | Path = "cfb",
    repos: list[str] | None = None,
    dry_run: bool = False,
    runner: Callable[[list[str]], None] | None = None,
    exists_check: Callable[[str, str], bool] | None = None,
) -> dict[str, object]:
    """Upload a dataset+season's files to each release tag (create-if-missing, clobber)."""
    run = runner or _gh_runner
    exists = exists_check or _gh_release_exists
    target_repos = repos if repos is not None else PUBLISH_REPOS
    files = _dataset_files(spec, season, base)
    uploaded: dict[str, int] = {}
    for repo in target_repos:
        if dry_run:
            print(f"[dry-run] would ensure release {repo}:{spec.tag} exists")
        else:
            ensure_release(spec.tag, repo, release_notes(spec.tag), run=run, exists=exists)
        count = 0
        for f in files:
            if dry_run:
                print(f"[dry-run] would upload {f} -> {repo}:{spec.tag}")
                continue
            run(["release", "upload", spec.tag, str(f), "--repo", repo, "--clobber"])
            count += 1
        if count:
            _stamp(spec.tag, run, repo)
        uploaded[repo] = count
    return {"tag": spec.tag, "files": [str(f) for f in files], "uploaded": uploaded}
