from __future__ import annotations

import os
import subprocess
from pathlib import Path

from sportsdataverse.release import upload_release_sidecars

from cfb_model_build.cfb_model_reports.discovery import discover_models

# Upload timeout. 300s was too short once plain-csv artifacts entered the mix:
# a pbp season csv is ~347 MB (vs ~34 MB parquet) and reliably blew the limit,
# aborting a 22-season publish after one season. Sized for the largest artifact
# we ship on a slow link; override with CFB_GH_TIMEOUT_SECONDS.
GH_TIMEOUT_SECONDS = int(os.getenv("CFB_GH_TIMEOUT_SECONDS", "1800"))

# Release-notes body used when auto-creating a missing release. Keyed by tag;
# falls back to a generic note for any other tag.
_RELEASE_BODY = {
    "espn_cfb_model_artifacts": (
        "All CFB model artifacts (EP/WP/QBR/CPOE/fourth-down .ubj + RB-eval .pkl) + model cards."
    ),
    "espn_cfb_model_pbp": ("College Football model play-by-play (EP/WP/QBR enriched; Python-built)."),
    "cfb_ratings": (
        "College Football opponent-adjusted team ratings, one row per team per "
        "season (SP+-style): offensive/defensive/special-teams EPA, FEI, tempo, "
        "dense ranks, and a net z-score. Built by sdv-py `cfb_ratings()` over the "
        "released `espn_cfb_pbp` play-by-play."
    ),
    "espn_cfb_adv_team_gamelog": (
        "College Football advanced team box score, ONE ROW PER TEAM-GAME, "
        "2004-2025. The adv_team metrics plus the game context they lack: "
        "opponent id/name, home/away, neutral site, points for/against, "
        "margin, win, and kickoff date -- the columns needed for "
        "opponent-adjustment, strength-of-schedule, home/away splits and "
        "rolling in-season trends. Built from the rebuilt espn_cfb_pbp."
    ),
    "cfb_ratings_weekly": (
        "College Football opponent-adjusted team ratings as of the END OF EACH "
        "REGULAR-SEASON WEEK, 2004-2025. LONG FORMAT: one asset per season "
        "carrying a `through_week` column with every week's cumulative "
        "snapshot stacked -- filter `through_week == W` for that week's view. "
        "The ridge is refit on everything up to week W, so this is NOT "
        "derivable by summing per-game rows."
    ),
    "cfb_team_summaries_weekly": (
        "College Football team season summaries as of the END OF EACH "
        "REGULAR-SEASON WEEK, 2004-2025. LONG FORMAT: one asset per season "
        "with a `through_week` column stacking every week's cumulative "
        "state. Built by re-running the season aggregation with plays "
        "filtered to `week <= W`."
    ),
    "cfb_recruiting_proj": (
        "College Football preseason team projections, one row per team per "
        "target season: predicted wins and scoring margin from an as-of ridge "
        "on roster features (247 talent composite, blue-chip ratio, returning "
        "production, prior wins). Built by sdv-py `cfb_recruiting_projection()`."
    ),
}


#: Release sidecar metadata: the loader a consumer reads each tag through.
#: R's sportsdataverse_save() writes this as package_function.txt/.json beside
#: every published asset; this publisher dropped it along with the timestamp
#: pair. Model tags with no loader name this producer instead -- the convention
#: the ncaa_*_rapm tags already carry on their published sidecars.
PKG_FUNCTION: dict[str, str] = {
    "cfb_crosswalk": "sportsdataverse.cfb.load_cfb_teams_crosswalk()",
    "cfb_fpi_weekly": "sportsdataverse.cfb.load_cfb_fpi_weekly()",
    "cfb_ratings": "sportsdataverse.cfb.load_cfb_ratings()",
    "cfb_ratings_weekly": "sportsdataverse.cfb.load_cfb_ratings_weekly()",
    "espn_cfb_model_pbp": "sportsdataverse.cfb.load_cfb_model_pbp()",
}
_PRODUCER = "python/cfb_model_build/cfb_model_publish/artifacts.py"


def plan_uploads(artifacts_dir) -> list:
    files: list = []
    for m in discover_models(artifacts_dir):
        files.append(m.model_path)
        files.append(m.card_path)
    # de-dup, stable order
    seen, out = set(), []
    for p in files:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _gh_runner(args: list) -> None:
    subprocess.run(["gh", *args], check=True, timeout=GH_TIMEOUT_SECONDS)


def _gh_release_exists(tag: str, repo: str) -> bool:
    """True if a GitHub release for ``tag`` already exists on ``repo``.

    Deliberately ``gh api`` (REST) and not ``gh release view``: the release
    commands go through GitHub's GraphQL endpoint, whose quota is a separate --
    and exhaustible -- budget. A publish must not fail its create-if-missing
    guard (and then create a duplicate release) because a GraphQL quota ran out.
    """
    r = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases/tags/{tag}", "--silent"],
        capture_output=True,
        timeout=GH_TIMEOUT_SECONDS,
    )
    return r.returncode == 0


def upload_artifacts(
    artifacts_dir,
    tag: str,
    repo: str,
    *,
    pattern: str | None = None,
    dry_run: bool = False,
    runner=None,
    exists_check=None,
) -> dict:
    """Upload each discovered model + card to the ``tag`` release on ``repo``.

    The release is created if it does not already exist (``gh release upload``
    does not create one), so a single call is self-sufficient. ``runner`` and
    ``exists_check`` are injectable for hermetic testing.

    Args:
        pattern: When given, upload ``artifacts_dir.glob(pattern)`` (sorted)
            instead of running model discovery. Dataset tags publish parquet +
            a card sidecar, which ``discover_models`` -- which looks for model
            files -- would not find. Omit for the model-artifacts path.
    """
    run = runner or _gh_runner
    exists = exists_check or _gh_release_exists
    files = sorted(Path(artifacts_dir).glob(pattern)) if pattern else plan_uploads(artifacts_dir)
    created_release = False
    if dry_run:
        print(f"[dry-run] would ensure release {repo}:{tag} exists")
    elif not exists(tag, repo):
        body = _RELEASE_BODY.get(tag, f"{tag} (auto-created by cfb_model_publish).")
        run(["release", "create", tag, "--repo", repo, "--title", tag, "--notes", body])
        created_release = True
    uploaded = 0
    for f in files:
        if dry_run:
            print(f"[dry-run] would upload {f} -> {repo}:{tag}")
            continue
        run(["release", "upload", tag, str(f), "--repo", repo, "--clobber"])
        uploaded += 1
    # stamp LAST so the timestamp describes a finished upload, and only when
    # something actually uploaded -- a stamp on a no-op run would claim data
    # moved when it did not
    if uploaded:
        upload_release_sidecars(
            tag, runner=run, pkg_function=PKG_FUNCTION.get(tag, _PRODUCER), repo=repo
        )
    return {
        "uploaded": uploaded,
        "files": [str(f) for f in files],
        "tag": tag,
        "created_release": created_release,
    }
