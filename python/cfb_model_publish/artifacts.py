from __future__ import annotations

import subprocess

from cfb_model_reports.discovery import discover_models

GH_TIMEOUT_SECONDS = 300

# Release-notes body used when auto-creating a missing release. Keyed by tag;
# falls back to a generic note for any other tag.
_RELEASE_BODY = {
    "espn_cfb_model_artifacts": (
        "All CFB model artifacts (EP/WP/QBR/CPOE/fourth-down .ubj + RB-eval .pkl) "
        "+ model cards."
    ),
    "espn_cfb_model_pbp": (
        "College Football model play-by-play (EP/WP/QBR enriched; Python-built)."
    ),
}


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
    """True if a GitHub release for ``tag`` already exists on ``repo``."""
    r = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        capture_output=True,
        timeout=GH_TIMEOUT_SECONDS,
    )
    return r.returncode == 0


def upload_artifacts(
    artifacts_dir,
    tag: str,
    repo: str,
    *,
    dry_run: bool = False,
    runner=None,
    exists_check=None,
) -> dict:
    """Upload each discovered model + card to the ``tag`` release on ``repo``.

    The release is created if it does not already exist (``gh release upload``
    does not create one), so a single call is self-sufficient. ``runner`` and
    ``exists_check`` are injectable for hermetic testing.
    """
    run = runner or _gh_runner
    exists = exists_check or _gh_release_exists
    files = plan_uploads(artifacts_dir)
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
    return {
        "uploaded": uploaded,
        "files": [str(f) for f in files],
        "tag": tag,
        "created_release": created_release,
    }
