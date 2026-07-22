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

from cfb_data_build.config import DatasetSpec
from cfb_model_publish.artifacts import _gh_release_exists, _gh_runner

# Mirror R PUBLISH_REPOS (``R/_data_utils.R:5``).
PUBLISH_REPOS: list[str] = ["sportsdataverse/sportsdataverse-data"]


def _dataset_files(spec: DatasetSpec, season: int, base: str | Path) -> list[Path]:
    """The on-disk release files for one dataset+season (parquet + rds + csv).

    All three released formats ship to the tag — the release is the distribution
    channel (rds/csv are not committed to this repo).
    """
    root = Path(base) / spec.dataset
    candidates = [
        root / "parquet" / f"{spec.stem}_{season}.parquet",
        root / "rds" / f"{spec.stem}_{season}.rds",
        root / "csv" / f"{spec.stem}_{season}.csv",
    ]
    return [f for f in candidates if f.exists()]


def publish_dataset(
    spec: DatasetSpec,
    season: int,
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
        elif not exists(spec.tag, repo):
            run(
                [
                    "release",
                    "create",
                    spec.tag,
                    "--repo",
                    repo,
                    "--title",
                    spec.tag,
                    "--notes",
                    f"{spec.tag} (CFB dataset, Python-built).",
                ]
            )
        count = 0
        for f in files:
            if dry_run:
                print(f"[dry-run] would upload {f} -> {repo}:{spec.tag}")
                continue
            run(["release", "upload", spec.tag, str(f), "--repo", repo, "--clobber"])
            count += 1
        uploaded[repo] = count
    return {"tag": spec.tag, "files": [str(f) for f in files], "uploaded": uploaded}
