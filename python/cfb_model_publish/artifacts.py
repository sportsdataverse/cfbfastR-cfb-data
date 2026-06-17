from __future__ import annotations

import subprocess
from pathlib import Path

from cfb_model_reports.discovery import discover_models


def plan_uploads(artifacts_dir) -> list:
    files: list = []
    for m in discover_models(artifacts_dir):
        files.append(m.model_path)
        files.append(m.card_path)
    # de-dup, stable order
    seen, out = set(), []
    for p in files:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def _gh_runner(args: list) -> None:
    subprocess.run(["gh", *args], check=True)


def upload_artifacts(artifacts_dir, tag: str, repo: str, *, dry_run: bool = False, runner=None) -> dict:
    run = runner or _gh_runner
    files = plan_uploads(artifacts_dir)
    uploaded = 0
    for f in files:
        if dry_run:
            print(f"[dry-run] would upload {f} -> {repo}:{tag}")
            continue
        run(["release", "upload", tag, str(f), "--repo", repo, "--clobber"])
        uploaded += 1
    return {"uploaded": uploaded, "files": [str(f) for f in files], "tag": tag}
