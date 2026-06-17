from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from . import RAW_BASE
from .schedule import season_game_ids


def final_url(game_id: int) -> str:
    return f"{RAW_BASE}/json/final/{game_id}.json"


def _default_downloader(url: str):
    from sportsdataverse.dl_utils import download  # pooled session + retry/backoff
    return download(url)


def fetch_final(
    seasons: list[int] | None,
    cache_dir: str | Path,
    *,
    schedule: str | Path | None = None,
    refresh: bool = False,
    downloader: Callable[..., object] | None = None,
) -> dict:
    """Fetch each season's final.json by URL into cache_dir/{game_id}.json. Fail-soft per game."""
    dl = downloader or _default_downloader
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ids = season_game_ids(schedule, seasons)
    fetched = skipped = missing = 0
    for gid in ids:
        dest = cache / f"{gid}.json"
        if dest.exists() and not refresh:
            try:
                json.loads(dest.read_text())  # corrupt-cache guard
                skipped += 1
                continue
            except Exception:  # noqa: BLE001 — corrupt cache: re-fetch once
                pass
        try:
            resp = dl(final_url(gid))
            if getattr(resp, "status_code", 200) != 200 or not getattr(resp, "text", ""):
                missing += 1
                continue
            json.loads(resp.text)  # validate
            dest.write_text(resp.text, encoding="utf-8")
            fetched += 1
        except Exception:  # noqa: BLE001 — one bad game cannot abort the batch
            missing += 1
    return {"fetched": fetched, "skipped": skipped, "missing": missing, "total": len(ids)}
