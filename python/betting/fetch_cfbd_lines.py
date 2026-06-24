"""Fetch CFBD betting lines, chunked by (season, week, seasonType).

CFBD ``/lines`` (https://api.collegefootballdata.com/lines) returns one object per
game with a ``lines: [{provider, spread, spreadOpen, overUnder, overUnderOpen,
homeMoneyline, awayMoneyline}]`` array. The game ``id`` IS the ESPN game_id, and
``homeTeam``/``awayTeam`` carry full names, so downstream keying to the SDV ESPN
schedule is native (no fuzzy name match needed for the modern era).

Pulls are chunked by (year, week, seasonType) -- small payloads, resumable, and
gentle on the CFBD rate limiter -- and cached one JSON per chunk so a re-run only
fetches missing chunks. Set ``CFB_DATA_API_KEY`` (or rely on the cfb-raw ``.env``).

Usage::

    uv run python -m betting.fetch_cfbd_lines --start 2020 --end 2025
    uv run python -m betting.fetch_cfbd_lines --start 2020 --end 2025 --refresh
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

_BASE_URL = "https://api.collegefootballdata.com"
_CACHE = Path(__file__).resolve().parent / ".cache" / "cfbd_lines"
# cfb-raw ships the shared CFBD key in its .env; fall back to it when the env var
# isn't already exported.
_ENV_FALLBACK = Path(__file__).resolve().parents[3] / "cfbfastR-cfb-raw" / ".env"

# Regular season weeks run 1-15 (occasionally 16 for a late championship slate);
# empty weeks return [] and are cached as such. Postseason is a single bucket.
_REGULAR_WEEKS = range(1, 17)
_SEASON_TYPES = ("regular", "postseason")


def _api_key() -> str:
    key = os.environ.get("CFB_DATA_API_KEY") or os.environ.get("CFBD_DATA_API_KEY")
    if not key and _ENV_FALLBACK.exists():
        for line in _ENV_FALLBACK.read_text().splitlines():
            if line.startswith(("CFB_DATA_API_KEY", "CFBD_DATA_API_KEY")) and "=" in line:
                key = line.split("=", 1)[1].strip()
                break
    if not key:
        raise SystemExit(
            "CFB_DATA_API_KEY not set and no cfb-raw/.env fallback found -- "
            "export a CFBD bearer token before fetching."
        )
    return key


def _get(session: requests.Session, year: int, week: int, season_type: str,
         *, max_retries: int = 6) -> list[dict]:
    """GET one (year, week, seasonType) chunk with exponential backoff on 429/5xx."""
    params = {"year": year, "week": week, "seasonType": season_type}
    delay = 1.0
    last = None
    for _ in range(max_retries):
        resp = session.get(f"{_BASE_URL}/lines", params=params, timeout=45)
        if resp.status_code == 200:
            return resp.json()
        last = f"HTTP {resp.status_code}: {resp.text[:150]}"
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        break
    raise RuntimeError(f"CFBD /lines {params} failed: {last}")


def fetch_range(start: int, end: int, *, refresh: bool = False) -> dict:
    """Fetch every (year, week, seasonType) chunk in [start, end] into the cache.

    Returns a small summary dict (chunks fetched / skipped / games seen)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {_api_key()}"})
    fetched = skipped = games = 0
    for year in range(start, end + 1):
        for season_type in _SEASON_TYPES:
            weeks = _REGULAR_WEEKS if season_type == "regular" else range(1, 2)
            for week in weeks:
                path = _CACHE / f"{year}_{season_type}_w{week:02d}.json"
                if path.exists() and not refresh:
                    skipped += 1
                    games += len(json.loads(path.read_text() or "[]"))
                    continue
                data = _get(session, year, week, season_type)
                path.write_text(json.dumps(data))
                fetched += 1
                games += len(data)
                # be polite even when not rate-limited
                time.sleep(0.15)
            print(f"  {year} {season_type}: cached through week {week}", flush=True)
    summary = {"fetched": fetched, "skipped": skipped, "games": games,
               "cache_dir": str(_CACHE)}
    print(f"fetch_range({start}-{end}): {summary}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="betting.fetch_cfbd_lines")
    ap.add_argument("--start", type=int, default=2020)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--refresh", action="store_true", help="re-fetch even cached chunks")
    args = ap.parse_args()
    fetch_range(args.start, args.end, refresh=args.refresh)


if __name__ == "__main__":
    main()
