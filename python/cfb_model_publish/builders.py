"""Build SDV-native CFB dataset parquet for the model-dataset release tags.

Thin orchestration over the ``sportsdataverse.cfb`` compute surface, mirroring
``nflverse-dev/nfl-data/python/nfl_model_publish/builders.py``:

* :func:`build_ratings` -> one ``cfb_ratings_{season}.parquet`` per season,
  plus a single ``cfb_ratings_card.json`` provenance sidecar.

The heavy lifting (loading the released play-by-play, the opponent-adjusted
ridge, FEI, special teams) lives in ``sportsdataverse.cfb.cfb_ratings``; this
module only sequences seasons, materializes frames to disk, and reports row
counts so the CLI can print a one-line summary.
"""

from __future__ import annotations

import json
from pathlib import Path

# One row per team per season, so a season that produces nothing is a real
# failure (an empty/absent pbp asset), not a quiet zero-row parquet. sdv-py's
# `cfb_ratings` returns a correctly-typed empty frame rather than raising when
# the season has no published asset -- catch that here instead of publishing it.
MIN_SEASON = 2004


def build_ratings(seasons: list[int], out_dir, *, compute=None) -> list[dict]:
    """Build per-season CFB ratings and write ``cfb_ratings_{season}.parquet``.

    Args:
        seasons: Seasons to build (one parquet per season).
        out_dir: Output directory (created if absent).
        compute: Injectable ``cfb_ratings``-shaped callable, for hermetic tests.
            Defaults to ``sportsdataverse.cfb.cfb_ratings``.

    Returns:
        List of ``{"season": int, "rows": int, "path": str}`` dicts, one per
        season, in input order.

    Raises:
        ValueError: If a season is below :data:`MIN_SEASON`, or if a season
            yields zero rows (an empty pbp asset -- publishing it would ship a
            silently-empty tag).
    """
    if compute is None:
        from sportsdataverse.cfb import cfb_ratings as compute

    too_old = [s for s in seasons if s < MIN_SEASON]
    if too_old:
        raise ValueError(
            f"cfb_ratings: seasons {too_old} predate the {MIN_SEASON} pbp floor"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for season in seasons:
        df = compute(season)
        if df.height == 0:
            raise ValueError(
                f"cfb_ratings: season {season} produced 0 rows -- refusing to publish an empty tag"
            )
        path = out_dir / f"cfb_ratings_{season}.parquet"
        df.write_parquet(path)
        results.append({"season": season, "rows": df.height, "path": str(path)})
        print(f"ratings: season={season} rows={df.height} -> {path}")
    return results


def write_ratings_card(results: list[dict], out_dir) -> Path:
    """Write the ``cfb_ratings`` model card next to the season parquet.

    Carries the parity anchors the ratings were gated on, so a consumer can
    tell which oracle values this tag's numbers were validated against.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "tag": "cfb_ratings",
        "grain": "one row per team per season",
        "source": "sdv-py sportsdataverse.cfb.cfb_ratings() over the released espn_cfb_pbp asset",
        "seasons": [r["season"] for r in results],
        "rows_by_season": {str(r["season"]): r["rows"] for r in results},
        "parity_anchors_2023": {
            "note": "Spearman vs published oracles, measured on the released pbp (not the fixture)",
            "adj_net_vs_espn_fpi": 0.9259,
            "adj_net_vs_sp_plus_overall": 0.9355,
            "adj_off_epa_vs_sp_plus_off": 0.8464,
            "adj_def_epa_vs_sp_plus_def": 0.7929,
            "fei_net_vs_fei": 0.9644,
        },
        "notes": [
            "adj_net is offense-minus-defense only; special teams is a separate column.",
            "def_rank is a dense rank ascending -- fewer EPA allowed ranks better.",
            "Team set is every team with competitive scrimmage plays, FCS included;"
            " only ~133 of them join the FBS-only published oracles.",
        ],
    }
    path = out_dir / "cfb_ratings_card.json"
    path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"card: {path}")
    return path
