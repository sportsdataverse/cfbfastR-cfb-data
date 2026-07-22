"""Dataset IO -- polars port of ``write_dataset`` / ``.append_manifest``.

Writes the released formats under ``cfb/{dataset}/`` and maintains the per-season
manifest. Port of ``R/_data_utils.R:83-109``.

Python writes all three released formats — **parquet + rds + csv** — plus the
manifest, matching the R producer. ``.rds`` is written natively via
:func:`sportsdataverse._rds.write_rds` (no R round-trip). The parity bar is the
**parquet** (the canonical cross-engine format). List cells are already
JSON-encoded to strings by :mod:`cfb_data_build.reshape`, so all three
serializers share one schema.

Only the **parquet** and the manifest are committed to this repo; ``rds/`` and
``csv/`` are build artifacts that ship to the dataset releases and are
git-ignored (they bloat the tree and the release is the distribution channel).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from sportsdataverse._rds import write_rds

# R writes these via plain ``saveRDS(df)`` on a data.table (R/_data_utils.R:92) —
# no custom S3 stamp — so mirror that class vector rather than a ``*_data`` one.
RDS_CLASS: tuple[str, ...] = ("data.table", "data.frame")


def _utc_now_str() -> str:
    """UTC timestamp, mirroring R ``format(Sys.time(), tz='UTC', usetz=TRUE)``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _append_manifest(
    dataset: str, season: int, row_count: int, base: str | Path
) -> Path:
    """Upsert one ``(season, row_count, generated_at_utc)`` row, sorted by season.

    Port of ``.append_manifest`` -- replaces any existing row for ``season`` so
    a re-run is idempotent.
    """
    f = Path(base) / dataset / f"cfb_{dataset}_in_data_repo.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    row = pl.DataFrame(
        {
            "season": [int(season)],
            "row_count": [int(row_count)],
            "generated_at_utc": [_utc_now_str()],
        }
    )
    if f.exists():
        old = pl.read_csv(f).filter(pl.col("season") != int(season))
        row = pl.concat([old, row], how="diagonal_relaxed")
    row.sort("season").write_csv(f)
    return f


def write_dataset(
    df: pl.DataFrame,
    dataset: str,
    season: int,
    stem: str,
    *,
    base: str | Path = "cfb",
) -> dict[str, Path] | None:
    """Write parquet + rds + csv under ``base/{dataset}/`` and append the manifest.

    Returns the written paths, or ``None`` for an empty frame (matches R's
    "0 rows, skipping write"). ``rds``/``csv`` are release artifacts (git-ignored);
    only the parquet and manifest are committed.
    """
    if df is None or df.height == 0:
        return None
    root = Path(base) / dataset
    parquet_path = root / "parquet" / f"{stem}_{season}.parquet"
    rds_path = root / "rds" / f"{stem}_{season}.rds"
    csv_path = root / "csv" / f"{stem}_{season}.csv"
    for sub in ("parquet", "rds", "csv"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    df.write_parquet(parquet_path)
    write_rds(df, rds_path, cls=RDS_CLASS)
    df.write_csv(csv_path)
    manifest_path = _append_manifest(dataset, season, df.height, base)
    return {
        "parquet": parquet_path,
        "rds": rds_path,
        "csv": csv_path,
        "manifest": manifest_path,
    }
