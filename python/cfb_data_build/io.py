"""Dataset IO -- polars port of ``write_dataset`` / ``.append_manifest``.

Writes the released formats under ``cfb/{dataset}/`` and maintains the per-season
manifest. Port of ``R/_data_utils.R:83-109``.

Scope note: R writes parquet **+ rds + csv.gz**; ``.rds`` is R's native
serialization and is left to the R producer. The dual-write parity bar is the
**parquet** (the canonical cross-engine format); Python writes parquet + csv.gz
+ manifest. List cells are already JSON-encoded to strings by
:mod:`cfb_data_build.reshape`, so all three serializers share one schema.
"""

from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


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
    """Write parquet + csv.gz under ``base/{dataset}/`` and append the manifest.

    Returns the written paths, or ``None`` for an empty frame (matches R's
    "0 rows, skipping write").
    """
    if df is None or df.height == 0:
        return None
    root = Path(base) / dataset
    parquet_path = root / "parquet" / f"{stem}_{season}.parquet"
    csv_path = root / "csv" / f"{stem}_{season}.csv.gz"
    for sub in ("parquet", "csv"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    df.write_parquet(parquet_path)
    with gzip.open(csv_path, "wb") as fh:
        df.write_csv(fh)
    manifest_path = _append_manifest(dataset, season, df.height, base)
    return {"parquet": parquet_path, "csv": csv_path, "manifest": manifest_path}
