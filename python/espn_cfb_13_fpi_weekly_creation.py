"""Builder: ESPN CFB weekly Football Power Index snapshots (``cfb_fpi_weekly``).

Thin entrypoint. The build lives in ``cfb_data_build.fpi``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.

No R twin: the R chain never produced this dataset, and 13 is the hole left when
injuries moved 13 -> 14. Number 12 is its sibling ``power_index`` (the per-game
matchup FPI from the same API family), so the two sit adjacent by design.

TIME-CRITICAL, unlike every other stage in this repo. The other datasets are
rebuildable from the raw store at any time; this one is not. ESPN OVERWRITES the
week-1 slot with a late-season computation (2024's week 1 is stamped 2024-12-15,
later than that season's week 16), so an as-of-week-N rating exists only in a
capture taken during week N. A backfill can never recover one. That is why this
stage runs on its own in-season schedule
(``.github/workflows/cfb_fpi_weekly.yml``) rather than inside the daily driver:
a missed week is permanent, so the capture must not be coupled to the 25-dataset
job's failures. Weeks already captured are preserved on re-run
(``fpi.merge_preserving_earliest``), and a run that finds no new week is a
no-op.

Example:
    One season::

        uv run python python/espn_cfb_13_fpi_weekly_creation.py -s 2026 -e 2026 --base ../cfb
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "fpi_weekly"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
