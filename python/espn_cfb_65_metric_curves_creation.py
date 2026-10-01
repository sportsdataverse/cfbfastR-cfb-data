"""Builder: rate curves along a continuous axis -- FG% by distance, cmp% / EPA by air yards,
4th-down conversion by yards to go, success by down x distance; league, team and player rows.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
Computed at build time from ONE season of the committed ``cfb/pbp``
(``sportsdataverse.metric_curves``), so the installed sdv-py defines the
bucket edges and the attempt populations.

Example:
    One season::

        uv run python python/espn_cfb_65_metric_curves_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "metric_curves"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
