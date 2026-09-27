"""Builder: rolling event-count windows -- a player's or team's form over its last N events.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
Computed at build time from the committed ``cfb/pbp`` history and the unified
schedule (``sportsdataverse.rolling_windows``), so the installed sdv-py
defines the windows and baselines.

Example:
    One season::

        uv run python python/espn_cfb_64_rolling_windows_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "rolling_windows"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
