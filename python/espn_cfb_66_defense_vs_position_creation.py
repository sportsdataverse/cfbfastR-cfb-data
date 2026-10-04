"""Builder: defense vs position -- EPA/play, success and explosive rate each defense allowed
to QBs, RBs, WRs and TEs, with percentiles among FBS qualifiers.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
Computed at build time from ONE season of the committed ``cfb/pbp``,
``cfb/cfb_rosters`` and ``cfb/cfb_schedules``
(``sportsdataverse.defense_vs_position``), so the installed sdv-py defines the
play populations and the position groups. Seasons before 2014 are not built.

Example:
    One season::

        uv run python python/espn_cfb_66_defense_vs_position_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "defense_vs_position"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
