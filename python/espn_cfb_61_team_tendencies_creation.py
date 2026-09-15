"""Builder: ESPN CFB season tendencies per team: pace, run/pass by down and score state, efficiency, red-zone / scoring-opportunity and fourth-down decisions.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
Computed at build time from the season's plays (``sportsdataverse.football.tendencies``),
so the installed sdv-py defines every metric.

**No R twin yet.** The tendencies live in sdv-py's shared football layer; the R
port is a tracked follow-up, and the dataset is declared in ``KNOWN_UNPAIRED``
in ``tests/test_r_python_parity.py`` until it lands.

Example:
    One season::

        uv run python python/espn_cfb_61_team_tendencies_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "team_tendencies"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
