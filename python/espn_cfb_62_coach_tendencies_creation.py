"""Builder: ESPN CFB season tendencies per head coach (team-season attribution from data/cfb_coach_seasons.csv).

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
The head coach of each team-season comes from the vendored roster
(``cfb_data_build.coaches``; refresh from CFBD with ``python -m cfb_data_build.coaches``).

**No R twin yet.** The tendencies live in sdv-py's shared football layer; the R
port is a tracked follow-up, and the dataset is declared in ``KNOWN_UNPAIRED``
in ``tests/test_r_python_parity.py`` until it lands.

Example:
    One season::

        uv run python python/espn_cfb_62_coach_tendencies_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "coach_tendencies"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
