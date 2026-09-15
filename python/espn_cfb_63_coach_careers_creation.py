"""Builder: ESPN CFB head-coach careers: every written coach season summed, rates recomputed.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
Reads the ``coach_tendencies`` parquets already under ``cfb/``, so run shim 62
for every season first; the file is season-less and rewritten each run.

**No R twin yet.** The tendencies live in sdv-py's shared football layer; the R
port is a tracked follow-up, and the dataset is declared in ``KNOWN_UNPAIRED``
in ``tests/test_r_python_parity.py`` until it lands.

Example:
    One season::

        uv run python python/espn_cfb_63_coach_careers_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "coach_careers"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
