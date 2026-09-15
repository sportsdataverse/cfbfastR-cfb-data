"""Builder: ESPN CFB season leaderboard: kickers.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
The section is computed at build time from each final's plays
(``sportsdataverse.football.usage_box``), so the installed sdv-py defines it.

**No R twin yet.** The usage box lives in sdv-py's shared football layer; the R
port is a tracked follow-up, and the dataset is declared in ``KNOWN_UNPAIRED``
in ``tests/test_r_python_parity.py`` until it lands.

Example:
    One season::

        uv run python python/espn_cfb_56_usage_st_kickers_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "usage_st_kickers"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
