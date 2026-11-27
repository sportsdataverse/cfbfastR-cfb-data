"""Builder: the adjusted-EPA ledger -- team_summaries' Adj EPA/Play laid out game by game.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
Computed at build time (``sportsdataverse.cfb.cfb_adjusted_epa_ledger``) from the same
released pbp + schedule frame ``team_summaries`` (stage 15) fits, so each team's
``contribution`` rows sum to that season's published ``net_adj_epa``; the build fails a
season that misses by more than 1e-9.

Example:
    One season, built but not published::

        uv run python python/espn_cfb_68_adjustment_ledger_creation.py -s 2025 -e 2025
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "adjustment_ledger"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
