"""Builder: Paper Index per game -- each team's deserved-win share in every scored game.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so
the directory listing is the pipeline and each dataset is runnable on its own.
Computed at build time from ONE season of the committed ``cfb/pbp``
(``sportsdataverse.paper_index``), so the installed sdv-py defines the weights,
the game filters and the fit spans behind ``paper_index_span``. The season sums
are the ``deserved_wins`` / ``luck_*`` columns of ``team_summaries`` (stage 15).

Example:
    One season::

        uv run python python/espn_cfb_67_paper_index_games_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "paper_index_games"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
