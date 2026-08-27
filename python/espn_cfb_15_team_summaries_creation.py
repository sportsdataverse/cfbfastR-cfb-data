"""Builder: ESPN CFB team summaries.

Thin entrypoint over the ``summaries`` build. Twin of
``R/espn_cfb_15_team_summaries_creation.R``, which is the ONE stage the daily
workflow still runs in R (see the comment in ``.github/workflows/daily_cfb.yml``).

``summaries`` is assembled from other stages rather than reshaped from a single
``final.json`` section, so it is not a REGISTRY row and
``tests/test_r_python_parity.py`` exempts it in ``NON_DATASET_STAGES``.

Example:
    One season::

        uv run python python/espn_cfb_15_team_summaries_creation.py -s 2026 -e 2026

    Cumulative snapshot through a week::

        uv run python python/espn_cfb_15_team_summaries_creation.py -s 2026 -e 2026 --through-week 5
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "summaries"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
