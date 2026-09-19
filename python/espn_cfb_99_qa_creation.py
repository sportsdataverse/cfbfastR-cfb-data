"""Builder: ESPN CFB per-game validation QA (report-only).

One row per game -- ``sportsdataverse.validation.validate_game`` over the
enriched plays of each ``final.json`` -- plus a season ``_summary.json``
sidecar carrying the error-free share, the per-rule counts and the pre-publish
drift gate against the previously published ``espn_cfb_pbp`` asset.

Numbered 99 because it is the LAST stage of a season: the drift gate reads the
season pbp parquet the earlier stages wrote. Report-only in this revision --
``cfb_data_build.qa.BLOCKING`` is ``False`` and nothing here fails a build. No
R twin (``tests/test_r_python_parity.py`` exempts it in ``KNOWN_UNPAIRED``).

Example:
    One season::

        uv run python python/espn_cfb_99_qa_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "qa"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
