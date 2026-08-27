"""Builder: ESPN CFB advanced box: defensive.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so the
directory listing is the pipeline and each dataset is runnable on its own.

**No numbered R twin of its own.** R builds all ten advanced-box datasets inside
the single stage ``R/espn_cfb_04_adv_box_creation.R`` (see its ``.ADV_MAP`` and
``.ADV_EXTRA``), so Python's decomposition is declared in ``KNOWN_UNPAIRED`` in
``tests/test_r_python_parity.py``. The 20-29 block is a Python-side split of
stage 04 -- the gap from 04 is deliberate and does NOT mean these build last.

Example:
    One season::

        uv run python python/espn_cfb_24_adv_defensive_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "adv_defensive"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
