"""Builder: ESPN CFB injuries.

Thin entrypoint. The build lives in ``cfb_data_build``; this file exists so the
directory listing is the pipeline and each dataset is runnable on its own.

Twin of ``R/espn_cfb_14_injuries_creation.R`` -- the number is the cross-language
dataset identity, so the two chains stay comparable by eye and by
``tests/test_r_python_parity.py``.

Example:
    One season::

        uv run python python/espn_cfb_14_injuries_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "injuries"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
