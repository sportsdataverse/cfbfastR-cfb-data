"""Thin shim -- the gamelog now lives in cfb_data_build.derived (P6).

Kept so existing invocations keep working. Prefer the unified CLI:
    python -m cfb_data_build --dataset gamelog -s 2004 -e 2025 --publish
"""

from __future__ import annotations

import sys

from cfb_data_build.cli import main

if __name__ == "__main__":
    sys.exit(main(["--dataset", "gamelog", *sys.argv[1:]]))
