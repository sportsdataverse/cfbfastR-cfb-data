"""Thin shim -- the weekly builders now live in cfb_data_build.derived (P6).

Kept so existing invocations keep working. Prefer the unified CLI:
    python -m cfb_data_build --dataset ratings_weekly -s 2004 -e 2025 --publish
    python -m cfb_data_build --dataset team_summaries_weekly -s 2004 -e 2025 --publish

`--dataset ratings|team_summaries` here maps to the *_weekly dataset names.
"""

from __future__ import annotations

import sys

from cfb_data_build.cli import main

_ALIAS = {"ratings": "ratings_weekly", "team_summaries": "team_summaries_weekly"}

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--dataset" in argv:
        i = argv.index("--dataset")
        argv[i + 1] = _ALIAS.get(argv[i + 1], argv[i + 1])
    sys.exit(main(argv))
