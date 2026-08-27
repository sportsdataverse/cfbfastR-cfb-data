"""Builder: ESPN CFB advanced box score -- all ten sections.

Orchestrator, not a dataset. ``R/espn_cfb_04_adv_box_creation.R`` writes all ten
advanced-box outputs from one pass over ``advBoxScore`` (its ``.ADV_MAP`` covers
eight sections, ``.ADV_EXTRA`` the two sdv-py expansion sections). Python builds
them as ten separate REGISTRY datasets, numbered 20-29.

This file is the twin of that R stage: same number, same key, and running it
produces the same ten outputs. It carries no REGISTRY row of its own, so
``tests/test_r_python_parity.py`` exempts it by name in ``NON_DATASET_STAGES``.

Example:
    One season, all ten::

        uv run python python/espn_cfb_04_adv_box_creation.py -s 2026 -e 2026
"""

from __future__ import annotations

import sys

from cfb_data_build.cli import main

#: The ten datasets R's stage 04 bundles. Order matches .ADV_MAP then .ADV_EXTRA
#: in the R twin, so a reader diffing the two files sees the same sequence.
ORDER = [
    "adv_team",
    "adv_passing",
    "adv_rushing",
    "adv_receiving",
    "adv_defensive",
    "adv_turnover",
    "adv_drives",
    "adv_situational",
    "adv_defensive_players",
    "adv_specialists",
]


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    failed: list[str] = []
    for dataset in ORDER:
        print(f"::group::{dataset}", flush=True)
        try:
            rc = main(["--dataset", dataset, *argv])
            if rc:
                failed.append(dataset)
        except Exception as exc:  # noqa: BLE001
            # One section failing must not cost the other nine, but the run still
            # goes red so somebody looks -- the same contract the sibling
            # -data repos' `00_all` orchestrators use.
            print(f"::warning ::{dataset} failed: {exc!r}", flush=True)
            failed.append(dataset)
        finally:
            print("::endgroup::", flush=True)
    for item in failed:
        print(f"::error ::{item} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
