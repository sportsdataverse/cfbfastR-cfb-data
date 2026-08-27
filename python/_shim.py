"""Shared entrypoint helper for the numbered dataset shims.

Not a stage: the name does not match the ``<NN>_<key>_creation`` pattern the
parity gate scans for, so it is invisible to ``tests/test_r_python_parity.py``.

Exists so the 24 shims state their dataset ONCE and cannot disagree with
``cfb_data_build.cli`` about which one they build.
"""

from __future__ import annotations

import sys

from cfb_data_build.cli import main

_FLAG = "--dataset"


def _reject_override(argv: list[str], dataset: str) -> None:
    """Refuse a caller-supplied ``--dataset`` instead of silently winning.

    The shims forward the caller's argv into `main`, so the fixed selector and a
    caller's one both reach argparse and the LAST occurrence wins. Ordering the
    fixed selector last would make the shim authoritative, but it would also
    silently ignore an explicit flag -- and a run that quietly builds something
    other than what was asked for is the failure mode these pipelines keep
    getting bitten by. Fail loudly instead.
    """
    for i, arg in enumerate(argv):
        if arg == _FLAG or arg.startswith(f"{_FLAG}="):
            given = (
                arg.split("=", 1)[1]
                if "=" in arg
                else (argv[i + 1] if i + 1 < len(argv) else "")
            )
            raise SystemExit(
                f"error: this entrypoint always builds {dataset!r}, so "
                f"{_FLAG} {given!r} would be ambiguous. Run the {given!r} shim "
                f"instead, or use: python -m cfb_data_build {_FLAG} {given} ..."
            )


def run_dataset(dataset: str, argv: list[str] | None = None) -> int:
    """Build exactly ``dataset``, forwarding the caller's season/output args."""
    argv = list(sys.argv[1:] if argv is None else argv)
    _reject_override(argv, dataset)
    return main([_FLAG, dataset, *argv])


def run_many(order: list[str], argv: list[str] | None = None) -> int:
    """Build several datasets in order; one failure does not cost the rest."""
    argv = list(sys.argv[1:] if argv is None else argv)
    for dataset in order:
        _reject_override(argv, dataset)
    failed: list[str] = []
    for dataset in order:
        print(f"::group::{dataset}", flush=True)
        try:
            if main([_FLAG, dataset, *argv]):
                failed.append(dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"::warning ::{dataset} failed: {exc!r}", flush=True)
            failed.append(dataset)
        finally:
            print("::endgroup::", flush=True)
    for item in failed:
        print(f"::error ::{item} failed", flush=True)
    return 1 if failed else 0
