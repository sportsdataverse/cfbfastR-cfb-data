"""Fingerprint + ledger runtime for the numbered model stages (Track C steps 3+5).

Each ``cfb_model_NN_*_creation.py`` shim forwards to its package CLI via
``run(PACKAGE)``: compute ``hash(package subtree, argv)``, skip when unchanged
AND every ``--out``-style artifact named in argv already exists (``--force``
retrains), else run the package as ``__main__`` and, on success, record the
fingerprint and append a ``models/ledger.jsonl`` line. When argv names no
output paths the stage never skips (a skip that can't verify its artifacts is
silent staleness).

Lives at the python/ top level beside ``_shim.py`` — the repo's documented
layout keeps runtime helpers the shims import flat (entry points at the top,
implementations in packages).
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "python" / "artifacts" / ".fingerprints.json"
LEDGER = REPO / "models" / "ledger.jsonl"

_OUT_FLAGS = {"--out", "--out-dir", "--oof-out", "--report-out"}


def _package_dir(package: str) -> Path:
    return REPO / "python" / Path(*package.split("."))


def _compute(package: str, argv: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(_package_dir(package).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(_package_dir(package)).as_posix().encode())
        h.update(p.read_bytes())
    h.update(json.dumps(argv, default=str).encode())
    return h.hexdigest()[:16]


def _out_paths(argv: list[str]) -> list[Path]:
    outs = []
    for i, a in enumerate(argv[:-1]):
        if a in _OUT_FLAGS:
            outs.append(Path(argv[i + 1]))
    return outs


def _load_store() -> dict:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run(package: str) -> int:
    argv = [a for a in sys.argv[1:]]
    force = "--force" in argv
    argv = [a for a in argv if a != "--force"]

    digest = _compute(package, argv)
    key = f"{package}::{' '.join(argv[:1]) or 'default'}"
    outs = _out_paths(argv)
    if (
        not force
        and outs
        and _load_store().get(key) == digest
        and all(o.exists() for o in outs)
    ):
        print(
            f"[{package}] fingerprint unchanged + artifacts present -> skip "
            f"(--force to retrain)"
        )
        return 0

    sys.argv = [f"python -m {package}", *argv]
    rc = 0
    try:
        runpy.run_module(package, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        rc = int(exc.code or 0)

    if rc == 0:
        store = _load_store()
        store[key] = digest
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(
            json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "package": package,
                        "argv": argv,
                        "fingerprint": digest,
                        "artifacts": [str(o) for o in outs],
                        "gates": "run inside the package CLI (never lowered)",
                        "delta_vs_champion": None,
                        "in_published_data": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return rc
