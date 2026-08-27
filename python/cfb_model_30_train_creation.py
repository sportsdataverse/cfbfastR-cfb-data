"""Model stage 30 (train) -- fit + gate the shared model bundle (ep, wp, cp, fg, qbr, xpass, 2pt, punt).

Thin shim over ``python -m cfb_model_build.model_training``: the directory listing IS the model pipeline,
mirroring the numbered dataset stages beside it.

Stage order is ingest -> features -> train -> evaluate/gate -> package ->
publish -> integrate. The numbers leave gaps on purpose:

* **10** ingest, **30-34** the model families (each owns its own
  features/train/gate/package -- they are separate MODELS, not separate stages
  of one model), **60** publish, **70** reports.
* 20 (shared feature build) and 40/50 (a standalone gate/package stage) are
  HOLES: today each family gates and packages inside its own train step. If one
  of those is ever factored out, it lands on its reserved number rather than
  renumbering everything after it.

Gates sit upstream of publish and are never lowered. Every artifact this
pipeline publishes needs a row in the **Model registry** in ``CLAUDE.md`` --
model, artifact, release tag, training data, fitting script, gates, last
retrain, cadence. ``tests/test_model_registry.py`` enforces that each row's
fitting script resolves to a real stage here.

Example:
    Forwarded straight through to the package CLI::

        source scripts/_venv.sh
        "$PY" python/cfb_model_30_train_creation.py --help
"""

from __future__ import annotations

import runpy
import sys

PACKAGE = "cfb_model_build.model_training"

if __name__ == "__main__":
    sys.argv[0] = f"python -m {PACKAGE}"
    runpy.run_module(PACKAGE, run_name="__main__", alter_sys=True)
