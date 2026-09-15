"""Model stage 35 (train) -- the matchup pipeline's fitted rates.

Thin shim over ``python -m cfb_model_build.cfb_matchup``: the directory listing
IS the model pipeline, mirroring the numbered dataset stages beside it. 35 is
the next free number in the 30-34 model-family block; the family owns three
artifacts (``scoring_opp`` and ``rush_expect`` logistic glms, the 120
``wepa_weights``) bundled at ``cfb_model_build/cfb_matchup/artifacts/`` and
read by the stage-41/42 dataset builders. The CLI writes to a candidate
dir; promotion into the bundle (``--promote``) is a deliberate step after
the gates below pass.

Gates sit upstream of publish and are never lowered: a refit must reproduce
the shipped coefficients on the shipped training rows within 1e-6 relative
(``tests/cfb_matchup/test_glm_parity.py``, integration) before its artifact
replaces the bundled one. Every artifact has a row in ``models/REGISTRY.md``.

Example:
    Forwarded straight through to the package CLI::

        source scripts/_venv.sh
        "$PY" python/cfb_model_35_matchup_creation.py train-scoring-opp --seasons 2014-2024 --out-dir python/.cache/matchup/candidate
"""

from __future__ import annotations

PACKAGE = "cfb_model_build.cfb_matchup"

if __name__ == "__main__":
    # Fingerprint + ledger runtime (Track C steps 3+5): skips when
    # hash(package subtree, argv) is unchanged and the argv's --out artifacts
    # exist; --force retrains. Appends models/ledger.jsonl on success.
    from _model_stage import run

    raise SystemExit(run(PACKAGE))
