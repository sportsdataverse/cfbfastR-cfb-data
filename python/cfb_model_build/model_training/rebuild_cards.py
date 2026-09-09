"""Rebuild a model card from a saved booster, without retraining.

The cards in the shipped `cfb_model_artifacts` bundle were written 2026-08-02,
before `_era_contract()` landed (#71), so the four era-consuming models publish
`era_contract: null` and every consumer keeps its own private copy of the era
cuts instead. That duplication is what caused cfb-data#70. Rebuilding the card
from the booster's own `feature_names` republishes the contract without touching
a single model byte.
"""

from __future__ import annotations

from pathlib import Path

import xgboost as xgb

from cfb_model_build.model_training.model_card import _introspect_features, write_xgb_model_card


def rebuild_card(model_path: Path | str, *, model_type: str, label: str) -> Path:
    """Rewrite ``<model_path>.json`` from the booster's own feature names.

    Args:
        model_path: Path to a saved ``.ubj`` booster.
        model_type: Card ``model_type`` (e.g. ``"fg"``, ``"xpass"``).
        label: Training label name recorded on the card.

    Returns:
        Path to the written card.

    Raises:
        ValueError: If the booster carries no feature names — a card with no
            features would validate nothing for a downstream caller.
    """
    model_path = Path(model_path)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    features = _introspect_features(booster)
    if not features:
        raise ValueError(f"{model_path.name} carries no feature names; cannot write a usable card")
    return write_xgb_model_card(
        model_path,
        model_type=model_type,
        label=label,
        features=features,
        model=booster,
    )
