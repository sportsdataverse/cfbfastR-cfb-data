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


#: Card ``model_type`` and training label per bundle asset. Derived from the
#: published cards; `cfb_cp_model` and `fd_model` ship no card at all, which is
#: why their entries are declared here rather than read back.
BUNDLE_MODELS: dict[str, tuple[str, str]] = {
    "ep_model": ("ep", "next_score_label"),
    "fg_model": ("fg", "made"),
    "wp_naive": ("wp_naive", "win"),
    "wp_spread": ("wp_spread", "win"),
    "cfb_cp_model": ("cp", "complete"),
    "xpass_model": ("xpass", "pass"),
    "two_pt_model": ("two_pt", "success"),
    "fd_model": ("fourth_down", "conversion"),
    "qbr_model": ("qbr", "qbr"),
}


def rebuild_all(artifacts_dir: Path | str) -> list[Path]:
    """Rebuild the card for every ``.ubj`` in ``artifacts_dir``.

    Writes each card as ``<stem>.card.json`` -- the filename the published
    bundle actually reads (e.g. ``fg_model.card.json``). ``rebuild_card``
    delegates to ``write_xgb_model_card()``, which writes the bare
    ``<stem>.json`` sibling; that file is renamed into place here so no stray
    bare-``.json`` card is left beside the ``.card.json`` one consumers read.

    Raises:
        RuntimeError: If any booster cannot be read. A skipped model would
            republish an incomplete bundle, which is the failure this exists to
            prevent -- so one bad file fails the run rather than being logged.
    """
    artifacts_dir = Path(artifacts_dir)
    written: list[Path] = []
    for ubj in sorted(artifacts_dir.glob("*.ubj")):
        model_type, label = BUNDLE_MODELS.get(ubj.stem, (ubj.stem, "y"))
        try:
            raw_card = rebuild_card(ubj, model_type=model_type, label=label)
        except Exception as exc:  # noqa: BLE001 - re-raised with the file name
            raise RuntimeError(f"could not rebuild card for {ubj.name}: {exc}") from exc
        card_path = ubj.with_suffix(".card.json")
        raw_card.replace(card_path)
        written.append(card_path)
    return written
