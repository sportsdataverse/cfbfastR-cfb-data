"""Rebuild a model card from a saved booster, without retraining.

The cards in the shipped `cfb_model_artifacts` bundle were written 2026-08-02,
before `_era_contract()` landed (#71), so the four era-consuming models publish
`era_contract: null` and every consumer keeps its own private copy of the era
cuts instead. That duplication is what caused cfb-data#70. Rebuilding the card
from the booster's own `feature_names` republishes the contract without touching
a single model byte.

Nothing is retrained here, so nothing that describes the TRAINING RUN may move.
A rebuild MERGES into the card already on disk: it may only add `era_contract`
and refresh `features`/`n_features`. `trained_date`, `hyperparameters`,
`objective`, `training_seasons`, `n_training_rows`, `num_boost_round` and
`training_frame` are carried through verbatim, because a booster cannot
regenerate them and a card that re-stamps `trained_date` claims a retrain that
never happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import xgboost as xgb

from cfb_model_build.model_training.model_card import _introspect_features, write_xgb_model_card

#: Card fields that describe the training run rather than the booster. A rebuild
#: cannot derive any of them, so they are read back off the existing card and
#: passed through unchanged.
PRESERVED_FIELDS: tuple[str, ...] = (
    "objective",
    "training_seasons",
    "n_training_rows",
    "hyperparameters",
    "num_boost_round",
    "training_frame",
    "trained_date",
    "xgboost_version",
    "source",
    "metrics",
)


def rebuild_card(model_path: Path | str, *, model_type: str, label: str) -> Path:
    """Rewrite ``<model_path>.card.json`` from the booster's own feature names.

    Merges into the existing card when there is one: every field in
    :data:`PRESERVED_FIELDS` is carried through verbatim, so the rebuild adds
    ``era_contract`` and refreshes ``features``/``n_features`` and nothing else.

    Args:
        model_path: Path to a saved ``.ubj`` booster.
        model_type: Card ``model_type`` (e.g. ``"fg"``, ``"xpass"``).
        label: Training label name recorded on the card.

    Returns:
        Path to the written ``<stem>.card.json`` -- the filename the published
        bundle actually reads. ``write_xgb_model_card()`` writes the bare
        ``<stem>.json`` sibling; it is renamed into place here so no direct
        caller leaves a stray bare-``.json`` beside the stale real card.

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

    card_path = model_path.with_suffix(".card.json")
    prior: dict = {}
    if card_path.exists():
        prior = json.loads(card_path.read_text(encoding="utf-8"))
    carried = {k: prior[k] for k in PRESERVED_FIELDS if k in prior}

    raw_card = write_xgb_model_card(
        model_path,
        model_type=model_type,
        label=label,
        features=features,
        model=booster,
        hyperparams=prior.get("hyperparameters"),
        n_rows=prior.get("n_training_rows"),
        seasons=prior.get("training_seasons"),
        # `extra` merges last, which is how a field the signature has no
        # parameter for (the original trained_date) survives the rebuild.
        extra=carried or None,
    )
    return raw_card.replace(card_path)


#: Card ``model_type`` and training label per bundle asset. The seven models that
#: ship a card were read back from it (``jq -r '.model_type, .label'``); only
#: `cfb_cp_model` and `fd_model`, which ship no card at all, are declared here
#: from their training code. Pinned by
#: ``test_bundle_models_matches_the_published_cards``.
BUNDLE_MODELS: dict[str, tuple[str, str]] = {
    "ep_model": ("ep", "next_score_label"),
    "fg_model": ("fg", "fg_made"),
    "wp_naive": ("wp_naive", "label"),
    "wp_spread": ("wp_spread", "label"),
    "cfb_cp_model": ("cp", "complete"),
    "xpass_model": ("xpass", "is_pass"),
    "two_pt_model": ("two_pt", "two_point_success"),
    "fd_model": ("fourth_down", "conversion"),
    "qbr_model": ("qbr", "qbr"),
}


def rebuild_all(artifacts_dir: Path | str) -> list[Path]:
    """Rebuild the card for every ``.ubj`` in ``artifacts_dir``.

    Every stem is checked against :data:`BUNDLE_MODELS` before anything is
    written, so an unmapped model fails the run before it can half-replace the
    bundle. Writing itself is NOT transactional: a booster that fails to load
    part-way through leaves the cards already written in place. Re-run after
    fixing the bad file -- a rebuild is idempotent.

    Raises:
        FileNotFoundError: If ``artifacts_dir`` is not a directory, or holds no
            ``.ubj`` at all. Silently exiting 0 on a typo'd path would read as a
            successful republish.
        KeyError: If a ``.ubj`` stem is not in :data:`BUNDLE_MODELS`. Guessing
            ``label="y"`` would publish a card that lies about the model.
        RuntimeError: If any booster cannot be read. A skipped model would
            republish an incomplete bundle, which is the failure this exists to
            prevent -- so one bad file fails the run rather than being logged.
    """
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"artifacts dir does not exist: {artifacts_dir}")
    boosters = sorted(artifacts_dir.glob("*.ubj"))
    if not boosters:
        raise FileNotFoundError(f"no .ubj boosters found in {artifacts_dir}")
    unmapped = [b.name for b in boosters if b.stem not in BUNDLE_MODELS]
    if unmapped:
        raise KeyError(
            f"unmapped booster(s) {unmapped}: add them to BUNDLE_MODELS with the "
            "model_type and label they were trained with"
        )

    written: list[Path] = []
    for ubj in boosters:
        model_type, label = BUNDLE_MODELS[ubj.stem]
        try:
            written.append(rebuild_card(ubj, model_type=model_type, label=label))
        except Exception as exc:  # noqa: BLE001 - re-raised with the file name
            raise RuntimeError(f"could not rebuild card for {ubj.name}: {exc}") from exc
    return written
