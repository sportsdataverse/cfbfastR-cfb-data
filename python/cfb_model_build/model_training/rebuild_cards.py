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

from cfb_model_build.cpoe.constants import TARGET_COL as CP_TARGET_COL
from cfb_model_build.model_training.fourth_down.constants import FD_YARDS_GAINED_COL
from cfb_model_build.model_training.model_card import _introspect_features, write_xgb_model_card

#: The only card fields a rebuild is allowed to recompute: `features` /
#: `n_features` / `era_contract` are read straight off the booster, and
#: `model_type` / `label` are the caller's declared contract for the model.
#: EVERYTHING ELSE on the prior card is carried through verbatim -- the set is
#: inverted deliberately, because the whitelist this replaced silently dropped
#: any card field added after it was written.
REBUILT_FIELDS: frozenset[str] = frozenset(
    {"features", "n_features", "era_contract", "model_type", "label"}
)


def rebuild_card(model_path: Path | str, *, model_type: str, label: str) -> Path:
    """Rewrite ``<model_path>.card.json`` from the booster's own feature names.

    Merges into the existing card when there is one: every field NOT in
    :data:`REBUILT_FIELDS` is carried through verbatim, so the rebuild adds
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
    carried = {k: v for k, v in prior.items() if k not in REBUILT_FIELDS}

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
#: `cfb_cp_model` and `fd_model`, which shipped no card at all, are declared here
#: from their training code -- and their labels are IMPORTED from the constants
#: that training reads, because hand-typing them is what published `cp`/`complete`
#: and `fourth_down`/`conversion`. Pinned by
#: ``test_bundle_models_matches_the_published_cards``.
BUNDLE_MODELS: dict[str, tuple[str, str]] = {
    "ep_model": ("ep", "next_score_label"),
    "fg_model": ("fg", "fg_made"),
    "wp_naive": ("wp_naive", "label"),
    "wp_spread": ("wp_spread", "label"),
    # cpoe/train_cp.py `save_cp_model(model_type="cpoe")`, cpoe/cli.py `mtype = "cpoe"`.
    "cfb_cp_model": ("cpoe", CP_TARGET_COL),
    "xpass_model": ("xpass", "is_pass"),
    "two_pt_model": ("two_pt", "two_point_success"),
    # fourth_down/cli.py `write_xgb_model_card(model_type="fourth_down", label=FD_YARDS_GAINED_COL)`.
    "fd_model": ("fourth_down", FD_YARDS_GAINED_COL),
    "qbr_model": ("qbr", "qbr"),
}


def rebuild_all(artifacts_dir: Path | str) -> list[Path]:
    """Rebuild the card for every ``.ubj`` in ``artifacts_dir``, all-or-nothing.

    Every stem is checked against :data:`BUNDLE_MODELS` before anything is
    written, so an unmapped model fails the run before it can half-replace the
    bundle. Writing is transactional: the pre-run bytes of every card this run
    could touch are held, and any failure restores them (deleting the ones that
    did not exist), so a failed run leaves the directory exactly as it found it.
    A half-rebuilt directory looks valid and could be republished as an
    inconsistent bundle, which is the failure this exists to prevent.

    Raises:
        FileNotFoundError: If ``artifacts_dir`` is not a directory, or holds no
            ``.ubj`` at all. Silently exiting 0 on a typo'd path would read as a
            successful republish.
        KeyError: If a ``.ubj`` stem is not in :data:`BUNDLE_MODELS`. Guessing
            ``label="y"`` would publish a card that lies about the model.
        RuntimeError: If any booster cannot be read. A skipped model would
            republish an incomplete bundle, so one bad file fails the run --
            after every card already written has been rolled back.
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

    # Both the real card and the bare-.json scratch sibling write_xgb_model_card
    # lays down on the way to it.
    touched = [p for b in boosters for p in (b.with_suffix(".card.json"), b.with_suffix(".json"))]
    preimage = {p: p.read_bytes() for p in touched if p.exists()}

    written: list[Path] = []
    try:
        for ubj in boosters:
            model_type, label = BUNDLE_MODELS[ubj.stem]
            try:
                written.append(rebuild_card(ubj, model_type=model_type, label=label))
            except Exception as exc:  # noqa: BLE001 - re-raised with the file name
                raise RuntimeError(f"could not rebuild card for {ubj.name}: {exc}") from exc
    except BaseException:
        for p in touched:
            if p in preimage:
                p.write_bytes(preimage[p])
            elif p.exists():
                p.unlink()
        raise
    return written
