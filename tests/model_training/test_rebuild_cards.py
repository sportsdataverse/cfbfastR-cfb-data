"""Rebuilding a card from a shipped booster must recover the era contract.

The seven published cards were written 2026-08-02, before `_era_contract()`
landed on 2026-09-07 (#71), so every one of them carries `era_contract: null` --
including fg_model and qbr_model, which consume one-hot era0..era3. Nothing is
retrained here; only the .card.json sibling is rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import xgboost as xgb

from cfb_model_build.model_training import constants as C
from cfb_model_build.model_training.rebuild_cards import rebuild_card


def _booster(tmp_path, name, features):
    """A minimal real Booster carrying `features` as its feature_names."""
    import numpy as np

    X = np.zeros((4, len(features)), dtype=float)
    y = np.array([0, 1, 0, 1])
    dm = xgb.DMatrix(X, label=y, feature_names=list(features))
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 1}, dm, num_boost_round=1)
    p = tmp_path / f"{name}.ubj"
    bst.save_model(str(p))
    return p


def test_rebuild_recovers_the_one_hot_era_contract(tmp_path):
    p = _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    card_path = rebuild_card(p, model_type="fg", label="fg_made")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    era = card["era_contract"]
    assert era["encoding"] == "one_hot"
    assert era["columns"] == C.ERA_ONEHOT_COLS
    assert era["cuts"] == list(C.ERA_BOUNDS)


def test_rebuild_recovers_the_ordinal_era_contract(tmp_path):
    p = _booster(tmp_path, "xpass_model", ["down", "distance", "era"])
    card = json.loads(rebuild_card(p, model_type="xpass", label="is_pass").read_text(encoding="utf-8"))
    assert card["era_contract"]["encoding"] == "ordinal"
    assert card["era_contract"]["cuts"] == list(C.ERA_BOUNDS)


def test_a_model_without_era_gets_no_contract(tmp_path):
    """`era_contract: null` is CORRECT for ep_model and the WP pair."""
    p = _booster(tmp_path, "ep_model", ["TimeSecsRem", "yards_to_goal", "distance"])
    card = json.loads(rebuild_card(p, model_type="ep", label="next_score_label").read_text(encoding="utf-8"))
    assert "era_contract" not in card or card["era_contract"] is None


def test_features_are_taken_from_the_booster_in_order(tmp_path):
    """Feature ORDER is load-bearing for XGBoost; the card is what fixes it."""
    feats = ["down", "distance", "yards_to_goal", "era"]
    p = _booster(tmp_path, "xpass_model", feats)
    card = json.loads(rebuild_card(p, model_type="xpass", label="is_pass").read_text(encoding="utf-8"))
    assert card["features"] == feats


def test_a_booster_without_feature_names_is_rejected(tmp_path):
    """A card with no features would validate nothing downstream."""
    import numpy as np

    dm = xgb.DMatrix(np.zeros((4, 2)), label=np.array([0, 1, 0, 1]))
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 1}, dm, num_boost_round=1)
    p = tmp_path / "nameless.ubj"
    bst.save_model(str(p))
    with pytest.raises(ValueError, match="no feature names"):
        rebuild_card(p, model_type="mystery", label="y")


def test_the_model_file_is_not_modified(tmp_path):
    """This plan rewrites cards only; any .ubj diff is a failure."""
    p = _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    before = p.read_bytes()
    rebuild_card(p, model_type="fg", label="fg_made")
    assert p.read_bytes() == before


def test_cli_rebuilds_every_booster_in_a_directory(tmp_path):
    """One command must cover all nine bundle models, not one at a time.

    Drives the real `rebuild-cards` entry point: the argparse wiring is the half
    an operator actually runs, and calling `rebuild_all` directly would leave it
    untested.
    """
    from cfb_model_build.model_training.cli import main

    _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    _booster(tmp_path, "xpass_model", ["down", "distance", "era"])
    assert main(["rebuild-cards", "--artifacts-dir", str(tmp_path)]) == 0
    written = sorted(tmp_path.glob("*.card.json"))
    assert [p.name for p in written] == ["fg_model.card.json", "xpass_model.card.json"]
    for p in written:
        assert json.loads(p.read_text(encoding="utf-8"))["era_contract"] is not None


def test_rebuild_all_reports_a_model_it_cannot_read(tmp_path):
    """A silently skipped model would republish an incomplete bundle."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    (tmp_path / "qbr_model.ubj").write_bytes(b"not a booster")
    with pytest.raises(RuntimeError, match=r"qbr_model\.ubj"):
        rebuild_all(tmp_path)


def test_rebuild_all_rolls_back_every_card_when_one_booster_fails(tmp_path):
    """A half-rebuilt directory looks valid and would republish an inconsistent bundle."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    stale = {"model_type": "fg", "label": "fg_made", "trained_date": "2026-08-02"}
    fg_card = tmp_path / "fg_model.card.json"
    fg_card.write_text(json.dumps(stale, indent=2), encoding="utf-8")
    before = fg_card.read_bytes()

    # Sorts after fg_model, so fg_model's card is already rewritten when this fails.
    (tmp_path / "qbr_model.ubj").write_bytes(b"not a booster")
    with pytest.raises(RuntimeError):
        rebuild_all(tmp_path)

    assert fg_card.read_bytes() == before, "an earlier card survived a failed run"
    assert not (tmp_path / "qbr_model.card.json").exists()
    assert not (tmp_path / "fg_model.json").exists()


def test_rebuild_all_removes_a_card_it_created_when_a_later_booster_fails(tmp_path):
    """Rollback must DELETE cards that did not exist, not just restore ones that did."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    (tmp_path / "qbr_model.ubj").write_bytes(b"not a booster")
    with pytest.raises(RuntimeError):
        rebuild_all(tmp_path)

    assert list(tmp_path.glob("*.json")) == []


def test_rebuild_all_writes_the_bundle_card_filename(tmp_path):
    """The bundle publishes <model>.card.json. Writing <model>.json instead would
    leave the stale card consumers actually read untouched, and the republish
    would succeed while changing nothing."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    written = rebuild_all(tmp_path)
    assert [p.name for p in written] == ["fg_model.card.json"]
    assert (tmp_path / "fg_model.card.json").exists()
    assert not (tmp_path / "fg_model.json").exists(), "stray bare-.json card left behind"


#: The seven `cfb_model_artifacts` cards exactly as published 2026-08-02, committed
#: so these two tests -- the only coverage for both Critical findings this branch
#: shipped -- always run. They used to point at /tmp/cards_preimage, which nothing
#: in this repo or in CI ever creates, so both silently skipped everywhere but the
#: one workstation that had rebuilt the bundle by hand.
PREIMAGE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "model_training" / "published_cards"


def test_the_published_card_fixtures_are_present():
    """A missing fixture must fail, never skip: a skip reads as coverage."""
    assert sorted(p.name for p in PREIMAGE_DIR.glob("*.card.json")) == [
        "ep_model.card.json",
        "fg_model.card.json",
        "qbr_model.card.json",
        "two_pt_model.card.json",
        "wp_naive.card.json",
        "wp_spread.card.json",
        "xpass_model.card.json",
    ]


def test_bundle_models_matches_the_published_cards():
    """BUNDLE_MODELS is hand-maintained, so pin it to the cards it claims to describe.

    Five of the nine labels were wrong when this table shipped (`made` for
    `fg_made`, `success` for `two_point_success`, `pass` for `is_pass`, `win` for
    both WP models). Nothing caught it because every other test passes
    model_type/label explicitly.
    """
    from cfb_model_build.model_training.rebuild_cards import BUNDLE_MODELS

    published = {}
    for card_file in sorted(PREIMAGE_DIR.glob("*.card.json")):
        card = json.loads(card_file.read_text(encoding="utf-8"))
        published[card_file.name.removesuffix(".card.json")] = (card["model_type"], card["label"])

    assert published, "no published cards found to pin against"
    for stem, expected in published.items():
        assert BUNDLE_MODELS[stem] == expected, f"{stem} disagrees with its published card"

    # The two cardless models are declared, not read back; assert only that the
    # table stays exactly the nine bundle assets.
    assert set(BUNDLE_MODELS) - set(published) == {"cfb_cp_model", "fd_model"}


def test_rebuild_preserves_every_published_provenance_field():
    """A rebuild may ADD era_contract and refresh features. Nothing else moves.

    Asserted against the inverted set: everything on a published card that is not
    recomputed from the booster is carried, so a field added to the card format
    later cannot silently drop the way a fixed whitelist would have dropped it.
    """
    from cfb_model_build.model_training.rebuild_cards import REBUILT_FIELDS

    for card_file in sorted(PREIMAGE_DIR.glob("*.card.json")):
        card = json.loads(card_file.read_text(encoding="utf-8"))
        for field in ("objective", "training_seasons", "n_training_rows", "hyperparameters",
                      "num_boost_round", "training_frame", "trained_date", "source",
                      "xgboost_version"):
            assert field not in REBUILT_FIELDS, f"{field} would be recomputed, not carried"
            assert field in card, f"{card_file.name}: {field}"
        # Only features/n_features are recomputed on today's cards; model_type and
        # label are recomputed too but pinned equal by the test above, and
        # era_contract is the one field a rebuild adds.
        assert set(card) & REBUILT_FIELDS == {"features", "n_features", "model_type", "label"}

    # qbr_model publishes n_training_rows: null, so the "actually carries a value"
    # half is spot-checked on fg_model, which populates every provenance field.
    fg = json.loads((PREIMAGE_DIR / "fg_model.card.json").read_text(encoding="utf-8"))
    for field in ("objective", "training_seasons", "n_training_rows", "hyperparameters",
                  "num_boost_round", "training_frame", "trained_date"):
        assert fg[field] is not None


def test_rebuild_merges_into_an_existing_card(tmp_path):
    """Nothing is retrained, so trained_date and the hyperparameters must survive."""
    p = _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    prior = {
        "model_type": "fg",
        "xgboost_version": "3.2.0",
        "objective": "binary:logistic",
        "features": ["stale"],
        "n_features": 1,
        "label": "fg_made",
        "training_seasons": [2004, 2025],
        "n_training_rows": 2219971,
        "hyperparameters": {"objective": "binary:logistic", "eta": 0.1},
        "source": "cfb_final_json",
        "trained_date": "2026-08-02",
        "num_boost_round": 60,
        "training_frame": "artifacts/pbp_full_v2.parquet",
        # Not a field the rebuild knows about. The fixed whitelist this replaced
        # dropped exactly this on the next rebuild.
        "calibration": {"method": "isotonic", "fitted_on": 2025},
    }
    (tmp_path / "fg_model.card.json").write_text(json.dumps(prior, indent=2), encoding="utf-8")

    card = json.loads(rebuild_card(p, model_type="fg", label="fg_made").read_text(encoding="utf-8"))
    for k, v in prior.items():
        if k in ("features", "n_features"):
            continue
        assert card[k] == v, f"{k} changed during a rebuild that retrained nothing"
    assert card["features"] == ["yards_to_goal", *C.ERA_ONEHOT_COLS]
    assert card["era_contract"]["encoding"] == "one_hot"
    assert set(card) - set(prior) == {"era_contract"}


def test_an_unmapped_booster_is_rejected(tmp_path):
    """A tenth model must not ship a card labelled `y`."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    _booster(tmp_path, "cfb_cp_model_air_yards", ["air_yards", "down"])
    with pytest.raises(KeyError, match="cfb_cp_model_air_yards"):
        rebuild_all(tmp_path)


def test_a_missing_artifacts_dir_fails(tmp_path):
    """Exiting 0 on a typo'd path would read as a successful republish."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    with pytest.raises(FileNotFoundError):
        rebuild_all(tmp_path / "nope")


def test_an_artifacts_dir_with_no_boosters_fails(tmp_path):
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    with pytest.raises(FileNotFoundError, match=r"no \.ubj"):
        rebuild_all(tmp_path)
