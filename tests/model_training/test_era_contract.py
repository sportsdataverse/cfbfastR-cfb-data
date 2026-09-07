"""The rule-era contract every model card carries.

cfbfastR-cfb-data#70: both consumers (cfbfastR, sportsdataverse-py) kept their
own copy of the ordinal era cut, both drifted to a third cut of 2017 that this
repo has never used, and 2018-2020 were scored an era off against models trained
with them in bucket 2. The card now states the contract so a consumer can read
it instead of keeping a copy.
"""

from __future__ import annotations

import json

from cfb_model_build.model_training import constants as C
from cfb_model_build.model_training.model_card import write_xgb_model_card


def _card(tmp_path, name, features):
    p = tmp_path / f"{name}.ubj"
    p.write_bytes(b"")
    write_xgb_model_card(p, model_type=name, label="y", features=features)
    return json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))


def test_ordinal_model_declares_the_ordinal_encoding(tmp_path):
    card = _card(tmp_path, "xpass", ["down", "distance", "era", "period"])
    era = card["era_contract"]
    assert era["encoding"] == "ordinal"
    assert era["columns"] == ["era"]
    assert era["cuts"] == list(C.ERA_BOUNDS)


def test_one_hot_model_declares_the_one_hot_encoding(tmp_path):
    card = _card(tmp_path, "fg", ["yards_to_goal", "era0", "era1", "era2", "era3"])
    era = card["era_contract"]
    assert era["encoding"] == "one_hot"
    assert era["columns"] == C.ERA_ONEHOT_COLS
    assert era["cuts"] == list(C.ERA_BOUNDS)


def test_both_encodings_publish_the_same_cuts(tmp_path):
    """The whole point: the encodings differ, the cutpoints do not.

    `_era()` and `_era_onehot()` both derive from ERA_BOUNDS, so a card that
    published different cuts per encoding would be describing a model this repo
    cannot produce.
    """
    ordinal = _card(tmp_path, "xpass", ["era"])["era_contract"]
    onehot = _card(tmp_path, "fg", C.ERA_ONEHOT_COLS)["era_contract"]
    assert ordinal["cuts"] == onehot["cuts"]
    assert ordinal["encoding"] != onehot["encoding"]


def test_the_published_cut_is_2020_not_2017(tmp_path):
    """Regression pin for #70.

    A 2017 third cut puts 2018-2020 in bucket 3; the models were trained with
    them in bucket 2. This asserts the value a consumer would read.
    """
    era = _card(tmp_path, "xpass", ["era"])["era_contract"]
    assert era["cuts"][2] == 2020
    assert era["buckets"][2] == "2014-2020"
    assert era["buckets"][3] == ">=2021"


def test_a_model_without_an_era_feature_publishes_no_contract(tmp_path):
    card = _card(tmp_path, "plain", ["down", "distance", "yards_to_goal"])
    assert "era_contract" not in card


def test_mixed_encodings_are_rejected(tmp_path):
    """Both encodings on one model is not a shape the trainer can produce."""
    import pytest

    with pytest.raises(ValueError, match="not interchangeable"):
        _card(tmp_path, "mixed", ["down", "era", "era0", "era1", "era2", "era3"])


def test_incomplete_one_hot_is_rejected(tmp_path):
    """Four bucket labels beside three columns would describe a different model."""
    import pytest

    with pytest.raises(ValueError, match="incomplete one-hot"):
        _card(tmp_path, "partial", ["yards_to_goal", "era0", "era1", "era2"])
