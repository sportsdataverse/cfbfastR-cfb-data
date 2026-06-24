"""Unit tests for the reshape primitives, pbp output schema, and build dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.build import build_dataset_frame
from cfb_data_build.config import REGISTRY
from cfb_data_build.pbp import (
    PBP_DROP_LAG_LEAD,
    apply_pbp_output_schema,
    build_pbp_frame,
)
from cfb_data_build.reshape import _norm_cell, bind_games, flat_block_frame

FIX = Path(__file__).parent / "fixtures"
GID = 401628455


def _game() -> dict:
    return json.loads((FIX / f"final_{GID}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ([], None),
        (["solo"], "solo"),  # length-1 list unboxed
        ([1], 1),
        (5, 5),
        ("x", "x"),
        (["a", "b"], '["a","b"]'),  # length-2+ -> compact json
        ({"k": 1, "j": 2}, '{"k":1,"j":2}'),  # dict -> compact json
        ([{"id": 1}], '{"id":1}'),  # length-1 list of object -> unbox then json
    ],
)
def test_norm_cell_matches_r_dict_to_row(value: object, expected: object) -> None:
    assert _norm_cell(value) == expected


def test_flat_block_frame_empty_block_is_empty_frame() -> None:
    assert flat_block_frame(None, {"id": 1, "season": 2024}).is_empty()
    assert flat_block_frame([], {"id": 1, "season": 2024}).is_empty()


def test_flat_block_frame_stamps_identity() -> None:
    g = {"id": "401", "season": "2024", "week": "1"}
    df = flat_block_frame([{"play_id": "a"}, {"play_id": "b"}], g)
    assert df.height == 2
    assert df["game_id"].to_list() == [401, 401]  # overwritten as int
    assert df["season"].to_list() == [2024, 2024]
    assert df["week"].to_list() == [1, 1]


def test_apply_pbp_output_schema_tiers_and_order() -> None:
    g = _game()
    raw = flat_block_frame(g["plays"], g)
    full = apply_pbp_output_schema(raw, "full")
    default = apply_pbp_output_schema(raw, "default")
    lean = apply_pbp_output_schema(raw, "lean")

    # default drops more than full; lean drops more than default
    assert default.width < full.width
    assert lean.width <= default.width
    # default drops the lag/lead intermediates
    assert not (set(PBP_DROP_LAG_LEAD) & set(default.columns))
    # canonical order: season leads, game_id second (the tier head)
    assert default.columns[:3] == ["season", "game_id", "game_play_number"]


def test_apply_pbp_output_schema_rejects_bad_tier() -> None:
    with pytest.raises(ValueError):
        apply_pbp_output_schema(pl.DataFrame({"a": [1]}), "bogus")


def test_bind_games_drops_empty_and_unions_columns() -> None:
    out = bind_games(
        [
            pl.DataFrame({"a": [1], "b": [2]}),
            None,
            pl.DataFrame(),
            pl.DataFrame({"a": [3], "c": [4]}),
        ]
    )
    assert out.height == 2
    assert set(out.columns) == {"a", "b", "c"}


def test_build_dataset_frame_dispatches_pbp_and_generic() -> None:
    g = _game()
    pp = build_dataset_frame(REGISTRY["play_participants"], g)
    pbp = build_dataset_frame(REGISTRY["pbp"], g)
    assert pp.shape == (158, 56)
    # pbp dispatch == direct conform
    assert pbp.columns == build_pbp_frame(g).columns
    assert pbp.shape == (169, 371)
