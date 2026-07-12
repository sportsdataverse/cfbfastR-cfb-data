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
from cfb_data_build.reshapers import reshape_linescores, reshape_power_index

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


# ``team_box_extra`` / ``power_index`` carry stamped game_id/season/week scalars
# alongside their real blocks (json.load yields bare ints; R's simplifyVector=
# FALSE yields length-1 lists that ``$linescores`` skips as NULL). Both reshapers
# must skip those scalars rather than treat them as blocks.
def _stamped(gid: int = 999, season: int = 2010, week: int = 1) -> dict:
    return {"game_id": gid, "season": season, "week": week}


def test_reshape_linescores_skips_stamped_scalars_and_numbers_overtime() -> None:
    # 4 regulation quarters + 2 OT periods -> periods 1..6 (index-based, == R
    # seq_along). team_box_extra also carries the stamped scalars, which must be
    # skipped, not crash on ``int.get``.
    def team(vals: list[str]) -> dict:
        return {"linescores": [{"displayValue": v} for v in vals]}

    g = {
        "id": 999,
        "season": 2010,
        "team_box_extra": {
            "52": team(["14", "9", "0", "0", "3", "6"]),
            "201": team(["7", "0", "8", "8", "3", "3"]),
            **_stamped(),
        },
    }
    df = reshape_linescores(g)
    assert df.height == 12  # 2 teams x 6 periods
    t52 = df.filter(pl.col("team_id") == 52).sort("period")
    assert t52.get_column("period").to_list() == [1, 2, 3, 4, 5, 6]
    assert t52.get_column("value").to_list() == ["14", "9", "0", "0", "3", "6"]


def test_reshape_power_index_no_fpi_game_is_empty_not_crash() -> None:
    # A no-FPI game ships power_index as only the stamped scalars (no ``items``);
    # must return an empty frame, never iterate the dict's string keys.
    assert reshape_power_index({"power_index": _stamped(), "id": 999}).height == 0
    # populated wrapper still flattens its items
    g = {
        "id": 999,
        "season": 2024,
        "week": 1,
        "power_index": {"items": [{"stat": "fpi", "value": 1.5}], **_stamped()},
    }
    assert reshape_power_index(g).height == 1
