"""Usage / situational datasets built from a real final.json (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cfb_data_build.build import build_dataset_frame
from cfb_data_build.config import PKG_FUNCTION, REGISTRY, USAGE_ADV_ORDER, USAGE_LEADERBOARD_ORDER

FIX = Path(__file__).parent / "fixtures" / "final_401628455.json"


def _game():
    return json.loads(FIX.read_text(encoding="utf-8"))


def test_registry_and_sidecars_cover_every_usage_dataset():
    for key in USAGE_ADV_ORDER + USAGE_LEADERBOARD_ORDER:
        spec = REGISTRY[key]
        assert spec.usage_section is not None and spec.tag in PKG_FUNCTION, key
    assert all(REGISTRY[k].aggregate for k in USAGE_LEADERBOARD_ORDER)
    assert not any(REGISTRY[k].aggregate for k in USAGE_ADV_ORDER)


def test_usage_sections_build_from_the_fixture_final():
    game = _game()
    player = build_dataset_frame(REGISTRY["adv_player_usage"], game)
    assert player.height > 0 and {"game_id", "season", "pos_team", "target_share", "fd_td_rate"} <= set(player.columns)
    assert player["game_id"].unique().to_list() == [int(game["id"])]
    for _, g in player.group_by("pos_team"):
        assert g["target_share"].fill_null(0).sum() <= 1.0 + 1e-9
    team = build_dataset_frame(REGISTRY["adv_team_usage"], game)
    assert team.height == 2 and "third_down_over_expected" in team.columns
    scripting = build_dataset_frame(REGISTRY["adv_drive_scripting"], game)
    assert set(scripting["script"].to_list()) == {"scripted", "non_scripted"}
    tackles = build_dataset_frame(REGISTRY["adv_tackles"], game)
    assert tackles.height
    for _, g in tackles.group_by("def_pos_team"):
        assert abs(g["tackle_share"].sum() - 1.0) < 1e-9


def test_stored_participants_give_real_tacklers_and_position_groups():
    """The stored participants predate *_position_id and hold numpy-repr list cells.

    "['5152441' '5220449']" once decoded to ONE glued id, and position groups were
    empty; the game roster now supplies positions and every tackler is a real athlete.
    """
    game = _game()
    roster_ids = {str(r["athlete_id"]) for r in game["game_rosters"]}
    cells = [v for r in game["play_participants"] for k, v in r.items() if k.endswith("_ids") and isinstance(v, str)]
    assert any("' '" in v for v in cells)  # the numpy shape under test is really in the fixture
    tackles = build_dataset_frame(REGISTRY["adv_tackles"], game)
    unknown = set(tackles["player_id"].drop_nulls().cast(pl.Utf8)) - roster_ids
    assert tackles.height and not unknown, sorted(unknown)[:5]
    assert tackles["position_group"].null_count() == 0
    # (position_group_usage is not asserted: this fixture's plays carry no rusher /
    # receiver ids, so there is nothing to group -- sdv-py's usage-box tests cover it)
    group_tackles = build_dataset_frame(REGISTRY["adv_position_group_tackles"], game)
    assert set(group_tackles["position_group"].to_list()) & {"DL", "LB", "DB"}


def test_leaderboard_aggregation_recomputes_rates():
    from sportsdataverse.football.usage_box import aggregate_usage_box

    game = _game()
    per_game = build_dataset_frame(REGISTRY["adv_player_usage"], game)
    lb = aggregate_usage_box("player_usage", [per_game, per_game])
    assert lb.height == per_game.height and (lb["games"] == 2).all() and "game_id" not in lb.columns
    top = per_game.sort("targets", descending=True).row(0, named=True)
    key = "player_id" if top.get("player_id") is not None else "player_name"
    row = lb.filter(pl.col(key) == top[key]).row(0, named=True)
    assert row["targets"] == 2 * top["targets"] and abs(row["target_share"] - top["target_share"]) < 1e-9
