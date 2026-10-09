"""team_opponent_splits: adv_team_gamelog projected per team-game + adv_situational success rate."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.cli import DERIVED
from cfb_data_build.config import PKG_FUNCTION
from cfb_data_build.derived import (
    BUILDERS,
    SPECS,
    TEAM_OPPONENT_SPLITS_SCHEMA,
    build_team_opponent_splits,
)

REPO = Path(__file__).resolve().parents[2]
SEASON = 2025


def _gamelog(rows: list[dict], *, id_dtype=pl.Int64) -> pl.DataFrame:
    """adv_team_gamelog's real shape (plus one metric column the projection drops)."""
    return pl.DataFrame(
        rows,
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "season_type": pl.Int32,
            "game_id": id_dtype,
            "team_id": id_dtype,
            "team": pl.Utf8,
            "opponent_id": id_dtype,
            "opponent": pl.Utf8,
            "is_home": pl.Boolean,
            "points_for": pl.Int64,
            "points_against": pl.Int64,
            "EPA_penalty": pl.Float64,
            "EPA_per_play": pl.Float64,
            "EPA_plays": pl.Int64,
            "scrimmage_plays": pl.Int64,
        },
    )


def _row(game_id, team_id, opponent_id, epa, plays, pf, pa, *, home=True) -> dict:
    return {
        "season": SEASON,
        "week": 1,
        "season_type": 2,
        "game_id": game_id,
        "team_id": team_id,
        "team": f"team {team_id}",
        "opponent_id": opponent_id,
        "opponent": f"team {opponent_id}",
        "is_home": home,
        "points_for": pf,
        "points_against": pa,
        "EPA_penalty": -0.5,
        "EPA_per_play": epa,
        # EPA_per_play is over scrimmage plays; EPA_plays adds special teams (E10)
        "EPA_plays": plays + 11,
        "scrimmage_plays": plays,
    }


def _situational(
    rows: list[tuple[int, int | str, float]], *, legacy=False
) -> pl.DataFrame:
    """adv_situational: ``pos_team_id`` + name in ``pos_team`` (current), or the
    id itself in ``pos_team`` as text and no ``pos_team_id`` (older assets)."""
    ids = (
        {"pos_team": pl.Utf8}
        if legacy
        else {"pos_team_id": pl.Int64, "pos_team": pl.Utf8}
    )
    data = [
        (
            {"pos_team": str(t)}
            if legacy
            else {"pos_team_id": t, "pos_team": f"team {t}"}
        )
        | {"EPA_success_rate": r, "game_id": g, "season": SEASON}
        for g, t, r in rows
    ]
    return pl.DataFrame(
        data,
        schema=ids
        | {"EPA_success_rate": pl.Float64, "game_id": pl.Int64, "season": pl.Int64},
    )


def _write(
    base: Path, gamelog: pl.DataFrame | None, situational: pl.DataFrame | None
) -> str:
    for name, df in (("adv_team_gamelog", gamelog), ("adv_situational", situational)):
        if df is not None:
            p = base / name / "parquet"
            p.mkdir(parents=True)
            df.write_parquet(p / f"{name}_{SEASON}.parquet")
    return str(base)


def test_one_row_carries_the_gamelog_points_and_epa_and_the_situational_success_rate(
    tmp_path,
):
    base = _write(
        tmp_path,
        _gamelog([_row(1, 333, 61, 0.21, 70, 24, 17)]),
        # the opponent's situational row must not fan the team's row out
        _situational([(1, 333, 0.47), (1, 61, 0.39)]),
    )
    out = build_team_opponent_splits(SEASON, base=base)
    assert out.height == 1
    r = out.row(0, named=True)
    assert (r["team_id"], r["opponent_id"], r["game_id"]) == (333, 61, 1)
    assert r["epa_per_play"] == pytest.approx(0.21)
    assert r["success_rate"] == pytest.approx(0.47)
    assert (r["points_for"], r["points_against"], r["plays"]) == (24, 17, 70)
    assert r["opponent"] == "team 61" and r["is_home"] is True


def test_plays_is_the_scrimmage_count_epa_per_play_is_over(tmp_path):
    """E10: ``plays`` was ``EPA_plays`` (special teams too, median 77) beside an
    ``epa_per_play`` averaged over the scrimmage plays (66)."""
    base = _write(
        tmp_path,
        _gamelog([_row(1, 333, 61, 0.21, 66, 24, 17)]),
        _situational([(1, 333, 0.47)]),
    )
    out = build_team_opponent_splits(SEASON, base=base)
    assert out["plays"].to_list() == [66]  # not 77


def test_a_gamelog_row_with_no_situational_match_keeps_a_null_success_rate(tmp_path):
    # game 2 has situational rows (the opponent's), just none for team 333
    base = _write(
        tmp_path,
        _gamelog(
            [
                _row(1, 333, 61, 0.21, 70, 24, 17),
                _row(2, 333, 2, 0.05, 64, 10, 13, home=False),
            ]
        ),
        _situational([(1, 333, 0.47), (2, 2, 0.52)]),
    )
    out = build_team_opponent_splits(SEASON, base=base).sort("game_id")
    assert out["game_id"].to_list() == [1, 2]  # left join: game 2 is not dropped
    assert out["success_rate"].to_list()[1] is None
    assert out["epa_per_play"].to_list()[1] == pytest.approx(0.05)


def test_an_older_situational_asset_with_the_id_in_pos_team_resolves_like_build_gamelog(
    tmp_path,
):
    base = _write(
        tmp_path,
        _gamelog([_row(1, 333, 61, 0.21, 70, 24, 17)]),
        _situational([(1, 333, 0.47), (1, 61, 0.39)], legacy=True),
    )
    out = build_team_opponent_splits(SEASON, base=base)
    assert out["success_rate"].to_list() == [pytest.approx(0.47)]


def test_a_game_with_no_situational_rows_at_all_raises(tmp_path):
    # a stale adv_situational (its stage failed this run) lacks the newest
    # games entirely; publishing would null their success_rate silently
    base = _write(
        tmp_path,
        _gamelog(
            [
                _row(1, 333, 61, 0.21, 70, 24, 17),
                _row(2, 333, 2, 0.05, 64, 10, 13, home=False),
            ]
        ),
        _situational([(1, 333, 0.47), (1, 61, 0.39)]),
    )
    with pytest.raises(ValueError, match="1 games have no situational rows"):
        build_team_opponent_splits(SEASON, base=base)


def test_duplicate_situational_keys_raise_instead_of_fanning_out(tmp_path):
    base = _write(
        tmp_path,
        _gamelog([_row(1, 333, 61, 0.21, 70, 24, 17)]),
        _situational([(1, 333, 0.47), (1, 333, 0.51)]),
    )
    with pytest.raises(pl.exceptions.ComputeError):
        build_team_opponent_splits(SEASON, base=base)


def test_a_missing_situational_parquet_raises(tmp_path):
    base = _write(tmp_path, _gamelog([_row(1, 333, 61, 0.21, 70, 24, 17)]), None)
    with pytest.raises(FileNotFoundError):
        build_team_opponent_splits(SEASON, base=base)


def test_ids_are_cast_to_int64_on_both_sides_before_the_join(tmp_path):
    # gamelog ids narrower than Int64 (the schedule master ships Int32 game_id)
    # against a legacy text pos_team: the build casts both sides, then asserts
    # the key dtypes agree before joining.
    base = _write(
        tmp_path,
        _gamelog([_row(1, 333, 61, 0.21, 70, 24, 17)], id_dtype=pl.Int32),
        _situational([(1, 333, 0.47)], legacy=True),
    )
    out = build_team_opponent_splits(SEASON, base=base)
    assert out.schema == pl.Schema(TEAM_OPPONENT_SPLITS_SCHEMA)
    assert out["success_rate"].to_list() == [pytest.approx(0.47)]


def test_a_season_with_no_gamelog_is_an_empty_frame_with_the_output_schema(tmp_path):
    base = _write(tmp_path, None, _situational([(1, 333, 0.47)]))
    out = build_team_opponent_splits(SEASON, base=base)
    assert out.height == 0
    assert out.schema == pl.Schema(TEAM_OPPONENT_SPLITS_SCHEMA)


def test_the_contract_dtypes():
    s = TEAM_OPPONENT_SPLITS_SCHEMA
    for c in (
        "season",
        "team_id",
        "opponent_id",
        "game_id",
        "points_for",
        "points_against",
        "plays",
        "week",
    ):
        assert s[c] == pl.Int64, c
    assert s["epa_per_play"] == s["success_rate"] == pl.Float64
    assert s["is_home"] == pl.Boolean and s["opponent"] == pl.Utf8


def test_registered_as_a_derived_dataset_that_runs_after_gamelog():
    assert SPECS["team_opponent_splits"].tag == "cfb_team_opponent_splits"
    assert SPECS["team_opponent_splits"].stem == "cfb_team_opponent_splits"
    assert BUILDERS["team_opponent_splits"] is build_team_opponent_splits
    assert PKG_FUNCTION["cfb_team_opponent_splits"].endswith(".py")
    assert DERIVED.index("team_opponent_splits") > DERIVED.index("gamelog")
    # the daily driver runs PY_DERIVED in order; the splits read this run's gamelog
    sh = (REPO / "scripts" / "daily_cfb_processor.sh").read_text()
    order = re.search(r'^PY_DERIVED="([^"]*)"', sh, re.M).group(1).split()
    assert order.index("team_opponent_splits") > order.index("gamelog")
