"""defense_vs_position (stage 66) over real released 2024 rows.

Provenance and the hand counts behind every literal here:
tests/fixtures/defense_vs_position/README.md. The builder reads three parquet
files under ``base`` and has no release-loader fallback, so nothing here can
reach the network; ``publish_dataset`` is stubbed to fail for the whole module.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sportsdataverse.defense_vs_position import OUTPUT_SCHEMA as T1_SCHEMA
from sportsdataverse.defense_vs_position import PBP_COLUMNS, defense_vs_position

from cfb_data_build.cli import DERIVED
from cfb_data_build.config import PKG_FUNCTION
from cfb_data_build.defense_vs_position import (
    FLOOR,
    OUTPUT_SCHEMA,
    PCT_METRICS,
    build_defense_vs_position,
    defense_vs_position_table,
)
from cfb_data_build.derived import BUILDERS, SPECS, build_derived
from cfb_data_build.publish import RELEASE_NOTES
from cfb_data_build.team_summaries import MIN_COHORT_TEAMS

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "defense_vs_position"
SEASON = 2024
PSU, UGA, OSU, TEX, ND, ORE, NDSU = 213, 61, 194, 251, 87, 2483, 2449
FBS_QUALIFIERS = (PSU, OSU, TEX, ND, ORE)  # three games each; Georgia has two
PCT_COLS = [f"{m}_pct" for m in PCT_METRICS]
#: base sub-directory -> file stem, as the builder reads them
_INPUTS = {
    "pbp": "play_by_play",
    "cfb_rosters": "cfb_rosters",
    "cfb_schedules": "cfb_schedules",
}


@pytest.fixture(autouse=True)
def _no_publish(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a defense_vs_position test reached publish_dataset")

    monkeypatch.setattr("cfb_data_build.publish.publish_dataset", refuse)


def _slice(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIX / f"{_INPUTS[name]}_2024_slice.parquet")


def _base(root: Path, season: int = SEASON, **frames: pl.DataFrame) -> str:
    """A ``base`` tree holding the fixture slices (or ``frames``) for ``season``."""
    for name, stem in _INPUTS.items():
        d = root / name / "parquet"
        d.mkdir(parents=True)
        frames.get(name, _slice(name)).write_parquet(d / f"{stem}_{season}.parquet")
    return str(root)


@pytest.fixture(scope="module")
def dvp(tmp_path_factory) -> pl.DataFrame:
    return build_defense_vs_position(SEASON, base=_base(tmp_path_factory.mktemp("cfb")))


def _cell(df: pl.DataFrame, team: int, group: str) -> dict:
    rows = df.filter(
        (pl.col("team_id") == team) & (pl.col("position_group") == group)
    ).to_dicts()
    assert len(rows) == 1, rows
    return rows[0]


def test_schema_keys_and_int_team_id(dvp):
    assert dvp.schema == pl.Schema(OUTPUT_SCHEMA)
    assert dvp.schema["team_id"] == pl.Int64
    assert dvp.select("season", "team_id", "position_group").is_duplicated().sum() == 0
    assert set(dvp["team_id"]) == {PSU, UGA, OSU, TEX, ND, ORE, NDSU}
    assert set(dvp["position_group"]) == {"QB", "RB", "WR", "TE"}


def test_t1_columns_pass_through_unchanged(dvp):
    t1 = defense_vs_position(
        _slice("pbp").select(PBP_COLUMNS["cfb"]), _slice("cfb_rosters"), "cfb"
    ).with_columns(pl.col("team_id").cast(pl.Int64))
    keys = ["season", "team_id", "position_group"]
    assert_frame_equal(dvp.select(list(T1_SCHEMA)), t1.sort(keys))


def test_penn_state_te_cell_is_t1s_hand_count(dvp):
    c = _cell(dvp, PSU, "TE")
    assert (c["plays"], c["games"], c["targets"]) == (14, 2, 14)
    assert c["epa_per_play_allowed"] == pytest.approx(12.263268947601318 / 14)
    assert c["epa_per_play_allowed"] == pytest.approx(0.875948, abs=1e-6)
    assert c["qualified"] is False


def test_team_display_columns_are_the_summaries_names(dvp):
    psu = _cell(dvp, PSU, "QB")
    assert (psu["pos_team"], psu["division"], psu["conference"]) == (
        "Penn State",
        "fbs",
        "Big Ten",
    )
    ndsu = _cell(dvp, NDSU, "QB")
    assert (ndsu["pos_team"], ndsu["division"]) == ("North Dakota State", "fcs")


def test_pct_lower_allowed_is_the_better_defense(dvp):
    """RB EPA/play allowed, 5 FBS qualifiers: Weibull 100 * (n + 1 - rank) / (n + 1)."""
    hand = {  # EPA sum / plays, lowest allowed first
        OSU: -11.423318937420845 / 63,
        TEX: -10.408038347959518 / 91,
        PSU: -7.0440216436982155 / 75,
        ND: 2.9307474493980408 / 92,
        ORE: 13.624835595488548 / 75,
    }
    for rank, (team, epa) in enumerate(hand.items(), start=1):
        c = _cell(dvp, team, "RB")
        assert c["epa_per_play_allowed"] == pytest.approx(epa)
        assert c["epa_per_play_allowed_pct"] == pytest.approx(100 * (6 - rank) / 6)


def test_pct_sack_rate_is_the_reverse_direction(dvp):
    """QB sack rate: the defense that sacks MORE is better, so highest ranks first."""
    hand = {OSU: 12 / 82, ORE: 7 / 108, ND: 5 / 79, PSU: 4 / 85, TEX: 4 / 91}
    for rank, (team, rate) in enumerate(hand.items(), start=1):
        c = _cell(dvp, team, "QB")
        assert c["sack_rate_allowed"] == pytest.approx(rate)
        assert c["sack_rate_allowed_pct"] == pytest.approx(100 * (6 - rank) / 6)


def test_non_qualifier_has_null_pct_and_counts_in_no_cohort(dvp):
    out = dvp.filter(pl.col("qualified") == False)  # noqa: E712
    assert set(out["team_id"]) >= {UGA, PSU}  # Georgia: 2 games; Penn State TE: 2
    assert out.select(pl.all_horizontal(pl.col(PCT_COLS).is_null()).all()).item()
    # Georgia's RB EPA/play (-0.1332) would rank 2nd of 6 and push Texas to 3rd
    assert _cell(dvp, UGA, "RB")["epa_per_play_allowed"] == pytest.approx(
        -6.262705460190773 / 47
    )
    assert _cell(dvp, TEX, "RB")["epa_per_play_allowed_pct"] == pytest.approx(400 / 6)


def test_non_fbs_qualifier_has_null_pct(dvp):
    ndsu = dvp.filter(pl.col("team_id") == NDSU)
    assert ndsu.filter(pl.col("qualified") == True).height >= 3  # noqa: E712
    assert ndsu.select(pl.all_horizontal(pl.col(PCT_COLS).is_null()).all()).item()


def test_cohort_under_the_f5_floor_has_null_pct(dvp):
    te = dvp.filter(
        (pl.col("position_group") == "TE")
        & (pl.col("division") == "fbs")
        & (pl.col("qualified") == True)  # noqa: E712
    )
    assert 0 < te.height < MIN_COHORT_TEAMS == len(FBS_QUALIFIERS)
    assert te["epa_per_play_allowed_pct"].null_count() == te.height


def test_extras_only_get_a_pct_on_their_own_group(dvp):
    fbs = dvp.filter(pl.col("team_id").is_in(FBS_QUALIFIERS))
    for group in ("RB", "WR"):
        assert fbs.filter(pl.col("position_group") == group)[
            "sack_rate_allowed_pct"
        ].null_count() == len(FBS_QUALIFIERS)
    rb = fbs.filter(pl.col("position_group") == "RB")
    assert rb["rush_yards_per_carry_allowed_pct"].null_count() == 0
    # Ohio State 143 / 52 = 2.75 yards per carry, the lowest of the five
    assert _cell(dvp, OSU, "RB")["rush_yards_per_carry_allowed_pct"] == pytest.approx(
        500 / 6
    )


def test_unattributed_target_share_on_a_hand_counted_game():
    """Georgia vs Clemson (401628323): 29 throws, 3 name no receiver (plays 20, 131, 149)."""
    game = pl.col("game_id") == 401628323
    out = defense_vs_position_table(
        _slice("pbp").filter(game), _slice("cfb_rosters"), _slice("cfb_schedules")
    )
    for group in ("WR", "TE"):
        assert _cell(out, UGA, group)["unattributed_target_share"] == pytest.approx(
            3 / 29
        )
    for group in ("QB", "RB"):
        assert _cell(out, UGA, group)["unattributed_target_share"] is None


def test_unattributed_target_share_is_per_defense_season(dvp):
    assert _cell(dvp, PSU, "WR")["unattributed_target_share"] == pytest.approx(40 / 81)
    assert _cell(dvp, PSU, "TE")["unattributed_target_share"] == pytest.approx(40 / 81)
    assert _cell(dvp, UGA, "WR")["unattributed_target_share"] == pytest.approx(6 / 36)


def test_pre_floor_season_is_the_empty_schema_and_says_why(tmp_path, capsys):
    assert FLOOR == 2014
    base = _base(tmp_path, season=2013)  # inputs present: only the floor stops it
    out = build_defense_vs_position(2013, base=base)
    assert out.height == 0
    assert out.schema == pl.Schema(OUTPUT_SCHEMA)
    assert "2014" in capsys.readouterr().out
    # and the stage driver writes nothing for it
    assert build_derived("defense_vs_position", 2013, 2013, base=base) == []
    assert not (tmp_path / "defense_vs_position").exists()


def test_missing_pbp_is_the_empty_schema(tmp_path):
    out = build_defense_vs_position(SEASON, base=str(tmp_path))
    assert out.height == 0
    assert out.schema == pl.Schema(OUTPUT_SCHEMA)


def test_pbp_of_another_season_raises(tmp_path):
    with pytest.raises(ValueError, match="2024"):
        build_defense_vs_position(2023, base=_base(tmp_path, season=2023))


@pytest.mark.parametrize(
    ("name", "col"),
    [
        ("pbp", "def_pos_team_id"),
        ("cfb_schedules", "home_id"),
        ("cfb_rosters", "athlete_id"),
    ],
)
def test_float_id_raises(tmp_path, name, col):
    floated = _slice(name).with_columns(pl.col(col).cast(pl.Float64))
    with pytest.raises(TypeError, match=col):
        build_defense_vs_position(SEASON, base=_base(tmp_path, **{name: floated}))


def test_stage_driver_writes_the_tree_layout(tmp_path):
    base = _base(tmp_path)
    assert build_derived("defense_vs_position", SEASON, SEASON, base=base) == []
    root = tmp_path / "defense_vs_position"
    written = pl.read_parquet(root / "parquet" / "cfb_defense_vs_position_2024.parquet")
    assert written.schema == pl.Schema(OUTPUT_SCHEMA)
    manifest = pl.read_csv(root / "cfb_defense_vs_position_in_data_repo.csv")
    assert manifest.select("season", "row_count").rows() == [(SEASON, written.height)]


def test_registered_as_a_derived_stage():
    assert "defense_vs_position" in DERIVED and "defense_vs_position" in BUILDERS
    spec = SPECS["defense_vs_position"]
    assert (spec.dataset, spec.stem, spec.tag) == (
        "defense_vs_position",
        "cfb_defense_vs_position",
        "cfb_defense_vs_position",
    )
    assert PKG_FUNCTION[spec.tag].endswith(
        "espn_cfb_66_defense_vs_position_creation.py"
    )
    notes = RELEASE_NOTES[spec.tag]
    assert "naming a receiver" in notes and "2014" in notes
