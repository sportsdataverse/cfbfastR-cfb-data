"""build_cfb_players_crosswalk: leg combine, ambiguous-vote pruning, team map, shrink refusal."""

from datetime import date

import build_cfb_players_crosswalk as bp
import polars as pl

GAMES = pl.DataFrame(
    {"espn_game_id": ["1", "2"], "game_date": [date(2025, 9, 6), date(2025, 9, 13)]},
)
ROSTERS = pl.DataFrame(
    {
        "espn_game_id": ["1", "2", "1", "1"],
        "espn_team_id": [202, 202, 202, 202],
        "espn_athlete_id": ["10", "10", "11", "12"],
        "espn_full_name": ["Will Smyth", "Will Smith", "Josh Smith", "Ty Lee"],
        "espn_jersey": ["1", "1", "2", "3"],
    },
    schema_overrides={"espn_team_id": pl.Int32},
)


def _leg(id_col, name_col, rows):
    return pl.DataFrame(
        rows,
        schema={
            "espn_team_id": pl.Int32,
            "espn_athlete_id": pl.Utf8,
            id_col: pl.Utf8,
            name_col: pl.Utf8,
        },
        orient="row",
    )


def test_combine_prunes_a_surname_twin_and_keeps_every_leg():
    cbs = _leg(
        "cbs_player_id",
        "cbs_player_name",
        [(202, "10", "c1", "Will Smith"), (202, "11", "c1", "Will Smith")],
    )
    yahoo = _leg("yahoo_player_id", "yahoo_player_name", [(202, "12", "y3", "Ty Lee")])
    out = bp.combine(2025, GAMES, ROSTERS, [cbs, yahoo])
    assert out.schema == pl.Schema(bp.SCHEMA)
    assert out.rows() == [
        (
            2025,
            202,
            "10",
            "Will Smith",
            "c1",
            "Will Smith",
            None,
            None,
            None,
            None,
        ),  # latest-game ESPN name
        (2025, 202, "12", "Ty Lee", None, None, None, None, "y3", "Ty Lee"),
    ]


def test_ambiguous_group_without_an_exact_name_is_left_as_voted():
    leg = _leg(
        "fox_athlete_id",
        "fox_player_name",
        [(202, "10", "f1", "Bill Smith"), (202, "11", "f1", "Bill Smith")],
    )
    names = ROSTERS.select(
        "espn_team_id",
        "espn_athlete_id",
        pl.col("espn_full_name").alias("espn_player_name"),
    )
    assert bp.drop_ambiguous(leg, names.unique(bp.KEYS)).height == 2


def test_team_map_drops_ids_listed_twice(tmp_path):
    (tmp_path / "crosswalk" / "parquet").mkdir(parents=True)
    pl.DataFrame(
        {"espn_team_id": [1, 2, 3, None], "fox_team_id": ["a", "b", "b", "z"]},
        schema={"espn_team_id": pl.Int64, "fox_team_id": pl.Utf8},
    ).write_parquet(
        tmp_path / "crosswalk" / "parquet" / "cfb_teams_crosswalk_2025.parquet"
    )
    m = bp.team_map("fox", 2025, tmp_path)
    assert m.schema == pl.Schema({"espn_team_id": pl.Int32, "team_id": pl.Utf8})
    assert m.rows() == [(1, "a")]
    assert bp.team_map("yahoo", 2024, tmp_path) is None


def test_shrunk_flags_only_a_lost_provider_leg():
    old = pl.DataFrame({"cbs_player_id": ["1", "2"], "fox_athlete_id": ["9", None]})
    new = pl.DataFrame(
        {"cbs_player_id": ["1", "2", "3"], "fox_athlete_id": [None, None, None]}
    )
    assert bp.shrunk(old, new) == {"fox_athlete_id": (1, 0)}
