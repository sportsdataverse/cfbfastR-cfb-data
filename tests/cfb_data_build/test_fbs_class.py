"""``fbs_class`` comes from SDV conference group ids, not conference names.

The fixture is a slice of the real ``cfb_groups`` release
(``cfb_team_group_seasons_{2010,2023,2024}.parquet``, fetched 2026-09-27). The
schedule conference names below are the ones the published team summaries carry.
Under the old name lists, every Pac-10 team (2004-2010) published as G5.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from cfb_data_build.team_summaries import _build_schools, _prepare_for_write

GROUPS = pl.read_parquet(
    Path(__file__).parent / "fixtures" / "cfb_team_group_seasons_fbs_class.parquet"
)

# season -> [(ESPN team id, school, schedule conference name, expected class)]
CASES = {
    2010: [
        ("2483", "Oregon", "Pac-10", "P5"),
        ("87", "Notre Dame", "FBS Independents", "P5"),
        ("97", "Louisville", "Big East", "G5"),
    ],
    2023: [
        ("264", "Washington", "Pac-12", "P5"),
        ("87", "Notre Dame", "FBS Independents", "P5"),
        ("349", "Army", "FBS Independents", "G5"),
        ("2567", "SMU", "American Athletic", "G5"),
    ],
    2024: [
        ("204", "Oregon State", "Pac-12", "G6"),
        ("265", "Washington State", "Pac-12", "G6"),
        ("2567", "SMU", "ACC", "P4"),
        ("251", "Texas", "SEC", "P4"),
        ("201", "Oklahoma", "SEC", "P4"),
        ("87", "Notre Dame", "FBS Independents", "P4"),
        ("41", "UConn", "FBS Independents", "G6"),
    ],
}


def _plays(rows: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    ids, names, confs, _ = zip(*rows)
    return pl.DataFrame(
        {
            "home_team_id": ids,
            "pos_team_id": ids,
            "pos_team": names,
            "home_team_division": ["fbs"] * len(ids),
            "away_team_division": ["fbs"] * len(ids),
            "home_team_conference": confs,
            "away_team_conference": confs,
        }
    )


@pytest.mark.parametrize("yr", sorted(CASES))
def test_fbs_class_from_group_ids(yr: int) -> None:
    rows = CASES[yr]
    schools = _build_schools(_plays(rows), yr, GROUPS)
    team = pl.DataFrame({"pos_team_id": [r[0] for r in rows], "EPAplay_off": 0.1})
    out = _prepare_for_write(team, yr, schools)
    # schema unchanged: identity first, fbs_class still the last column
    assert out.columns == [
        "team_id", "pos_team", "division", "conference", "season",
        "EPAplay_off", "fbs_class",
    ]  # fmt: skip
    got = dict(zip(out["pos_team"], out["fbs_class"]))
    assert got == {r[1]: r[3] for r in rows}


def test_team_missing_from_groups_raises() -> None:
    plays = _plays([("2483", "Oregon", "Pac-12", "P4")])
    with pytest.raises(ValueError, match="missing from cfb_groups"):
        _build_schools(plays, 2024, GROUPS)
