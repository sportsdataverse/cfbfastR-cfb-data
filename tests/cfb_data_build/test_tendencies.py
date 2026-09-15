"""Team / coach tendencies and the coach roster, offline over the fixture final."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from cfb_data_build import coaches as coaches_mod
from cfb_data_build import tendencies as tend
from cfb_data_build.build import build_careers, build_dataset_frame
from cfb_data_build.config import PKG_FUNCTION, REGISTRY, TENDENCIES_ORDER
from cfb_data_build.io import dataset_stem, write_dataset

FIX = Path(__file__).parent / "fixtures" / "final_401628455.json"


@pytest.fixture(scope="module")
def plays() -> pl.DataFrame:
    game = json.loads(FIX.read_text(encoding="utf-8"))
    return build_dataset_frame(REGISTRY["team_tendencies"], game)


def test_registry_and_sidecars_cover_the_tendencies_datasets():
    for key in TENDENCIES_ORDER:
        spec = REGISTRY[key]
        assert spec.tendencies is not None and spec.tag in PKG_FUNCTION, key
    assert dataset_stem("coach_careers", None) == "coach_careers"
    assert dataset_stem("coach_tendencies", 2024) == "coach_tendencies_2024"


def test_team_tendencies_from_the_fixture_final(plays):
    assert plays.height > 100 and {
        "pos_team",
        "def_pos_team",
        "season",
        "seasonType",
    } <= set(plays.columns)
    t = tend.team_tendencies(plays)
    assert t.height == 2 and set(t["pos_team"].to_list()) == {194, 2006}
    assert {
        "games",
        "plays",
        "pass_rate_neutral",
        "sec_per_play",
        "go_rate",
        "rz_td_rate",
        "def_epa_per_play",
    } <= set(t.columns)
    assert (t["games"] == 1).all()
    # preseason snaps never count
    assert tend.team_tendencies(plays.with_columns(seasonType=pl.lit(1))).height == 0


def test_coach_attribution_and_careers(plays, tmp_path):
    coaches = pl.DataFrame(
        {
            "season": [2024, 2024],
            "team_id": [194, 2006],
            "coach": ["Coach Home", "Coach Away"],
        },
        schema=coaches_mod.TEAM_COACH_SCHEMA,
    )
    c = tend.coach_tendencies(plays, coaches)
    assert (
        c.columns[:4] == ["season", "pos_team", "coach", "role"]
        and (c["role"] == "HC").all()
    )
    assert set(c["coach"].to_list()) == {"Coach Home", "Coach Away"}
    t = tend.team_tendencies(plays)
    home_t = t.filter(pl.col("pos_team") == 194).row(0, named=True)
    home_c = c.filter(pl.col("coach") == "Coach Home").row(0, named=True)
    assert (
        home_c["plays"] == home_t["plays"]
        and abs(home_c["pass_rate"] - home_t["pass_rate"]) < 1e-12
    )
    # only the home side attributed: its row still equals the full team-season
    # (every snap it ran and every snap it faced); the other side forms no row
    partial = tend.coach_tendencies(plays, coaches.head(1))
    assert partial["coach"].to_list() == ["Coach Home"]
    p = partial.row(0, named=True)
    assert p["plays"] == home_t["plays"] and p["def_plays"] == home_t["def_plays"]
    assert abs(p["def_success_rate"] - home_t["def_success_rate"]) < 1e-12
    assert tend.coach_tendencies(plays, coaches.head(0)).height == 0
    # a season with no games binds to a frame with NO columns; still no rows
    bare = tend.attach_coaches(pl.DataFrame(), coaches)
    assert bare.height == 0 and {"coach", "def_coach"} <= set(bare.columns)
    assert tend.coach_tendencies(pl.DataFrame(), coaches).height == 0
    with pytest.raises(RuntimeError, match="cfb_data_build.coaches -s 2024"):
        tend.require_coaches(coaches.head(0), 2024)
    # careers: two written seasons of the same coach double the counts and keep the rates
    write_dataset(c, "coach_tendencies", 2024, "coach_tendencies", base=tmp_path)
    write_dataset(
        c.with_columns(season=pl.lit(2023)),
        "coach_tendencies",
        2023,
        "coach_tendencies",
        base=tmp_path,
    )
    careers = build_careers(base=tmp_path)
    assert (tmp_path / "coach_careers" / "parquet" / "coach_careers.parquet").exists()
    assert careers.columns[:6] == [
        "coach",
        "role",
        "teams",
        "seasons",
        "first_season",
        "last_season",
    ]
    row = careers.filter(pl.col("coach") == "Coach Home").row(0, named=True)
    assert (
        row["seasons"] == 2
        and row["plays"] == 2 * home_c["plays"]
        and abs(row["pass_rate"] - home_c["pass_rate"]) < 1e-12
    )
    assert row["teams"] == "194"
    assert tend.coach_careers([]).height == 0


def test_vendored_roster_and_school_ids(tmp_path):
    roster = coaches_mod.load_coach_seasons()
    assert roster.height > 2500 and roster.schema == coaches_mod.COACH_SCHEMA
    assert roster.filter(pl.col("season") == 2024).height > 100
    # school -> id through a schedule master shaped like the real one
    sched = pl.DataFrame(
        {
            "season": [2024, 2024],
            "home_id": ["194", "2006"],
            "home_location": ["Ohio State", "Akron"],
            "away_id": ["2006", "194"],
            "away_location": ["Akron", "Ohio State"],
        }
    )
    path = tmp_path / "sched.parquet"
    sched.write_parquet(path)
    ids = coaches_mod.school_team_ids(path, 2024)
    assert ids.schema == {"team_id": pl.Int64, "school": pl.Utf8} and ids.height == 2
    tc = coaches_mod.team_coaches(2024, path, roster)
    assert tc.schema == coaches_mod.TEAM_COACH_SCHEMA
    assert tc.filter(pl.col("team_id") == 194)["coach"].to_list() == ["Ryan Day"]
    assert coaches_mod.team_coaches(1999, path, roster).height == 0


def test_cfbd_rows_and_refresh(tmp_path):
    payload = [
        {
            "first_name": "Full",
            "last_name": "Season",
            "seasons": [
                {"school": "Alpha", "year": 2026, "games": 12, "wins": 8, "losses": 4}
            ],
        },
        {
            "first_name": "Fired",
            "last_name": "Early",
            "seasons": [
                {"school": "Beta", "year": 2026, "games": 5, "wins": 1, "losses": 4}
            ],
        },
        {
            "first_name": "Interim",
            "last_name": "Guy",
            "seasons": [
                {"school": "Beta", "year": 2026, "games": 7, "wins": 3, "losses": 4}
            ],
        },
        {
            "first_name": "Other",
            "last_name": "Year",
            "seasons": [
                {"school": "Alpha", "year": 2025, "games": 12, "wins": 6, "losses": 6}
            ],
        },
    ]
    rows = coaches_mod.coach_rows_from_cfbd(payload, 2026)
    assert rows.schema == coaches_mod.COACH_SCHEMA and rows.height == 3
    assert rows.filter(pl.col("school") == "Alpha")["clean_attribution"].to_list() == [
        True
    ]
    assert rows.filter(pl.col("school") == "Beta")["clean_attribution"].to_list() == [
        False,
        False,
    ]
    csv = tmp_path / "roster.csv"
    pl.DataFrame(
        {
            "season": [2025],
            "coach": ["Kept Coach"],
            "school": ["Gamma"],
            "conference": ["C"],
            "games": [12],
            "wins": [9],
            "losses": [3],
            "clean_attribution": [True],
        }
    ).write_csv(csv)
    out = coaches_mod.refresh_coach_seasons(
        [2026],
        path=csv,
        api_key="not-a-real-key",
        fetch=lambda s, api_key=None: payload,
    )
    assert out.filter(pl.col("season") == 2025)["coach"].to_list() == ["Kept Coach"]
    assert out.filter(pl.col("season") == 2026).height == 3
    assert coaches_mod.load_coach_seasons(csv).schema == coaches_mod.COACH_SCHEMA
    assert coaches_mod.load_coach_seasons(tmp_path / "missing.csv").height == 0
    # an empty answer for a season keeps its existing rows instead of erasing them
    again = coaches_mod.refresh_coach_seasons(
        [2026], path=csv, api_key="k", fetch=lambda s, api_key=None: []
    )
    assert again.filter(pl.col("season") == 2026).height == 3


def test_cfbd_error_envelope_is_an_error_not_an_empty_season(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"error": "rate limited"}'

    monkeypatch.setattr(
        coaches_mod.urllib.request, "urlopen", lambda req, timeout=120: _Resp()
    )
    with pytest.raises(TypeError, match="unexpected payload shape"):
        coaches_mod.fetch_cfbd_coaches(2026, api_key="k")
