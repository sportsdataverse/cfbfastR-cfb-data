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
    # games is games coached: never smaller than the record
    assert (roster["games"] >= roster["wins"] + roster["losses"]).all()
    bad = tmp_path / "bad.csv"
    roster.head(2).with_columns(games=pl.lit(1)).write_csv(bad)
    with pytest.raises(ValueError, match="fewer games than wins"):
        coaches_mod.load_coach_seasons(bad)
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

    # the live shape: camelCase names, conference on the season entry, games not yet counted
    def live_entry(first, last, school):
        season = {
            "school": school,
            "conference": "C",
            "year": 2026,
            "games": 0,
            "wins": 0,
            "losses": 0,
        }
        return {"firstName": first, "lastName": last, "seasons": [season]}

    live = [
        live_entry("Sole", "Coach", "Gamma"),
        live_entry("Shared", "One", "Delta"),
        live_entry("Shared", "Two", "Delta"),
    ]
    lrows = coaches_mod.coach_rows_from_cfbd(live, 2026)
    assert (
        lrows.filter(pl.col("school") == "Gamma").row(0, named=True)[
            "clean_attribution"
        ]
        is True
    )
    assert lrows.filter(pl.col("school") == "Gamma")["conference"].to_list() == ["C"]
    assert lrows.filter(pl.col("school") == "Delta")["clean_attribution"].to_list() == [
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


def test_cfbd_games_below_the_record_take_the_record_as_the_floor():
    """CFBD ships Todd Berry, UL Monroe 2015 as games=12 with a 2-11 record.

    `games` is games coached, so a row like that fails load_coach_seasons and
    takes coach_tendencies down for every season refreshed afterwards.
    """
    payload = [
        {
            "firstName": "Todd",
            "lastName": "Berry",
            "seasons": [
                {
                    "school": "UL Monroe",
                    "year": 2015,
                    "games": 12,
                    "wins": 2,
                    "losses": 11,
                }
            ],
        }
    ]
    row = coaches_mod.coach_rows_from_cfbd(payload, 2015).row(0, named=True)
    assert row["games"] == 13 and (row["wins"], row["losses"]) == (2, 11)

    # an in-progress season reports 0-0 and must stay 0, not become a played game
    live = [
        {
            "firstName": "In",
            "lastName": "Progress",
            "seasons": [
                {"school": "Alpha", "year": 2026, "games": 0, "wins": 0, "losses": 0}
            ],
        }
    ]
    assert coaches_mod.coach_rows_from_cfbd(live, 2026).row(0, named=True)["games"] == 0


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


# --- game context (ctx_* / def_ctx_*) --------------------------------------------
# Team 194's three games, the fixture final played three times: A home win 24-21
# (its opener), B away loss 14 days later, C at a neutral site 7 days after that
# against a home_rank=5 opponent. Every game is against 2006.
GAME_A, GAME_B, GAME_C = 401628455, 900000002, 900000003


def _schedule(**cols) -> pl.DataFrame:
    base = {
        "game_id": [GAME_A, GAME_B, GAME_C],
        "season": [2024, 2024, 2024],
        "season_type_id": [2, 2, 2],
        "start_date": [
            "2024-08-31T16:00:00.000Z",
            "2024-09-14T16:00:00.000Z",
            "2024-09-21T16:00Z",  # ESPN-only rows carry the coarser string
        ],
        "status": ["STATUS_FINAL", None, "STATUS_FINAL"],
        "neutral_site": [False, False, True],
        "home_id": [194, 2006, 2006],
        "away_id": [2006, 194, 194],
        "home_points": [24, 30, 31],
        "away_points": [21, 10, 10],
        "home_rank": [None, None, 5],
        "away_rank": [None, None, None],
    }
    return pl.DataFrame({**base, **cols}).with_columns(
        pl.col("home_rank", "away_rank").cast(pl.Int64)
    )


@pytest.fixture(scope="module")
def three_games(plays) -> pl.DataFrame:
    return pl.concat(
        [
            plays.with_columns(game_id=pl.lit(g, pl.Int64))
            for g in (GAME_A, GAME_B, GAME_C)
        ]
    )


def _true(df: pl.DataFrame, game: int, team: int, side: str = "off") -> set[str]:
    """The context flags true for ``team`` in ``game``: its offense snaps' ``ctx_*``,
    or (``side="def"``) its defensive snaps' ``def_ctx_*``."""
    key, prefix = (
        ("pos_team", "ctx_") if side == "off" else ("def_pos_team", "def_ctx_")
    )
    rows = df.filter((pl.col("game_id") == game) & (pl.col(key) == team)).select(
        pl.col(f"^{prefix}.*$")
    )
    assert rows.height and rows.unique().height == 1, (game, team, side)
    return {c.removeprefix(prefix) for c, v in rows.row(0, named=True).items() if v}


def test_attach_context_flags_each_game(three_games):
    ctx = tend.attach_context(three_games, _schedule())
    assert ctx.height == three_games.height
    assert all(dt == pl.Boolean for c, dt in ctx.schema.items() if "ctx_" in c) and {
        "ctx_home",
        "def_ctx_vs_ranked",
    } <= set(ctx.columns)
    assert _true(ctx, GAME_A, 194) == {"home", "one_score_game", "win", "opener"}
    # 14 days after A; A itself is NOT after a bye: the season's first game has
    # no previous game in the season
    assert _true(ctx, GAME_B, 194) == {"away", "after_bye"}
    assert _true(ctx, GAME_C, 194) == {"neutral_site", "vs_ranked"}
    # the opponent's own perspective
    assert _true(ctx, GAME_A, 2006) == {"away", "one_score_game", "opener"}
    assert _true(ctx, GAME_B, 2006) == {"home", "after_bye", "win"}
    assert _true(ctx, GAME_C, 2006) == {"neutral_site", "win"}
    # the defense twins are the DEFENDING team's context, i.e. the offense's opponent
    for g in (GAME_A, GAME_B, GAME_C):
        for team in (194, 2006):
            assert _true(ctx, g, team, "def") == _true(ctx, g, team), (g, team)


def test_attach_context_keys_are_int64_on_both_sides(three_games):
    # the raw master's shape: ids as String. The boundary casts them to Int64, so
    # the join still matches instead of silently matching nothing
    stringly = _schedule().with_columns(
        pl.col("game_id", "home_id", "away_id").cast(pl.Utf8)
    )
    ctx = tend.attach_context(three_games, stringly)
    assert {ctx.schema[k] for k in ("game_id", "pos_team", "def_pos_team")} == {
        pl.Int64
    }
    assert _true(ctx, GAME_C, 194) == {"neutral_site", "vs_ranked"}
    with pytest.raises(pl.exceptions.InvalidOperationError):
        tend.attach_context(three_games, _schedule(home_id=["194x", "2006", "2006"]))


def test_the_first_game_is_never_after_a_bye_and_the_opener_is_regular_season():
    # teams 9998 / 9999 play once, four weeks after every other game: that game
    # has no previous game in the season, so it is not after a bye
    extra = (
        _schedule()
        .head(1)
        .with_columns(
            game_id=pl.lit(900000004, pl.Int64),
            start_date=pl.lit("2024-10-19T16:00:00.000Z"),
            home_id=pl.lit(9998, pl.Int64),
            away_id=pl.lit(9999, pl.Int64),
        )
    )
    first = tend.game_context(pl.concat([_schedule(), extra])).filter(
        pl.col("game_id") == 900000004
    )
    assert first["after_bye"].to_list() == [False, False]
    assert first["opener"].to_list() == [True, True]
    # a postseason game is never the opener, even as a team's first game
    ctx = tend.game_context(_schedule(season_type_id=[3, 2, 2]))
    assert ctx.filter(pl.col("opener"))["game_id"].unique().to_list() == [GAME_B]


def test_a_tie_is_not_a_win(three_games):
    ctx = tend.attach_context(three_games, _schedule(home_points=[21, 30, 31]))
    assert _true(ctx, GAME_A, 194) == {"home", "one_score_game", "opener"}
    assert _true(ctx, GAME_A, 2006) == {"away", "one_score_game", "opener"}


def test_a_cancelled_game_is_not_a_previous_game(three_games):
    # a game called off between A and B does not shorten B's rest
    sched = pl.concat(
        [
            _schedule(),
            _schedule()
            .head(1)
            .with_columns(
                game_id=pl.lit(900000009, pl.Int64),
                start_date=pl.lit("2024-09-07T16:00:00.000Z"),
                status=pl.lit("STATUS_CANCELED"),
            ),
        ]
    )
    assert _true(tend.attach_context(three_games, sched), GAME_B, 194) == {
        "away",
        "after_bye",
    }


def test_schedule_without_ranks_emits_no_vs_ranked(three_games):
    for sched in (
        _schedule().drop("home_rank", "away_rank"),  # a pre-IF-0 release
        _schedule(home_rank=[None, None, None]),  # ranks unknown for the season
    ):
        ctx = tend.attach_context(three_games, sched)
        assert "ctx_vs_ranked" not in ctx.columns
        assert "def_ctx_vs_ranked" not in ctx.columns
        assert "ctx_home" in ctx.columns
        t = tend.team_tendencies(three_games, sched)
        assert "games_vs_ranked" not in t.columns and "games_home" in t.columns


def test_team_and_coach_tendencies_carry_game_context(three_games):
    t = tend.team_tendencies(three_games, _schedule())
    osu = t.filter(pl.col("pos_team") == 194).row(0, named=True)
    assert osu["games"] == 3
    assert (
        osu["games_home"] + osu["games_away"] + osu["games_neutral_site"]
        == osu["games"]
    )
    assert (osu["games_home"], osu["wins_home"], osu["win_rate_home"]) == (1, 1, 1.0)
    assert (osu["games_vs_ranked"], osu["wins_vs_ranked"]) == (1, 0)
    assert (
        osu["games_after_bye"],
        osu["games_opener"],
        osu["games_one_score_game"],
    ) == (1, 1, 1)
    # the defense's own perspective: 194 defended at home in A and won it
    assert (
        osu["def_games_home"],
        osu["def_wins_home"],
        osu["def_games_vs_ranked"],
    ) == (1, 1, 1)
    akron = t.filter(pl.col("pos_team") == 2006).row(0, named=True)
    assert (akron["games_home"], akron["wins_home"], akron["games_vs_ranked"]) == (
        1,
        1,
        0,
    )
    # without a schedule nothing changes and no context column appears
    assert not any(
        "ctx" in c or c.startswith("games_")
        for c in tend.team_tendencies(three_games).columns
    )
    coaches = pl.DataFrame(
        {
            "season": [2024, 2024],
            "team_id": [194, 2006],
            "coach": ["Coach Home", "Coach Away"],
        },
        schema=coaches_mod.TEAM_COACH_SCHEMA,
    )
    c = tend.coach_tendencies(three_games, coaches, _schedule())
    home_c = c.filter(pl.col("coach") == "Coach Home").row(0, named=True)
    for k in (
        "games_home",
        "wins_home",
        "games_vs_ranked",
        "def_games_home",
        "games_after_bye",
    ):
        assert home_c[k] == osu[k], k


def test_build_season_joins_the_season_schedule(tmp_path, monkeypatch):
    """build_season reads ``{base}/cfb_schedules`` for the tendencies specs."""
    from cfb_data_build import build as build_mod

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{GAME_A}.json").write_text(
        FIX.read_text(encoding="utf-8"), encoding="utf-8"
    )
    master = tmp_path / "master.parquet"
    pl.DataFrame(
        {
            "game_id": [GAME_A],
            "season": [2024],
            "home_id": ["194"],
            "home_display_name": ["Ohio State Buckeyes"],
            "away_id": ["2006"],
            "away_display_name": ["Akron Zips"],
        }
    ).write_parquet(master)
    base = tmp_path / "cfb"
    sched_dir = base / "cfb_schedules" / "parquet"
    sched_dir.mkdir(parents=True)
    _schedule().head(1).write_parquet(sched_dir / "cfb_schedules_2024.parquet")
    df = build_mod.build_season(
        REGISTRY["team_tendencies"],
        2024,
        cache_dir=cache,
        schedule=master,
        fetch=False,
        base=base,
    )
    osu = df.filter(pl.col("pos_team_id") == 194).row(0, named=True)
    assert (osu["games_home"], osu["wins_home"], osu["games_opener"]) == (1, 1, 1)
    # no schedule anywhere: the build still writes, without context columns
    monkeypatch.setattr(build_mod, "_release_schedule", lambda season: None)
    df = build_mod.build_season(
        REGISTRY["team_tendencies"],
        2024,
        cache_dir=cache,
        schedule=master,
        fetch=False,
        base=tmp_path / "bare",
    )
    assert df.height == 2 and "games_home" not in df.columns
