"""Side inputs (talent, coaches), team meta and weather vs the delivered 2025 line.

Oracle: ``fixtures/matchup/matchup_line_2025.csv`` (the source pipeline's own
2025 line). Inputs: the CFBD payloads captured 2026-09-15 by
``ops/oneoff/20260915_matchup_oracle_capture/capture_side_inputs.py`` (tidied
parquet; provenance in the fixtures README). No network: everything here runs
in the unit suite.

Bars (observed 2026-09-15, never lowered): weather exact on every game (10/10
columns at 1.000, nulls aligned); ``join_name``, ``talent``, and the identity /
venue-key meta columns exact (1.000 -- ``talent`` after the academy rule);
``head_coach`` >= 0.96 raw and >= 0.98 once initials spacing and case are
folded (observed 0.972 / see test); ``hc_tenure`` >= 0.93 (observed 0.952 /
0.947); the venue detail columns >= 0.93 each (observed 0.95-0.97: CFBD has
corrected and back-filled venue rows since the source captured them);
``alt_name2`` >= 0.99 (a curly vs straight apostrophe). ``color`` /
``alt_color`` / ``logo`` / ``logo_2`` are NOT compared: CFBD re-issued team
colours and moved its logo CDN after the capture (0-40% raw agreement, every
difference a source-version change), so they are asserted non-null only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cfb_data_build.matchup_features import to_display_names
from cfb_data_build.matchup_line import SIDE_INPUT_COLS, TEAM_META_COLS, WEATHER_COLS
from cfb_data_build.matchup_side import (
    ACADEMIES_NO_TALENT,
    _cfbd,
    _typed,
    head_coaches,
    team_side_inputs,
    tidy_cfbd_weather,
)

FIX = Path(__file__).parent / "fixtures" / "matchup"

EXACT_META = (
    "mascot", "abbreviation", "alt_name1", "alt_name3", "classification", "twitter",
    "venue_id", "venue_name", "city", "state", "dome",
)  # fmt: skip
VENUE_DETAIL = (
    "zip", "country_code", "timezone", "latitude", "longitude", "elevation",
    "capacity", "year_constructed", "grass",
)  # fmt: skip
SOURCE_VERSION_ONLY = ("color", "alt_color", "logo", "logo_2")
PROPRIETARY = tuple(
    c
    for c in SIDE_INPUT_COLS
    if c not in ("join_name", "talent", "head_coach", "hc_tenure")
)


@pytest.fixture(scope="module")
def oracle() -> pl.DataFrame:
    return pl.read_csv(
        FIX / "matchup_line_2025.csv", null_values=["NA", ""], infer_schema_length=10000
    )


@pytest.fixture(scope="module")
def coaches() -> pl.DataFrame:
    return pl.read_parquet(FIX / "cfbd_coaches_thru_2025.parquet")


@pytest.fixture(scope="module")
def side(coaches: pl.DataFrame) -> pl.DataFrame:
    return team_side_inputs(
        2025,
        talent=pl.read_parquet(FIX / "cfbd_talent_2025.parquet"),
        teams=pl.read_parquet(FIX / "cfbd_teams_2025.parquet"),
        coaches=coaches,
    )


def _match(a: pl.Series, b: pl.Series, *, atol: float = 1e-6) -> float:
    """Share of rows equal (nulls equal to nulls)."""
    if a.dtype.is_numeric() and b.dtype.is_numeric():
        x, y = a.cast(pl.Float64).to_numpy(), b.cast(pl.Float64).to_numpy()
        ok = np.isclose(x, y, atol=atol, equal_nan=True) | (np.isnan(x) & np.isnan(y))
    else:
        ok = (
            (a.cast(pl.Utf8) == b.cast(pl.Utf8))
            .fill_null(a.is_null() & b.is_null())
            .to_numpy()
        )
    return float(ok.mean())


def _fold(name: pl.Series) -> pl.Series:
    return name.str.replace_all(r"\.\s+", ".").str.to_lowercase()


def test_weather_matches_the_delivered_line_exactly(oracle: pl.DataFrame) -> None:
    weather = pl.read_parquet(FIX / "cfbd_weather_2025.parquet")
    assert weather.columns == ["game_id", *WEATHER_COLS]
    assert weather.schema["game_id"] == oracle.schema["game_id"] == pl.Int64
    assert weather["game_id"].n_unique() == weather.height
    joined = oracle.select("game_id", *WEATHER_COLS).join(
        weather, on="game_id", how="left", suffix="_py"
    )
    assert joined.height == 773
    for c in WEATHER_COLS:
        assert _match(joined[c], joined[f"{c}_py"], atol=1e-9) == 1.0, c


def test_side_inputs_and_meta_match_the_delivered_line(
    oracle: pl.DataFrame, side: pl.DataFrame
) -> None:
    assert side.columns == ["team", *SIDE_INPUT_COLS, *TEAM_META_COLS]
    mapped = to_display_names(side, "team")
    for prefix in ("home", "away"):
        key = f"{prefix}_team"
        j = oracle.select(
            key, *[f"{prefix}_{c}" for c in (*SIDE_INPUT_COLS, *TEAM_META_COLS)]
        ).join(mapped.rename({"team": key}), on=key, how="left")
        assert j.height == 773 and j["join_name"].null_count() == 0, (
            f"{prefix}: schools missing from /teams: "
            f"{j.filter(pl.col('join_name').is_null())[key].unique().to_list()}"
        )
        for c in ("join_name", "talent", *EXACT_META):
            assert _match(j[f"{prefix}_{c}"], j[c]) == 1.0, (prefix, c)
        raw = _match(j[f"{prefix}_head_coach"], j["head_coach"])
        folded = _match(_fold(j[f"{prefix}_head_coach"]), _fold(j["head_coach"]))
        assert raw >= 0.96 and folded >= 0.98, (prefix, raw, folded)
        assert _match(j[f"{prefix}_hc_tenure"], j["hc_tenure"]) >= 0.93, prefix
        for c in VENUE_DETAIL:
            assert _match(j[f"{prefix}_{c}"], j[c]) >= 0.93, (prefix, c)
        assert _match(j[f"{prefix}_alt_name2"], j["alt_name2"]) >= 0.99, prefix
        for c in SOURCE_VERSION_ONLY:
            assert j[c].null_count() == 0, (prefix, c)
        for c in PROPRIETARY:
            assert j[c].null_count() == j.height, (prefix, c)
            assert j.schema[c] == oracle.schema[f"{prefix}_{c}"] or j[c].dtype in (
                pl.Int64,
                pl.Float64,
                pl.Utf8,
            ), (prefix, c)


def test_head_coach_is_the_incumbent_and_tenure_counts_majority_seasons(
    coaches: pl.DataFrame,
) -> None:
    """Real 2025 cases from the fixture.

    Penn State and Virginia Tech changed coaches mid-2025: the interim coached
    as many or more games, but the line carries the preseason incumbent.
    Ryan Day's 2018 (3 games while the head coach was suspended) and Luke
    Fickell's 2022 bowl do not start their tenures; Chris Klieman's does in 2019.
    """
    hc = head_coaches(coaches, 2025)
    got = {r["team"]: (r["head_coach"], r["hc_tenure"]) for r in hc.to_dicts()}
    assert got["Penn State"] == ("James Franklin", 11)
    assert got["Virginia Tech"][0] == "Brent Pry"
    assert got["Ohio State"] == ("Ryan Day", 6)
    assert got["Wisconsin"] == ("Luke Fickell", 2)
    assert got["Kansas State"] == ("Chris Klieman", 6)
    assert hc["team"].n_unique() == hc.height


def test_academies_score_zero_talent_when_absent(side: pl.DataFrame) -> None:
    rows = side.filter(pl.col("team").is_in(ACADEMIES_NO_TALENT))
    assert rows.height == 2 and (rows["talent"] == 0.0).all()
    assert side.filter(pl.col("team") == "Army")["talent"].item() > 0


def test_weather_tidy_keeps_the_first_row_per_game_and_the_documented_dtypes() -> None:
    payload = [
        {"id": 1, "temperature": 70.5, "humidity": 40, "weatherCondition": "Clear"},
        {"id": 1, "temperature": 99.0, "humidity": 1, "weatherCondition": "Storm"},
        {"id": 2, "temperature": None, "humidity": 55.0},
    ]
    out = tidy_cfbd_weather(payload)
    assert out.height == 2 and out.columns == ["game_id", *WEATHER_COLS]
    assert out.row(0, named=True)["weather_condition"] == "Clear"
    assert out.schema["humidity"] == pl.Int64 and out["humidity"].to_list() == [40, 55]
    assert out.schema["snowfall"] == pl.Int64 and out["snowfall"].null_count() == 2
    assert tidy_cfbd_weather([]).columns == ["game_id", *WEATHER_COLS]


def test_typed_refuses_a_lossy_integer_cast() -> None:
    with pytest.raises(ValueError, match="snowfall"):
        _typed(pl.DataFrame({"snowfall": [0.0, 0.4]}), ("snowfall",))


def test_cfbd_error_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import sportsdataverse.dl_utils as dl

    class _Resp:
        def json(self):
            return {"message": "rate limited"}

    monkeypatch.setenv("CFBD_API_KEY", "test")
    monkeypatch.setattr(dl, "download", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="unexpected body"):
        _cfbd("/talent?year=2025")
    monkeypatch.delenv("CFBD_API_KEY")
    with pytest.raises(RuntimeError, match="CFBD_API_KEY"):
        _cfbd("/talent?year=2025")
