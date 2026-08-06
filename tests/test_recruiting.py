"""Tests for the raw-store -> publishable recruiting compile.

The risk this module carries is not arithmetic (that is sdv-py's, and is reused
rather than reimplemented) -- it is publishing something incomplete while
looking healthy. These tests concentrate there.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from cfb_data_build.checks import assert_talent_is_real
from cfb_data_build.recruiting import (
    TALENT_WINDOW,
    available_years,
    build_recruiting,
    build_recruits,
    load_year,
)


def _write_year(root, year: int, *, complete: bool = True, players: int = 8) -> None:
    """Lay down a raw class year the way the scraper does."""
    d = root / "cfb" / "recruits" / "json" / str(year)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "pagination": {
            "currentPage": 1,
            "itemsPerPage": 250,
            "count": players,
            "pageCount": 1,
        },
        "players": [
            {
                "key": year * 1000 + i,
                "firstName": "First",
                "lastName": f"Last{i}",
                # Stars must VARY across teams or every blue-chip ratio is
                # identical and the talent gate (rightly) rejects the build --
                # a uniform fixture is not a realistic recruiting class.
                "compositeStarRating": 5 - (i % 4),
                "compositeRating": 95.0 - i,
                "primaryPosition": "QB",
                "committedInstitution": {
                    "teamKey": 71 + i,
                    "fullName": f"Team {71 + i}",
                },
            }
            for i in range(players)
        ],
    }
    (d / "page_0001.json").write_text(
        json.dumps(
            {
                "year": year,
                "page": 1,
                "page_size": 250,
                "sport_key": 1,
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )
    if complete:
        (d / "_manifest.json").write_text(
            json.dumps(
                {
                    "year": year,
                    "pages": 1,
                    "rows": players,
                    "expected_rows": players,
                    "complete": True,
                }
            ),
            encoding="utf-8",
        )


def test_available_years_only_offers_complete_classes(tmp_path) -> None:
    """An unfinished scrape must not be offered to the compile.

    Compiling a half-scraped class understates every team's talent without
    erroring -- the exact silent-degradation shape this pipeline exists to end.
    """
    _write_year(tmp_path, 2015, complete=True)
    _write_year(tmp_path, 2016, complete=False)
    assert available_years(tmp_path) == [2015]


def test_available_years_empty_when_store_absent(tmp_path) -> None:
    assert available_years(tmp_path / "nope") == []


def test_load_year_refuses_an_incomplete_class(tmp_path) -> None:
    _write_year(tmp_path, 2016, complete=False)
    with pytest.raises(FileNotFoundError, match="no complete manifest"):
        load_year(tmp_path, 2016)


def test_load_year_reads_the_raw_store_offline(tmp_path) -> None:
    """No network: the parse + normalize chain runs entirely off disk."""
    _write_year(tmp_path, 2016, players=3)
    df = load_year(tmp_path, 2016)
    assert df.height == 3
    assert df.schema["team_id"] == pl.Utf8
    assert df["season"].unique().to_list() == [2016]
    # integer-origin 247 key -> "71", never "71.0"
    assert set(df["team_id_247"].to_list()) == {"71", "72", "73"}
    # `team_id` is the ESPN id, resolved from the team NAME -- these synthetic
    # schools do not exist, so it stays null. That is the correct outcome: an
    # unresolvable team must not silently inherit 247's key.
    assert df["team_id"].null_count() == df.height


def test_build_recruits_raises_when_nothing_is_available(tmp_path) -> None:
    with pytest.raises(ValueError, match="no complete recruit classes"):
        build_recruits(tmp_path, [2016])


def test_build_recruiting_requires_the_full_talent_window(tmp_path) -> None:
    """A short window silently understates talent, so refuse to publish it.

    Talent for season S accumulates classes S-3..S. With only some of them
    scraped the composite is smaller for every team, uniformly and invisibly --
    it still looks like a plausible leaderboard.
    """
    for y in range(2016 - TALENT_WINDOW + 1, 2016):  # everything except 2016 itself
        _write_year(tmp_path, y)
    failures = build_recruiting(
        "team_talent", 2016, 2016, raw_root=tmp_path, base=str(tmp_path / "out")
    )
    assert failures == [(2016, "FileNotFoundError")]


def test_build_recruiting_writes_parquet_when_the_window_is_whole(
    tmp_path, monkeypatch
) -> None:
    # Synthetic schools never resolve to ESPN ids, and talent drops unresolved
    # teams -- correctly. Stub the re-key so this test stays about the window
    # and publish logic; the re-key itself is covered by its own test.
    import sys as _sys

    tal_mod = _sys.modules["sportsdataverse.cfb.cfb_roster_talent"]
    monkeypatch.setattr(
        tal_mod,
        "_add_espn_team_id",
        lambda df: df.with_columns(pl.col("team_id_247").alias("team_id")),
    )
    for y in range(2016 - TALENT_WINDOW + 1, 2017):
        _write_year(tmp_path, y)
    out = tmp_path / "out"
    failures = build_recruiting(
        "team_talent", 2016, 2016, raw_root=tmp_path, base=str(out)
    )
    assert failures == []
    written = out / "cfb_team_talent" / "cfb_team_talent_2016.parquet"
    assert written.is_file()
    df = pl.read_parquet(written)
    assert df.height > 0 and "talent_composite" in df.columns


def test_build_recruiting_rejects_an_unknown_dataset(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown recruiting dataset"):
        build_recruiting("nope", 2016, 2016, raw_root=tmp_path)


def test_talent_gate_fires_on_the_shapes_the_bug_produced() -> None:
    """Empty, all-null and flat must each raise; only real data passes.

    `cfb_roster_talent` returned zero rows from the day it was written; nothing
    measured the talent itself, so it shipped. This gate is that measurement.
    """
    real = pl.DataFrame(
        {
            "season": [2016] * 4,
            "team_id": ["1", "2", "3", "4"],
            "talent_composite": [900.0, 700.0, 500.0, 300.0],
            "blue_chip_ratio": [0.70, 0.45, 0.20, 0.00],
        }
    )
    assert_talent_is_real(real, label="real")  # must not raise

    with pytest.raises(ValueError, match="EMPTY"):
        assert_talent_is_real(real.head(0), label="empty")
    with pytest.raises(ValueError, match="entirely null"):
        assert_talent_is_real(
            real.with_columns(pl.lit(None, dtype=pl.Float64).alias("talent_composite")),
            label="null",
        )
    with pytest.raises(ValueError, match="every team looks alike"):
        assert_talent_is_real(
            real.with_columns(pl.lit(0.2, dtype=pl.Float64).alias("blue_chip_ratio")),
            label="flat",
        )
    with pytest.raises(ValueError, match="missing"):
        assert_talent_is_real(real.drop("blue_chip_ratio"), label="missing")


def test_build_recruits_emits_espn_ids_not_247_keys(tmp_path) -> None:
    """The compile must resolve `team_id` to the ESPN id, like the sdv-py loader.

    The producer assembles pages itself rather than going through
    `load_recruit_classes`, so it has to apply the ESPN re-key too. When it
    didn't, `team_id` stayed at the null placeholder the normalizer emits and
    every downstream join silently matched nothing.

    Worse than nothing, actually: before the re-key existed at all, `team_id`
    held 247's own key, and because both are small integers a fraction of them
    COLLIDE with ESPN ids -- an inner join then returns plausible rows for the
    wrong teams. Measured during this change: the 2024 leaderboard read
    Rutgers / Cornell / Yale / MIT carrying Alabama's and Georgia's numbers.
    """
    _write_year(tmp_path, 2016)
    df = build_recruits(tmp_path, [2016])
    assert "team_id_247" in df.columns, df.columns
    assert "team_id" in df.columns
    # the synthetic teams are not real schools, so they will not resolve --
    # what matters is that the column exists, is Utf8, and is NOT the 247 key
    assert df.schema["team_id"] == pl.Utf8
    assert df.schema["team_id_247"] == pl.Utf8
    resolved = df.filter(pl.col("team_id").is_not_null())
    if resolved.height:
        assert (resolved["team_id"] != resolved["team_id_247"]).any(), (
            "team_id is identical to team_id_247 -- the ESPN re-key did not run"
        )


def test_returning_production_needs_no_raw_store(tmp_path, monkeypatch) -> None:
    """`returning_production` builds from the player box, not the 247 store.

    It is grouped with the recruiting datasets because it is the same
    roster-continuity family and publishes identically, but requiring a raw
    store for it would make an empty store block a dataset that never reads one.
    """
    import cfb_data_build.recruiting as rec_mod

    fake = pl.DataFrame(
        {
            "season": [2012] * 4,
            "team_id": ["1", "2", "3", "4"],
            "off_returning": [0.70, 0.55, 0.40, 0.30],
            "def_returning": [0.60, 0.50, 0.45, 0.35],
            "overall_returning": [0.70, 0.55, 0.40, 0.30],
            "n_returning": [20, 18, 15, 12],
        }
    )
    monkeypatch.setattr(
        "sportsdataverse.cfb.cfb_returning_production",
        lambda *a, **k: fake,
        raising=False,
    )
    import sportsdataverse.cfb as sdv_cfb

    monkeypatch.setattr(
        sdv_cfb, "cfb_returning_production", lambda *a, **k: fake, raising=False
    )

    out = tmp_path / "out"
    # tmp_path holds NO recruit store, which is the point
    failures = rec_mod.build_recruiting(
        "returning_production", 2012, 2012, raw_root=tmp_path, base=str(out)
    )
    assert failures == [], failures
    assert (
        out / "cfb_returning_production" / "cfb_returning_production_2012.parquet"
    ).is_file()


def test_returning_gate_rejects_a_collapsed_join() -> None:
    """A returning table where every team matches must not publish.

    Returning production is a ratio over two joins; when either key collapses
    every team lands on the same value and the frame still looks well-formed.
    """
    from cfb_data_build.checks import assert_returning_is_real

    real = pl.DataFrame(
        {
            "season": [2012] * 4,
            "team_id": ["1", "2", "3", "4"],
            "off_returning": [0.70, 0.55, 0.40, 0.30],
            "overall_returning": [0.70, 0.55, 0.40, 0.30],
        }
    )
    assert_returning_is_real(real, label="real")

    with pytest.raises(ValueError, match="EMPTY"):
        assert_returning_is_real(real.head(0), label="empty")
    with pytest.raises(ValueError, match="a join key collapsed"):
        assert_returning_is_real(
            real.with_columns(pl.lit(0.5, dtype=pl.Float64).alias("off_returning")),
            label="flat",
        )
    # A low mean must trip the RANGE check, not the sd check -- so this frame
    # keeps real spread (sd 0.08) while sitting far below a plausible season.
    # Scaling the real values instead collapsed sd as well and silently tested
    # the wrong branch.
    zeroish = real.with_columns(
        pl.Series("off_returning", [0.0, 0.0, 0.0, 0.16]),
        pl.Series("overall_returning", [0.0, 0.0, 0.0, 0.16]),
    )
    assert zeroish["off_returning"].std() > 0.05  # the sd gate must NOT be what fires
    with pytest.raises(ValueError, match="outside"):
        assert_returning_is_real(zeroish, label="zeroish")
