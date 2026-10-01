"""Poll history analytics (F8): pure functions on a constructed ranks frame, plus the
capture on a committed ESPN fixture (2024 t2 w5, captured 2026-10-01) with the
fetch monkeypatched. No network."""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from cfb_data_build import polls
from cfb_data_build.cli import POLLS, build_parser, main
from cfb_data_build.config import PKG_FUNCTION
from cfb_data_build.polls import (
    ANALYTICS_SCHEMA,
    SUMMARY_SCHEMA,
    fetch_ranks,
    poll_analytics,
    poll_rows,
    poll_week_summary,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "polls"
A, B, C, D = 1, 2, 3, 4


def _ranks(rows: list[tuple[int, int, int, str, int, int]]) -> pl.DataFrame:
    """(season, season_type, week, poll, team_id, rank) -> a ranks frame."""
    cols = ["season", "season_type", "week", "poll", "team_id", "rank"]
    df = pl.DataFrame(
        [dict(zip(cols, r)) for r in rows],
        schema={c: polls.RANKS_SCHEMA[c] for c in cols},
    )
    return df.with_columns(
        points=pl.lit(None, pl.Float64), first_place_votes=pl.lit(None, pl.Int64)
    )


@pytest.fixture
def two_weeks() -> pl.DataFrame:
    # week 1: A1 B2 C3 ; week 2: B1 A2 D3
    return _ranks(
        [
            (2024, 2, 1, "ap", A, 1),
            (2024, 2, 1, "ap", B, 2),
            (2024, 2, 1, "ap", C, 3),
            (2024, 2, 2, "ap", B, 1),
            (2024, 2, 2, "ap", A, 2),
            (2024, 2, 2, "ap", D, 3),
        ]
    )


def _row(df: pl.DataFrame, week: int, team: int) -> dict:
    out = df.filter((pl.col("week") == week) & (pl.col("team_id") == team))
    assert out.height == 1, (week, team, out)
    return out.row(0, named=True)


def test_movement_entries_and_exits(two_weeks):
    an = poll_analytics(two_weeks)
    assert an.schema == pl.Schema(ANALYTICS_SCHEMA)
    assert _row(an, 2, B)["move"] == 1
    assert _row(an, 2, A)["move"] == -1
    d = _row(an, 2, D)
    assert d["entered"] is True and d["prev_rank"] is None and d["move"] is None
    c = _row(an, 2, C)
    assert c["rank"] is None and c["prev_rank"] == 3 and c["exited"] is True
    assert c["points"] is None and c["first_place_votes"] is None
    # week 1: nobody entered, nobody exited, no previous rank
    wk1 = an.filter(pl.col("week") == 1)
    assert wk1.height == 3
    assert not wk1["entered"].any() and not wk1["exited"].any()
    assert wk1["prev_rank"].null_count() == 3 and wk1["move"].null_count() == 3


def test_weeks_ranked_is_cumulative_and_carried_on_exit_rows(two_weeks):
    an = poll_analytics(two_weeks)
    assert _row(an, 2, A)["weeks_ranked"] == 2
    assert _row(an, 2, C)["weeks_ranked"] == 1
    assert _row(an, 2, D)["weeks_ranked"] == 1


def test_week_summary(two_weeks):
    sm = poll_week_summary(poll_analytics(two_weeks))
    assert sm.schema == pl.Schema(SUMMARY_SCHEMA)
    assert sm.height == 2
    wk1, wk2 = sm.sort("week").rows(named=True)
    assert wk1["volatility"] is None and wk1["chaos"] is None
    assert wk1["entries"] == 0 and wk1["exits"] == 0
    assert wk2["entries"] == 1 and wk2["exits"] == 1
    # union {A,B,C,D}: d = -1, +1, -23, +23 with the unranked side at 26
    assert wk2["chaos"] == 48
    assert math.isclose(wk2["volatility"], math.sqrt(265), abs_tol=1e-9)


def test_postseason_week_1_sequences_after_the_regular_season_last_week():
    an = poll_analytics(
        _ranks(
            [
                (2024, 2, 15, "ap", A, 1),
                (2024, 2, 15, "ap", B, 2),
                (2024, 3, 1, "ap", B, 1),
                (2024, 3, 1, "ap", A, 2),
            ]
        )
    )
    post = an.filter(pl.col("season_type") == 3)
    assert post.height == 2
    assert _row(post, 1, B)["prev_rank"] == 2 and _row(post, 1, B)["move"] == 1
    assert not post["entered"].any()
    sm = poll_week_summary(an).filter(pl.col("season_type") == 3)
    assert sm["chaos"][0] == 2 and sm["entries"][0] == 0


def test_two_teams_tied_at_12_are_both_kept():
    an = poll_analytics(
        _ranks(
            [
                (2024, 2, 1, "ap", A, 11),
                (2024, 2, 1, "ap", B, 12),
                (2024, 2, 1, "ap", C, 12),
            ]
        )
    )
    assert an.filter(pl.col("rank") == 12)["team_id"].sort().to_list() == [B, C]


def test_polls_are_sequenced_independently():
    """A week the CFP has not published yet is not the CFP's first week."""
    an = poll_analytics(
        _ranks(
            [
                (2024, 2, 1, "ap", A, 1),
                (2024, 2, 2, "ap", A, 1),
                (2024, 2, 2, "cfp", A, 1),
                (2024, 2, 3, "cfp", B, 1),
            ]
        )
    )
    cfp = an.filter(pl.col("poll") == "cfp")
    assert not _row(cfp, 2, A)["entered"], "the CFP's first week has no entries"
    assert _row(cfp, 3, B)["entered"] and _row(cfp, 3, A)["exited"]
    sm = poll_week_summary(an)
    assert (
        sm.filter((pl.col("poll") == "cfp") & (pl.col("week") == 2))["chaos"][0] is None
    )
    assert (
        sm.filter((pl.col("poll") == "cfp") & (pl.col("week") == 3))["chaos"][0] == 50
    )


def test_empty_input_carries_the_documented_schemas():
    an = poll_analytics(pl.DataFrame(schema=polls.RANKS_SCHEMA))
    assert an.height == 0 and an.schema == pl.Schema(ANALYTICS_SCHEMA)
    sm = poll_week_summary(an)
    assert sm.height == 0 and sm.schema == pl.Schema(SUMMARY_SCHEMA)


# --- capture on the committed fixture -----------------------------------------


def test_fixture_parses_to_the_25_ap_ranks():
    payload = json.loads((FIX / "2024_t2_w5_ap.json").read_text())
    assert payload["id"] == "1" and payload["type"] == "ap"
    rows = poll_rows(2024, 2, 5, "ap", payload)
    assert len(rows) == 25
    df = pl.DataFrame(rows, schema=polls.RANKS_SCHEMA)
    assert df.schema["team_id"] == pl.Int64
    assert df["rank"].sort().to_list() == list(range(1, 26))
    assert all(isinstance(r["team_id"], int) for r in rows)
    assert df.filter(pl.col("rank") == 1).row(0, named=True)["team_id"] == 251  # Texas
    assert df["first_place_votes"].sum() == 62 and df["points"].max() == 1527.0
    # "others receiving votes" are not ranks
    assert payload["others"] and all(r["team_id"] not in {0, None} for r in rows)


@pytest.fixture
def fake_espn(monkeypatch):
    """Serve the fixture for 2024 t2 w5, an empty listing elsewhere; record every URL."""
    listing = json.loads((FIX / "2024_t2_w5_rankings.json").read_text())
    ap = json.loads((FIX / "2024_t2_w5_ap.json").read_text())
    urls: list[str] = []

    def _get(url: str, **_):
        urls.append(url)
        assert url.startswith("https://sports.core.api.espn.com/"), url
        if url.endswith("/seasons/2024/types/2/weeks/5/rankings"):
            return listing
        if "/weeks/5/rankings/1?" in url or "/weeks/5/rankings/2?" in url:
            return ap  # the Coaches ref serves the same 25 teams
        if "/rankings/" in url.split("/weeks/")[-1]:
            raise AssertionError(f"a dropped poll was fetched: {url}")
        return {"count": 0, "items": []}

    monkeypatch.setattr(polls, "_get", _get)
    monkeypatch.setattr(polls, "_PAUSE", 0)
    return urls


def test_fetch_ranks_follows_only_the_kept_polls(fake_espn):
    df = fetch_ranks(2024)
    assert df.height == 50 and df["poll"].unique().sort().to_list() == ["ap", "coaches"]
    assert df.filter(pl.col("poll") == "ap").height == 25
    assert df.schema == pl.Schema(polls.RANKS_SCHEMA)
    assert df["week"].unique().to_list() == [5] and df[
        "season_type"
    ].unique().to_list() == [2]
    # 40 slots probed (2 types x 20 weeks) + AP + Coaches followed; 20/11/12 never
    assert len(fake_espn) == 42
    assert not [
        u
        for u in fake_espn
        if "/rankings/2?" not in u
        and "/rankings/1?" not in u
        and "/rankings/" in u.split("/weeks/")[-1]
    ]


@pytest.mark.parametrize(
    "payload",
    [{"ranks": []}, {"error": {"code": 500, "message": "oops"}}],
    ids=["no-ranks", "error-envelope"],
)
def test_a_listed_poll_that_returns_no_ranks_raises(monkeypatch, payload):
    """A hole, not an empty week: writing it would publish and commit a gap."""
    listing = json.loads((FIX / "2024_t2_w5_rankings.json").read_text())

    def _get(url: str, **_):
        if url.endswith("/weeks/5/rankings"):
            return listing
        if "/weeks/5/rankings/" in url:
            return payload
        return {"count": 0, "items": []}

    monkeypatch.setattr(polls, "_get", _get)
    monkeypatch.setattr(polls, "_PAUSE", 0)
    with pytest.raises(ValueError, match="wk5 ap: listed but returned no ranks"):
        fetch_ranks(2024)


def test_a_paginated_listing_raises(monkeypatch):
    """`count` beyond the items served means polls we never saw."""
    one = [{"$ref": "http://sports.core.api.espn.com/v2/x/weeks/1/rankings/1?lang=en"}]

    def _get(url: str, **_):
        if url.endswith("/weeks/1/rankings"):
            return {"count": 6, "items": one}
        return {"count": 0, "items": []}

    monkeypatch.setattr(polls, "_get", _get)
    monkeypatch.setattr(polls, "_PAUSE", 0)
    with pytest.raises(ValueError, match="paginated"):
        fetch_ranks(2024)


def test_publish_runs_even_when_the_local_table_is_unchanged(
    fake_espn, tmp_path, monkeypatch
):
    """Local equality is no proof the release has the files (#124 review):
    an upload that failed after the write, or a build-only run followed by
    --publish, must still upload."""
    from cfb_data_build import publish as publish_mod

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        publish_mod,
        "publish_dataset",
        lambda spec, season, *, base: calls.append((spec.tag, season)),
    )
    base = tmp_path / "cfb"
    argv = [
        "--dataset",
        "poll_analytics",
        "-s",
        "2024",
        "-e",
        "2024",
        "--base",
        str(base),
    ]
    assert main(argv) == 0  # build-only: written, not published
    assert calls == []
    assert (
        main([*argv, "--publish"]) == 0
    )  # unchanged locally -> still uploaded, both tags
    assert calls == [("cfb_poll_analytics", 2024), ("cfb_poll_week_summary", 2024)]
    # every release format is on disk for the upload to find
    for d, stem in (
        ("poll_analytics", "cfb_poll_analytics"),
        ("poll_week_summary", "cfb_poll_week_summary"),
    ):
        for sub in ("parquet", "rds", "csv"):
            assert (base / d / sub / f"{stem}_2024.{sub}").exists(), (d, sub)


def test_cli_group_builds_both_tables_and_skips_an_unchanged_season(
    fake_espn, tmp_path, capsys
):
    assert POLLS == ("poll_analytics",)
    choices = next(a.choices for a in build_parser()._actions if a.dest == "dataset")
    assert "poll_analytics" in choices and "poll_week_summary" not in choices
    for tag in ("cfb_poll_analytics", "cfb_poll_week_summary"):
        assert PKG_FUNCTION[tag] == "python/cfb_data_build/polls.py"

    base = tmp_path / "cfb"
    argv = [
        "--dataset",
        "poll_analytics",
        "-s",
        "2024",
        "-e",
        "2024",
        "--base",
        str(base),
    ]
    assert main([*argv, "--dry-run"]) == 0
    assert not base.exists(), "dry-run must not write"

    assert main(argv) == 0
    an = pl.read_parquet(
        base / "poll_analytics" / "parquet" / "cfb_poll_analytics_2024.parquet"
    )
    sm = pl.read_parquet(
        base / "poll_week_summary" / "parquet" / "cfb_poll_week_summary_2024.parquet"
    )
    assert an.height == 50 and an.schema == pl.Schema(ANALYTICS_SCHEMA)
    assert sm.height == 2 and sm.schema == pl.Schema(SUMMARY_SCHEMA)
    assert sm["chaos"].null_count() == 2  # the only week is each poll's first week

    capsys.readouterr()
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("unchanged, skipped") == 2
