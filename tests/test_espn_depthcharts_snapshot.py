"""Offline tests for the ESPN depth-charts daily snapshot.

Covers the contracts that make the dataset trustworthy: the Int64 id boundary and
its no-silent-loss guarantee, the depth-chart grain (slot key, not position id),
same-day idempotency of the append, the hard rule that a zero-row league is never
written, and that the publish uploads the assets this run actually produced.

The payload flatten itself belongs to
``sportsdataverse.espn_snapshots.parse_depthchart_snapshot`` and is tested there
against real captures; these tests own the producer's half.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from sportsdataverse.espn_snapshots import DEPTHCHART_SNAPSHOT_SCHEMA

import espn_depthcharts_daily_snapshot as snap

STAMP = date(2026, 9, 2)


def _frame(*, teams: int = 2, season: int = 2026, as_of: date = STAMP) -> pl.DataFrame:
    """A parser-shaped frame: Utf8 ids, two teams, WR1/WR2 sharing a position id."""
    rows = []
    for team in range(1, teams + 1):
        for slot, rank_count in (("wr1", 2), ("wr2", 1)):
            for rank in range(1, rank_count + 1):
                rows.append(
                    {
                        "as_of_date": as_of,
                        "league": "nfl",
                        "season": season,
                        "season_type": 1,
                        "team_id": str(team),
                        "team_display_name": f"Team {team}",
                        "team_abbreviation": f"T{team}",
                        "group_id": "21",
                        "group_name": "3WR 1TE",
                        "position_slot": slot,
                        "position_id": "1",
                        "position_abbreviation": "WR",
                        "position_name": "Wide Receiver",
                        "depth_rank": rank,
                        "athlete_id": f"{4000000 + team * 10 + rank}",
                        "athlete_display_name": f"Player {team}-{slot}-{rank}",
                        "athlete_short_name": f"P. {rank}",
                        "espn_timestamp": "2026-09-02T18:02:40Z",
                    }
                )
    return pl.DataFrame(rows, schema=DEPTHCHART_SNAPSHOT_SCHEMA)


def _empty() -> pl.DataFrame:
    return pl.DataFrame([], schema=DEPTHCHART_SNAPSHOT_SCHEMA)


def test_schema_tracks_the_library_parser():
    """Derived from the parser's contract, not restated, so a column added
    upstream arrives here instead of being silently dropped."""
    assert list(snap.SCHEMA) == list(DEPTHCHART_SNAPSHOT_SCHEMA)
    assert all(snap.SCHEMA[c] == pl.Int64 for c in snap.ID_COLUMNS)
    assert all(snap.SCHEMA[c] == dtype for c, dtype in DEPTHCHART_SNAPSHOT_SCHEMA.items() if c not in snap.ID_COLUMNS)


def test_ids_are_cast_through_int_never_float():
    """``str(123.0)`` is ``"123.0"`` -- the trap the repo's id rule exists for.

    A float-shaped id must not round to 123 and must not vanish into a null join
    key either: it raises, and ``build`` skips that league with a log line.
    """
    cast = snap.cast_ids(_frame())
    assert all(cast.schema[c] == pl.Int64 for c in snap.ID_COLUMNS)
    assert cast["athlete_id"][0] == 4000011

    broken = _frame().with_columns(pl.lit("123.0").alias("athlete_id"))
    with pytest.raises(ValueError, match="did not survive the Int64 cast"):
        snap.cast_ids(broken)


def test_the_slot_key_survives_to_the_published_frame():
    """NFL ships wr1/wr2/wr3 slots that all carry position id 1. Without the slot
    key the grain is not unique and "who is the WR1" is unanswerable."""
    df = snap.cast_ids(_frame())

    assert df["position_id"].unique().to_list() == [1]
    assert sorted(df["position_slot"].unique().to_list()) == ["wr1", "wr2"]
    assert df.select(snap.SORT_KEYS).is_duplicated().sum() == 0


def test_empty_league_is_skipped_not_written(tmp_path):
    written = snap.build(
        ["nfl"],
        tmp_path,
        as_of=STAMP,
        fetch=lambda *_: _empty(),
        prior_reader=lambda *a, **k: None,
    )

    assert written == {}
    assert not list(tmp_path.rglob("*.parquet"))  # hard rule: no zero-row asset


def test_rerunning_the_same_day_is_idempotent():
    day1 = snap.cast_ids(_frame(as_of=date(2026, 9, 1)))
    day2 = snap.cast_ids(_frame(as_of=STAMP))
    history = snap.append_snapshot(day1, day2, snap.SORT_KEYS)

    assert history.height == 12  # 6 rows a day, two days
    assert history["as_of_date"].n_unique() == 2

    # the chart changed within the day: a team dropped out of the report
    day2_again = snap.cast_ids(_frame(teams=1, as_of=STAMP))
    rerun = snap.append_snapshot(history, day2_again, snap.SORT_KEYS)

    assert rerun.height == 9  # 6 from day 1 + 3 REPLACING day 2's 6, not 15
    assert rerun.filter(pl.col("as_of_date") == STAMP).height == 3
    assert rerun.filter(pl.col("as_of_date") == date(2026, 9, 1)).height == 6
    assert snap.append_snapshot(rerun, day2_again, snap.SORT_KEYS).equals(rerun)


def test_build_appends_to_the_prior_release_asset(tmp_path):
    prior = snap.append_snapshot(None, snap.cast_ids(_frame(as_of=date(2026, 9, 1))), snap.SORT_KEYS)
    seen: dict[str, str] = {}

    def prior_reader(tag, asset, **_):
        seen["tag"], seen["asset"] = tag, asset
        return prior

    written = snap.build(
        ["nfl"],
        tmp_path,
        as_of=STAMP,
        fetch=lambda *_: _frame(),
        prior_reader=prior_reader,
    )

    assert seen == {"tag": "espn_nfl_depthcharts", "asset": "depthcharts_2026.parquet"}
    assert written == {"espn_nfl_depthcharts/depthcharts_2026.parquet": 12}

    on_disk = pl.read_parquet(tmp_path / "espn_nfl_depthcharts" / "depthcharts_2026.parquet")
    assert on_disk["as_of_date"].n_unique() == 2
    assert on_disk.schema["athlete_id"] == pl.Int64


def test_one_failing_league_does_not_sink_the_run(tmp_path):
    def fetch(league, _as_of):
        if league == "nba":
            raise RuntimeError("boom")
        return _frame()

    written = snap.build(
        ["nba", "nfl"],
        tmp_path,
        as_of=STAMP,
        fetch=fetch,
        prior_reader=lambda *a, **k: None,
    )

    assert list(written) == ["espn_nfl_depthcharts/depthcharts_2026.parquet"]


def test_only_the_three_leagues_espn_publishes_are_wired():
    """nhl / wnba / cfb answer 200 with the ``depthchart`` key absent (probed
    live 2026-09-02). Wiring them would cost 122 requests a day for zero rows.
    ``team_transactions`` is dead ({} on 10/10 probes) and is not wired at all."""
    assert snap.LEAGUES == ("nfl", "nba", "mlb")
    assert snap.REQUEST_DELAY >= 1.5


def test_publish_is_opt_in(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(snap, "fetch_league", lambda *_: _frame())
    monkeypatch.setattr(snap, "read_prior", lambda *a, **k: None)
    monkeypatch.setattr(
        snap,
        "_publish",
        lambda *a, **k: pytest.fail("published without --publish"),
    )

    assert snap.main(["-l", "nfl", "--out", str(tmp_path)]) == 0
    assert "espn_nfl_depthcharts/depthcharts_2026.parquet: 6 rows" in capsys.readouterr().out


def test_build_output_is_publishable_end_to_end(tmp_path, monkeypatch):
    """``build`` writes the files and ``_publish`` uploads THOSE files.

    The injuries stage shipped a publish that matched no path and still returned
    0, because both of its tests hand-wrote a key shape ``build()`` never emits.
    Feed build()'s own return value in, or the handoff is untested.
    """
    import sportsdataverse.release as rel

    written = snap.build(
        ["nfl"],
        tmp_path,
        as_of=STAMP,
        fetch=lambda *_: _frame(),
        prior_reader=lambda *a, **k: None,
    )

    calls = []
    monkeypatch.setattr(
        rel,
        "sportsdataverse_upload",
        lambda files, tag, repo=None, **kw: calls.append((tag, sorted(f.name for f in files))) or True,
    )
    assert snap._publish(tmp_path, written, "r/r", dry_run=False) == 0
    assert calls == [("espn_nfl_depthcharts", ["depthcharts_2026.parquet"])]
