"""Offline tests for the ESPN league-wide injuries daily snapshot.

Covers the four contracts that make the dataset trustworthy: the explode shape,
the pinned Int64 id dtypes, same-day idempotency of the append, and the hard
rule that a zero-row league is never written.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import espn_injuries_daily_snapshot as snap


def _payload(*, season: int = 2026, athletes: int = 2) -> dict:
    """Minimal payload in ESPN's real shape (athlete id only in ``links``)."""
    return {
        "season": {"year": season},
        "injuries": [
            {
                "id": "22",
                "displayName": "Arizona Cardinals",
                "injuries": [
                    {
                        "id": str(635850 + i),
                        "status": "Questionable",
                        "date": "2026-09-01T19:38Z",
                        "shortComment": "short",
                        "longComment": "long",
                        "source": {"description": "basic/manual"},
                        "type": {
                            "name": "INJURY_STATUS_QUESTIONABLE",
                            "abbreviation": "Q",
                            "description": "questionable",
                        },
                        "details": {
                            "type": "Ankle",
                            "location": "Leg",
                            "side": "Not Specified",
                            "returnDate": "2026-09-13",
                            "fantasyStatus": {"description": "QUESTIONABLE"},
                        },
                        "athlete": {
                            "displayName": f"Player {i}",
                            "shortName": f"P. {i}",
                            "position": {
                                "abbreviation": "RB",
                                "displayName": "Running Back",
                            },
                            "team": {"abbreviation": "ARI"},
                            "links": [
                                {
                                    "href": f"https://www.espn.com/nfl/player/_/id/{4870808 + i}/p"
                                }
                            ],
                        },
                    }
                    for i in range(athletes)
                ],
            }
        ],
    }


def test_explode_shape_and_id_dtypes():
    df = snap.explode(_payload(), "nfl", date(2026, 9, 2))

    # one row per (team, athlete, injury), not one per team
    assert df.height == 2
    assert list(df.columns) == list(snap.SCHEMA)
    assert df.schema["team_id"] == pl.Int64
    assert df.schema["athlete_id"] == pl.Int64
    assert df.schema["injury_id"] == pl.Int64
    assert df.schema["as_of_date"] == pl.Date

    row = df.row(0, named=True)
    assert row["team_id"] == 22
    assert row["athlete_id"] == 4870808  # recovered from links, not a top-level id
    assert row["league"] == "nfl"
    assert row["season"] == 2026
    assert row["position_abbreviation"] == "RB"
    assert row["team_abbreviation"] == "ARI"
    assert row["fantasy_status"] == "QUESTIONABLE"


def test_athlete_id_falls_back_to_headshot():
    payload = _payload(athletes=1)
    athlete = payload["injuries"][0]["injuries"][0]["athlete"]
    athlete["links"] = []
    athlete["headshot"] = {
        "href": "https://a.espncdn.com/i/headshots/nfl/players/full/12345.png"
    }

    assert snap.explode(payload, "nfl", date(2026, 9, 2))["athlete_id"][0] == 12345


def test_empty_league_yields_empty_frame_with_schema():
    df = snap.explode(
        {"season": {"year": 2027}, "injuries": []}, "mbb", date(2026, 9, 2)
    )

    assert df.is_empty()
    assert list(df.columns) == list(
        snap.SCHEMA
    )  # empty frames carry the documented schema


def test_empty_league_is_skipped_not_written(tmp_path):
    written = snap.build(
        ["mbb"],
        tmp_path,
        as_of=date(2026, 9, 2),
        fetch=lambda _: {"season": {"year": 2027}, "injuries": []},
        prior_reader=lambda *a, **k: None,
    )

    assert written == {}
    assert not list(tmp_path.rglob("*.parquet"))  # hard rule: no zero-row asset


def test_append_adds_a_day():
    day1 = snap.explode(_payload(), "nfl", date(2026, 9, 1))
    day2 = snap.explode(_payload(), "nfl", date(2026, 9, 2))

    merged = snap.append_snapshot(day1, day2)

    assert merged.height == 4
    assert merged["as_of_date"].n_unique() == 2


def test_rerunning_the_same_day_is_idempotent():
    day1 = snap.explode(_payload(), "nfl", date(2026, 9, 1))
    day2 = snap.explode(_payload(), "nfl", date(2026, 9, 2))
    history = snap.append_snapshot(day1, day2)

    # same day, but the report changed: one athlete cleared
    day2_again = snap.explode(_payload(athletes=1), "nfl", date(2026, 9, 2))
    rerun = snap.append_snapshot(history, day2_again)

    assert rerun.height == 3  # 2 from day1 + 1 replacing day2's 2, not 5
    assert rerun.filter(pl.col("as_of_date") == date(2026, 9, 2)).height == 1
    assert rerun.filter(pl.col("as_of_date") == date(2026, 9, 1)).height == 2

    # and a third identical run changes nothing
    assert snap.append_snapshot(rerun, day2_again).equals(rerun)


def test_build_appends_to_the_prior_release_asset(tmp_path):
    prior = snap.append_snapshot(
        None, snap.explode(_payload(), "nfl", date(2026, 9, 1))
    )
    seen: dict[str, str] = {}

    def prior_reader(tag, asset, **_):
        seen["tag"], seen["asset"] = tag, asset
        return prior

    written = snap.build(
        ["nfl"],
        tmp_path,
        as_of=date(2026, 9, 2),
        fetch=lambda _: _payload(),
        prior_reader=prior_reader,
    )

    assert seen == {"tag": "espn_nfl_injuries", "asset": "injuries_2026.parquet"}
    assert written == {"espn_nfl_injuries/injuries_2026.parquet": 4}

    on_disk = pl.read_parquet(tmp_path / "espn_nfl_injuries" / "injuries_2026.parquet")
    assert on_disk["as_of_date"].n_unique() == 2
    assert on_disk.schema["athlete_id"] == pl.Int64


def test_one_failing_league_does_not_sink_the_run(tmp_path):
    def fetch(league):
        if league == "nba":
            raise RuntimeError("boom")
        return _payload()

    written = snap.build(
        ["nba", "nfl"],
        tmp_path,
        as_of=date(2026, 9, 2),
        fetch=fetch,
        prior_reader=lambda *a, **k: None,
    )

    assert list(written) == ["espn_nfl_injuries/injuries_2026.parquet"]


def test_read_prior_returns_none_on_404_and_raises_otherwise(monkeypatch):
    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            raise RuntimeError(f"HTTP {self.status_code}")

    monkeypatch.setattr(snap.requests, "get", lambda *a, **k: Resp(404))
    assert snap.read_prior("espn_nfl_injuries", "injuries_2026.parquet") is None

    # a 403 / rate limit must NOT read as "no history" -- that would truncate it
    monkeypatch.setattr(snap.requests, "get", lambda *a, **k: Resp(403))
    with pytest.raises(RuntimeError):
        snap.read_prior("espn_nfl_injuries", "injuries_2026.parquet")


def test_publish_is_opt_in(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(snap, "fetch_league", lambda _: _payload())
    monkeypatch.setattr(snap, "read_prior", lambda *a, **k: None)

    def explode_on_upload(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("published without --publish")

    monkeypatch.setattr(snap, "_publish", explode_on_upload)

    assert snap.main(["-l", "nfl", "--out", str(tmp_path)]) == 0
    assert "espn_nfl_injuries/injuries_2026.parquet: 2 rows" in capsys.readouterr().out


def test_athlete_id_coverage_makes_a_silent_id_loss_visible():
    """The id is RECOVERED from ESPN player links, not read from a field.

    If ESPN reshapes those links every id becomes null, the rows still look fine,
    and the snapshot quietly stops being a usable player time series. The measure
    is what turns that into a visible warning.
    """
    import polars as pl

    from espn_injuries_daily_snapshot import athlete_id_coverage

    full = pl.DataFrame({"athlete_id": pl.Series([1, 2, 3], dtype=pl.Int64)})
    assert athlete_id_coverage(full) == 1.0

    partial = pl.DataFrame(
        {"athlete_id": pl.Series([1, None, None, 4], dtype=pl.Int64)}
    )
    assert athlete_id_coverage(partial) == 0.5

    none_at_all = pl.DataFrame({"athlete_id": pl.Series([None, None], dtype=pl.Int64)})
    assert athlete_id_coverage(none_at_all) == 0.0

    # an empty frame is not a coverage failure -- it is the empty-league skip
    assert athlete_id_coverage(pl.DataFrame()) == 1.0
