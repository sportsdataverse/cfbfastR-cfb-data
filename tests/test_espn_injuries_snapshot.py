"""Offline tests for the ESPN league-wide injuries daily snapshot.

Covers the contracts that make the dataset trustworthy: the explode shape (now
delegated to ``sportsdataverse.espn_snapshots.parse_injuries_snapshot``), the
Int64 id boundary and its no-silent-loss guarantee, same-day idempotency of the
append, the hard rule that a zero-row league is never written, and that the
publish uploads the assets this run actually produced.
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
    assert row["athlete_position"] == "RB"
    assert row["detail_fantasy_status"] == "QUESTIONABLE"


def test_schema_tracks_the_library_parser():
    """The producer schema is the library's, with ids re-pinned and season added.

    Derived rather than restated so a column added upstream cannot go missing
    here, and so no second copy of the column list can drift from the parser.
    """
    from sportsdataverse.espn_snapshots import INJURY_SNAPSHOT_SCHEMA

    assert set(snap.SCHEMA) == set(INJURY_SNAPSHOT_SCHEMA) | {"season"}
    assert all(snap.SCHEMA[c] == pl.Int64 for c in snap.ID_COLUMNS)
    assert all(
        snap.SCHEMA[c] == dtype
        for c, dtype in INJURY_SNAPSHOT_SCHEMA.items()
        if c not in snap.ID_COLUMNS
    )


def test_an_unrecoverable_athlete_id_is_null_and_visible():
    """ESPN omits ``athlete.id`` entirely; the id comes from the player-card
    link. If that link is not there the id is null -- never guessed -- and the
    coverage measure is what makes the loss visible instead of silent."""
    payload = _payload(athletes=1)
    payload["injuries"][0]["injuries"][0]["athlete"]["links"] = []

    df = snap.explode(payload, "nfl", date(2026, 9, 2))

    assert df["athlete_id"][0] is None
    assert snap.athlete_id_coverage(df) == 0.0


def test_ids_are_cast_through_int_never_float():
    """``str(123.0)`` is ``"123.0"``: the exact trap the id-dtype rule exists for.

    A float-shaped id must not become 123 by rounding, and must not vanish into
    a null join key either -- it raises, and ``build`` skips that league.
    """
    good = snap.explode(_payload(athletes=1), "nfl", date(2026, 9, 2))
    assert good["athlete_id"][0] == 4870808

    float_shaped = pl.DataFrame(
        {c: pl.Series(["123.0"], dtype=pl.Utf8) for c in snap.ID_COLUMNS}
    )
    with pytest.raises(ValueError, match="did not survive the Int64 cast"):
        snap._cast_ids(float_shaped)


def test_empty_league_yields_empty_frame_with_schema():
    df = snap.explode(
        {"season": {"year": 2027}, "injuries": []}, "mbb", date(2026, 9, 2)
    )

    assert df.is_empty()
    assert list(df.columns) == list(
        snap.SCHEMA
    )  # empty frames carry the documented schema
    assert df.schema["athlete_id"] == pl.Int64  # ...including the id dtypes


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


def _fake_upload(calls, *, raises_for=None):
    def _upload(files, tag, repo=None, **kw):
        calls.append((tag, sorted(f.name for f in files)))
        if raises_for and tag == raises_for:
            raise RuntimeError("retries exhausted")
        return True

    return _upload


def test_publish_uploads_only_what_this_run_wrote(tmp_path, monkeypatch):
    """sportsdataverse_upload defaults to overwrite=True, so a stale local file
    would REPLACE a good release asset. Only `written` may be uploaded."""
    import sportsdataverse.release as rel

    from espn_injuries_daily_snapshot import _publish

    tag = "espn_nfl_injuries"
    (tmp_path / tag).mkdir(parents=True)
    (tmp_path / tag / "injuries_2026.parquet").write_bytes(b"x")
    (tmp_path / tag / "injuries_1999.parquet").write_bytes(b"stale")  # earlier run

    calls = []
    monkeypatch.setattr(rel, "sportsdataverse_upload", _fake_upload(calls))
    written = {f"{tag}/injuries_2026.parquet": 800}  # exactly build()'s key shape
    assert _publish(tmp_path, written, "r/r", dry_run=False) == 0
    assert calls == [(tag, ["injuries_2026.parquet"])], "the stale asset was uploaded"


def test_one_failing_tag_does_not_abandon_the_others(tmp_path, monkeypatch):
    """A rate-limited league must cost one tag, not the seven that follow it."""
    import sportsdataverse.release as rel

    from espn_injuries_daily_snapshot import _publish

    written = {}
    for tag in ("espn_mlb_injuries", "espn_nfl_injuries", "espn_nhl_injuries"):
        (tmp_path / tag).mkdir(parents=True)
        (tmp_path / tag / "injuries_2026.parquet").write_bytes(b"x")
        written[f"{tag}/injuries_2026.parquet"] = 10

    calls = []
    monkeypatch.setattr(
        rel,
        "sportsdataverse_upload",
        _fake_upload(calls, raises_for="espn_mlb_injuries"),
    )
    failed = _publish(tmp_path, written, "r/r", dry_run=False)

    assert failed == 1
    assert [t for t, _ in calls] == [
        "espn_mlb_injuries",
        "espn_nfl_injuries",
        "espn_nhl_injuries",
    ], "a raise in the first tag stopped the rest"


def test_build_output_is_publishable_end_to_end(tmp_path, monkeypatch):
    """``build`` writes the files and ``_publish`` uploads THOSE files.

    Both halves passed on hand-written keys while the real handoff uploaded
    nothing: ``_publish`` re-appended ".parquet" to an asset name that already
    carried it, every path missed, and the run still returned 0. Only feeding
    build()'s own return value into _publish can catch that.
    """
    import sportsdataverse.release as rel

    from espn_injuries_daily_snapshot import _publish

    written = snap.build(
        ["nfl"],
        tmp_path,
        as_of=date(2026, 9, 2),
        fetch=lambda _: _payload(),
        prior_reader=lambda *a, **k: None,
    )

    calls = []
    monkeypatch.setattr(rel, "sportsdataverse_upload", _fake_upload(calls))
    assert _publish(tmp_path, written, "r/r", dry_run=False) == 0
    assert calls == [("espn_nfl_injuries", ["injuries_2026.parquet"])]


def _prior_with_a_retired_column() -> pl.DataFrame:
    """A prior release asset written by an EARLIER schema.

    Not hypothetical: `team_abbreviation` is exactly the column this stage
    dropped when it moved onto the library parser, so the first append after
    that change reads a prior asset that still has it.
    """
    return pl.DataFrame(
        {
            "as_of_date": [date(2026, 9, 1)],
            "league": ["nfl"],
            "season": [2026],
            "team_id": [22],
            "team_abbreviation": ["ARI"],  # retired
            "athlete_id": [4870808],
            "injury_id": [1],
            "status": ["Out"],
        }
    )


def test_a_retired_column_in_the_prior_asset_never_reaches_the_new_write(caplog):
    """diagonal_relaxed unions the columns, so without normalization a column
    the schema no longer has rides into every future asset, all-null forever."""
    today = snap.explode(_payload(), "nfl", date(2026, 9, 2))

    with caplog.at_level("WARNING"):
        merged = snap.append_snapshot(_prior_with_a_retired_column(), today)

    assert list(merged.columns) == list(snap.SCHEMA)
    assert dict(merged.schema) == dict(today.schema)  # and no widened join key
    assert merged.height == 3  # the prior day survives; only its extra column goes
    assert "append_dropped_retired_columns" in caplog.text
    assert "team_abbreviation" in caplog.text


def test_build_writes_the_declared_schema_even_from_a_drifted_prior(tmp_path):
    """The end of the path that matters: what lands on disk and gets published."""
    written = snap.build(
        ["nfl"],
        tmp_path,
        as_of=date(2026, 9, 2),
        fetch=lambda _: _payload(),
        prior_reader=lambda *a, **k: _prior_with_a_retired_column(),
    )

    on_disk = pl.read_parquet(tmp_path / "espn_nfl_injuries" / "injuries_2026.parquet")
    assert written == {"espn_nfl_injuries/injuries_2026.parquet": 3}
    assert list(on_disk.columns) == list(snap.SCHEMA)
    assert on_disk.schema["athlete_id"] == pl.Int64
