"""A rebuild that lost a provider's ids never replaces the file that has them."""

import polars as pl

import build_cfb_crosswalk as bx


def test_coverage_collapse_is_caught_and_never_guards_uncovered_seasons(tmp_path):
    p = tmp_path / "cfb_schedule_crosswalk_2024.parquet"
    assert not bx.coverage_collapsed(pl.DataFrame({"yahoo_game_id": [None]}), p, "yahoo_game_id")  # no file yet
    pl.DataFrame({"yahoo_game_id": ["a", "b", "c", "d"]}).write_parquet(p)
    outage = pl.DataFrame({"yahoo_game_id": pl.Series([None] * 3, dtype=pl.String)})
    assert bx.coverage_collapsed(outage, p, "yahoo_game_id")
    assert bx.coverage_collapsed(pl.DataFrame({"espn_game_id": [1]}), p, "yahoo_game_id")  # column gone
    assert not bx.coverage_collapsed(pl.DataFrame({"yahoo_game_id": ["a", "b", "c"]}), p, "yahoo_game_id")
    pl.DataFrame({"yahoo_game_id": pl.Series([None], dtype=pl.String)}).write_parquet(p)  # pre-Yahoo season
    assert not bx.coverage_collapsed(outage, p, "yahoo_game_id")


def test_a_guarded_season_keeps_the_old_file_and_exits_zero(tmp_path, monkeypatch):
    p = tmp_path / "cfb_schedule_crosswalk_2024.parquet"
    pl.DataFrame({"yahoo_game_id": ["a", "b"]}).write_parquet(p)
    before = p.read_bytes()
    monkeypatch.setattr(bx, "build_schedule", lambda s: pl.DataFrame({"yahoo_game_id": pl.Series([None, None], dtype=pl.String)}))
    rc = bx.main(["-s", "2024", "-e", "2024", "--datasets", "schedule", "--out-dir", str(tmp_path)])
    assert rc == 0 and p.read_bytes() == before
