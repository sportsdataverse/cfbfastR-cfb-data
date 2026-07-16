"""Hermetic tests for the cfb_ratings builder.

The sdv-py compute seam is stubbed, so these assert *orchestration* -- season
ordering, the empty-frame refusal, the card sidecar, and per-file upload -- not
the ratings math (that is gated in sdv-py's own oracle suite).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from cfb_model_publish.artifacts import upload_artifacts
from cfb_model_publish.builders import MIN_SEASON, build_ratings, write_ratings_card
from cfb_model_publish.cli import _seasons, main


def _fake_ratings(season: int) -> pl.DataFrame:
    """A `cfb_ratings`-shaped frame: one row per team."""
    return pl.DataFrame(
        {
            "season": [season, season],
            "team_id": ["130", "194"],
            "adj_off_epa": [0.39, 0.26],
            "adj_net": [0.65, 0.49],
            "net_rank": [1, 2],
        }
    )


def test_build_ratings_writes_one_parquet_per_season_in_order(tmp_path):
    results = build_ratings([2022, 2023], tmp_path, compute=_fake_ratings)

    assert [r["season"] for r in results] == [2022, 2023]
    assert [r["rows"] for r in results] == [2, 2]
    for season in (2022, 2023):
        path = tmp_path / f"cfb_ratings_{season}.parquet"
        assert path.exists()
        # season round-trips -- guards against every file being the same season
        assert pl.read_parquet(path)["season"].unique().to_list() == [season]


def test_build_ratings_refuses_an_empty_season(tmp_path):
    """sdv-py returns a typed EMPTY frame (it does not raise) when a season has
    no published pbp asset -- publishing that would ship a silently-empty tag.
    """
    empty = pl.DataFrame(schema={"season": pl.Int64, "team_id": pl.Utf8})

    with pytest.raises(ValueError, match="0 rows"):
        build_ratings([2023], tmp_path, compute=lambda s: empty)

    assert not (tmp_path / "cfb_ratings_2023.parquet").exists()


def test_build_ratings_rejects_seasons_below_the_pbp_floor(tmp_path):
    with pytest.raises(ValueError, match=str(MIN_SEASON)):
        build_ratings([MIN_SEASON - 1], tmp_path, compute=_fake_ratings)


def test_card_carries_seasons_and_parity_anchors(tmp_path):
    results = build_ratings([2023], tmp_path, compute=_fake_ratings)
    path = write_ratings_card(results, tmp_path)

    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["tag"] == "cfb_ratings"
    assert card["seasons"] == [2023]
    assert card["rows_by_season"] == {"2023": 2}
    # the anchors are the point of the card -- they must be real numbers
    assert card["parity_anchors_2023"]["adj_net_vs_espn_fpi"] == pytest.approx(0.9259)


def test_upload_pattern_selects_parquet_and_card_not_models(tmp_path):
    """`pattern=` must bypass model discovery -- `discover_models` finds no
    parquet, so without it a dataset tag would upload nothing.
    """
    (tmp_path / "cfb_ratings_2023.parquet").write_bytes(b"x")
    (tmp_path / "cfb_ratings_card.json").write_text("{}")
    (tmp_path / "unrelated.txt").write_text("no")

    calls: list = []
    res = upload_artifacts(
        tmp_path,
        "cfb_ratings",
        "sportsdataverse/sportsdataverse-data",
        pattern="cfb_ratings_*.*",
        runner=lambda args: calls.append(args),
        exists_check=lambda tag, repo: True,
    )

    names = sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in res["files"])
    assert names == ["cfb_ratings_2023.parquet", "cfb_ratings_card.json"]
    assert res["uploaded"] == 2
    assert all("--clobber" in c for c in calls)


def test_seasons_parses_range_and_single():
    assert _seasons("2023") == [2023]
    assert _seasons("2004:2007") == [2004, 2005, 2006, 2007]


def test_cli_build_only_writes_files_and_skips_upload(tmp_path, monkeypatch):
    import cfb_model_publish.cli as cli

    monkeypatch.setattr(
        cli,
        "build_ratings",
        lambda seasons, out, **kw: build_ratings(seasons, out, compute=_fake_ratings),
    )
    monkeypatch.setattr(
        cli,
        "upload_artifacts",
        lambda *a, **k: pytest.fail("--build-only must not upload"),
    )

    rc = main(["ratings", "--seasons", "2023", "--out", str(tmp_path), "--build-only"])

    assert rc == 0
    assert (tmp_path / "cfb_ratings_2023.parquet").exists()
    assert (tmp_path / "cfb_ratings_card.json").exists()
