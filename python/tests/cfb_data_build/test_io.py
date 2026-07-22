"""IO contract tests -- the Python ``write_dataset`` mirror of R ``test-data-utils.R``."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cfb_data_build.io import _append_manifest, write_dataset


def test_write_dataset_writes_parquet_rds_csv_and_manifest(tmp_path: Path) -> None:
    df = pl.DataFrame({"game_id": [1, 2], "x": [10.0, 20.0]})
    out = write_dataset(df, "demo", 2024, "demo", base=tmp_path)

    assert out is not None
    assert (tmp_path / "demo" / "parquet" / "demo_2024.parquet").exists()
    # all three released formats, matching R (parquet + rds + plain csv)
    assert (tmp_path / "demo" / "rds" / "demo_2024.rds").exists()
    assert (tmp_path / "demo" / "csv" / "demo_2024.csv").exists()
    assert not (tmp_path / "demo" / "csv" / "demo_2024.csv.gz").exists()

    # parquet round-trips identically
    back = pl.read_parquet(tmp_path / "demo" / "parquet" / "demo_2024.parquet")
    assert back.shape == (2, 2)

    # csv is PLAIN text (not gzipped): header + 2 rows
    lines = (
        (tmp_path / "demo" / "csv" / "demo_2024.csv").read_text().strip().splitlines()
    )
    assert lines[0] == "game_id,x"
    assert len(lines) == 3

    # rds is a real (gzip-compressed) RDS payload, non-empty
    assert (tmp_path / "demo" / "rds" / "demo_2024.rds").stat().st_size > 0

    # manifest row
    m = pl.read_csv(tmp_path / "demo" / "cfb_demo_in_data_repo.csv")
    assert m["season"].to_list() == [2024]
    assert m["row_count"].to_list() == [2]
    assert "generated_at_utc" in m.columns


def test_write_dataset_skips_empty_frame(tmp_path: Path) -> None:
    assert write_dataset(pl.DataFrame(), "demo", 2024, "demo", base=tmp_path) is None
    assert not (tmp_path / "demo").exists()


def test_append_manifest_upserts_season_sorted(tmp_path: Path) -> None:
    _append_manifest("demo", 2024, 100, tmp_path)
    _append_manifest("demo", 2023, 50, tmp_path)
    _append_manifest("demo", 2024, 111, tmp_path)  # re-run overwrites 2024

    m = pl.read_csv(tmp_path / "demo" / "cfb_demo_in_data_repo.csv")
    assert m["season"].to_list() == [2023, 2024]  # sorted, de-duped
    assert m.filter(pl.col("season") == 2024)["row_count"].item() == 111


def test_list_columns_serialize_to_json_strings_in_parquet(tmp_path: Path) -> None:
    # mirrors R test-data-utils.R "write_dataset serializes list-columns".
    # cfb_data_build folds the JSON-encode into reshape, so the frame arrives
    # with the participants column already a string; confirm it round-trips.
    df = pl.DataFrame({"game_id": [1], "participants": ['[{"id":1,"role":"rusher"}]']})
    write_dataset(df, "nested", 2024, "nested", base=tmp_path)
    back = pl.read_parquet(tmp_path / "nested" / "parquet" / "nested_2024.parquet")
    assert back["participants"].dtype == pl.String
    assert "rusher" in back["participants"].item()
