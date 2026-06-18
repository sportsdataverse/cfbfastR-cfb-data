import os
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from pregame_wp.cli import _read_boxes, build_parser


def test_subcommands_present():
    p = build_parser()
    choices = set(p._subparsers._group_actions[0].choices.keys())  # type: ignore[attr-defined]
    expected = {"build-boxes", "train", "predict-matchup"}
    assert expected <= choices


def test_help_exits_zero():
    python_dir = str(pathlib.Path(__file__).resolve().parents[2])  # == python/ (packages root)
    env = {**os.environ, "PYTHONPATH": python_dir}
    result = subprocess.run(
        [sys.executable, "-m", "pregame_wp", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "build-boxes" in result.stdout


def test_build_boxes_args_parse():
    p = build_parser()
    ns = p.parse_args([
        "build-boxes", "--seasons", "2018:2019",
        "--raw-dir", "raw/", "--out", "boxes/", "--no-fetch", "--quiet",
    ])
    assert ns.cmd == "build-boxes"
    assert ns.seasons == "2018:2019"
    assert ns.raw_dir == "raw/"
    assert ns.no_fetch is True
    assert ns.quiet is True


def test_predict_matchup_args_parse():
    p = build_parser()
    ns = p.parse_args([
        "predict-matchup", "--home", "LSU", "--away", "Clemson",
        "--year", "2019", "--week", "-1", "--neutral-site", "--covid",
    ])
    assert ns.cmd == "predict-matchup"
    assert ns.home == "LSU"
    assert ns.away == "Clemson"
    assert ns.year == 2019
    assert ns.neutral_site is True
    assert ns.covid is True


def test_read_boxes_single_file(tmp_path):
    df = pd.DataFrame({"Team": ["A", "B"], "5FR": [8.0, 4.0]})
    p = tmp_path / "box-scores.parquet"
    df.to_parquet(p, index=False)
    out = _read_boxes(str(p))
    assert len(out) == 2
    assert set(out["Team"]) == {"A", "B"}


def test_read_boxes_directory_concats(tmp_path):
    for i, team in enumerate(["A", "B"]):
        pd.DataFrame({"Team": [team], "5FR": [float(i)]}).to_parquet(
            tmp_path / f"part{i}.parquet", index=False
        )
    out = _read_boxes(str(tmp_path))
    assert len(out) == 2


def test_read_boxes_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _read_boxes(str(tmp_path / "nope.parquet"))
