"""The numbered shims must build the dataset their filename claims."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY_DIR = REPO / "python"
sys.path.insert(0, str(PY_DIR))

import _shim  # noqa: E402

from cfb_data_build.cli import build_parser  # noqa: E402


def _shims():
    return sorted(PY_DIR.glob("espn_cfb_*_creation.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli_choices() -> set[str]:
    for action in build_parser()._actions:
        if action.dest == "dataset":
            return set(action.choices)
    raise AssertionError("cli parser has no --dataset action")


def test_shims_exist():
    """Guard the guard: an empty glob would pass everything below."""
    assert _shims(), "no numbered shims found"


def test_every_shim_targets_a_real_cli_dataset():
    """A shim naming a dataset the CLI cannot build fails only when someone runs it."""
    choices = _cli_choices()
    bad = []
    for path in _shims():
        mod = _load(path)
        names = [mod.DATASET] if hasattr(mod, "DATASET") else list(mod.ORDER)
        bad += [(path.name, n) for n in names if n not in choices]
    assert not bad, f"shims naming a dataset the CLI cannot build: {bad}"


@pytest.mark.parametrize(
    "argv",
    [
        ["--dataset", "betting"],
        ["--dataset=betting"],
        ["-s", "2026", "--dataset", "betting", "-e", "2026"],
    ],
)
def test_run_dataset_rejects_a_conflicting_selector(argv):
    """A shim forwards the caller's argv, so a caller's --dataset would reach
    argparse alongside the fixed one and the LAST would win -- the pbp shim would
    quietly build betting. Refuse rather than silently pick one."""
    with pytest.raises(SystemExit) as exc:
        _shim.run_dataset("pbp", argv)
    assert "pbp" in str(exc.value) and "betting" in str(exc.value)


def test_run_many_rejects_before_building_anything(monkeypatch):
    """The orchestrator must reject up front, not after building some of the ten."""
    called = []
    monkeypatch.setattr(_shim, "main", lambda a: called.append(a) or 0)
    with pytest.raises(SystemExit):
        _shim.run_many(["adv_team", "adv_passing"], ["--dataset", "pbp"])
    assert called == [], "rejected only after building; partial run already happened"


def test_run_dataset_forwards_season_args(monkeypatch):
    seen = []
    monkeypatch.setattr(_shim, "main", lambda a: seen.append(a) or 0)
    assert _shim.run_dataset("pbp", ["-s", "2026", "-e", "2026"]) == 0
    assert seen == [["--dataset", "pbp", "-s", "2026", "-e", "2026"]]
