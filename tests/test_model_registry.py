"""The model registry and the numbered model stages must describe each other.

A retrain recipe referenced by nothing is how `retrain_xg_models.R` got
stranded: the registry claimed a cadence, but no stage could run it. These
checks close that loop from both directions.

Scope is deliberately the PACKAGE, not the exact command string. The registry's
`fitting script` cells are prose (`model_training/train_ep.py (train-ep)`,
`python -m cfb_model_build.cpoe --loso`, `cfb_model_publish ratings`), and asserting on prose
would fail on a reworded cell while missing a genuinely stranded model.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY_DIR = REPO / "python"
#: The registry moved out of CLAUDE.md (2026-08-28). A table this test parses
#: is repository data, not agent instructions. `models/` rather than
#: `docs/models/` because `cfb_model_reports` regenerates the latter and
#: overwrites its README on every run.
REGISTRY = REPO / "models" / "REGISTRY.md"

#: Stages that legitimately have NO registry row, each with its reason. Keep this
#: as tight as possible: every name here is a stage the "no stage without a
#: registry row" check can no longer catch.
#:
#: The bar is "does it PUBLISH a model artifact", not "is it in the pipeline".
NON_PUBLISHING_STAGES: dict[str, str] = {
    "cfb_model_build.cfb_model_reports": (
        "Report stage: emits model cards and evaluation reports ABOUT artifacts "
        "other stages publish. It ships no model of its own, so a registry row "
        "would have nothing to put in artifact/gates/cadence."
    ),
    "cfb_model_build.cfb_higher_models": (
        "Research package. Its outputs are experiment records -- pregame_fit.json, "
        "gbm_tuning.json, game_heads.json, experiments.json -- not published "
        "artifacts; nothing downstream consumes them. It earns a registry row the "
        "day one of its models is published, and not before: inventing "
        "training-data/gates/cadence cells for an unpublished model is worse than "
        "the gap."
    ),
}


#: Numbered model stage shims declare the package they forward to.
STAGE_RE = re.compile(r"^cfb_model_(?P<num>\d{2})_(?P<name>.+)_creation\.py$")
PACKAGE_RE = re.compile(r'^PACKAGE = "(?P<pkg>[a-z_.]+)"$', re.M)


def _stages() -> dict[str, str]:
    """package -> stage filename, for every numbered model shim."""
    out: dict[str, str] = {}
    for path in sorted(PY_DIR.glob("cfb_model_*_creation.py")):
        if not STAGE_RE.match(path.name):
            continue
        m = PACKAGE_RE.search(path.read_text(encoding="utf-8"))
        assert m, f"{path.name} declares no PACKAGE"
        pkg = m.group("pkg")
        # A dict would keep the LAST shim declaring a package and silently drop the
        # first, so two stages pointing at one package would read as one stage and
        # every check below would pass on a pipeline that is actually mis-wired.
        assert pkg not in out, (
            f"{path.name} and {out[pkg]} both declare PACKAGE {pkg!r}; "
            "a package must have exactly one numbered stage."
        )
        out[pkg] = path.name
    return out


def _registry_rows() -> list[str]:
    """The registry table rows.

    The file IS the registry now, rather than a section inside a larger document,
    so every pipe-table row in it belongs to the registry.
    """
    text = REGISTRY.read_text(encoding="utf-8")
    return [
        ln for ln in text.splitlines() if ln.startswith("|") and not set(ln) <= set("|- ")
    ][1:]  # drop the header row


MODEL_PKG = "cfb_model_build"


def _packages_on_disk() -> set[str]:
    """Model packages, as the shims name them: ``cfb_model_build.<family>``.

    They live under one package now (mirroring ``cfb_data_build`` on the dataset
    side), so a stage's PACKAGE is dotted and the bare family name alone is no
    longer importable.
    """
    root = PY_DIR / MODEL_PKG
    return {
        f"{MODEL_PKG}.{p.name}"
        for p in root.iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
    }


def test_parsers_find_something():
    """Guard the guard -- an empty parse would pass everything below."""
    assert _stages(), "no numbered model stages found"
    assert _registry_rows(), "model registry parsed empty"
    assert _packages_on_disk(), "no packages found under python/"


def test_every_stage_forwards_to_a_real_package():
    missing = {pkg: f for pkg, f in _stages().items() if pkg not in _packages_on_disk()}
    assert not missing, f"stages forwarding to a package that does not exist: {missing}"


def test_every_registry_model_is_reachable_from_a_stage():
    """A registry row whose package no stage exposes is a stranded retrain recipe."""
    stages, on_disk = _stages(), _packages_on_disk()
    rows, stranded = _registry_rows(), []
    for row in rows:
        # Registry cells cite the FAMILY name as an operator types it
        # (`model_training/train_ep.py`, `python -m cpoe --loso`), not the
        # dotted import path, so match on the last segment.
        cited = {p for p in on_disk if re.search(rf"\b{re.escape(p.split(".")[-1])}\b", row)}
        if not cited:
            continue  # row cites no in-repo package (e.g. an sdv-py entry point)
        if not (cited & set(stages)):
            model = row.split("|")[1].strip()
            stranded.append((model, sorted(cited)))
    assert not stranded, (
        "registry models whose package no numbered stage exposes: "
        f"{stranded}\nAdd the stage, or the retrain recipe is unrunnable."
    )


def test_no_model_stage_absent_from_the_registry():
    """The inverse: a stage nothing in the registry claims is undocumented output."""
    rows = "\n".join(_registry_rows())
    orphans = sorted(
        f"{f} ({pkg})"
        for pkg, f in _stages().items()
        if pkg not in NON_PUBLISHING_STAGES
        and not re.search(rf"\b{re.escape(pkg.split(chr(46))[-1])}\b", rows)
    )
    assert not orphans, (
        f"model stages no registry row mentions: {orphans}\n"
        "Every published artifact needs a registry row."
    )


def test_non_publishing_exemptions_are_live():
    """An exemption for a stage that no longer exists is dead weight that would
    silently cover a future stage of the same name."""
    stale = sorted(set(NON_PUBLISHING_STAGES) - set(_stages()))
    assert not stale, f"NON_PUBLISHING_STAGES exempts {stale}, but no such stage exists."
