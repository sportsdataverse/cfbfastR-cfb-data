# CFB Model Card Contract Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every model in the `cfb_model_artifacts` bundle carry a complete, machine-readable contract — a card for all nine models, with `era_contract` populated wherever the model consumes an era feature — so downstream calculators can validate inputs against the published artifact instead of restating the rules.

**Architecture:** Add a `rebuild-cards` subcommand to `cfb_model_build.model_training.cli` that loads each shipped `.ubj`, introspects its feature names, and rewrites the card via the existing `write_xgb_model_card()`. No retraining and no model bytes change — only the `.card.json` siblings. Then republish the bundle with a bumped `model_version`.

**Tech Stack:** Python 3.13, xgboost 3.2.0, polars, pytest, uv, `gh` CLI for release upload.

**Spec:** `sportsdataverse-py/docs/superpowers/specs/2026-09-09-cfb-model-calculators-design.md`

## Global Constraints

- **polars 1.x modern API only** — `group_by`, `with_row_index`, `pl.len()`, `map_elements(..., return_dtype=)`. Bool masks explicit.
- **uv for everything** — `uv run pytest`, `uv run ruff check`. Never bare `python`/`pip`.
- **`uv run` can silently re-lock `uv.lock`** — check `git status` after; never let a lockfile bump ride into an unrelated commit.
- **Never add AI co-author trailers or attribution footers** on any commit or PR.
- **Stage explicit paths** — no blind `git add -A`. Branch + PR; never push `main` directly.
- **`ERA_BOUNDS = (2006, 2013, 2020)`** and `ERA_ONEHOT_COLS = ["era0","era1","era2","era3"]` — copied verbatim from `model_training/constants.py`. These are the trainer's values and the only correct ones; cfb-data#70 was caused by consumers keeping a 2017 copy.
- **Model bytes must not change.** This plan rewrites `.card.json` files only. Any diff to a `.ubj` is a plan failure.

---

## Context an implementer needs

The bundle at release tag `cfb_model_artifacts` (repo `sportsdataverse/sportsdataverse-data`) ships nine models. Seven have cards; `cfb_cp_model` and `fd_model` have none. All seven existing cards were written **2026-08-02**, before `_era_contract()` was added on **2026-09-07** (PR #71), so all seven carry `era_contract: null` — including `fg_model` and `qbr_model`, which consume one-hot `era0..era3`, and `two_pt_model` and `xpass_model`, which consume ordinal `era`.

`era_contract: null` is **correct** for `ep_model` and the two WP models: they have no era feature. The bug is only on the four that do.

`write_xgb_model_card(model_path, *, model_type, label, features=None, model=None, ...)` already derives the contract from the feature list via `_era_contract(features)`, and `_introspect_features(model)` reads `booster.feature_names`. So a card can be rebuilt from a saved `.ubj` alone.

## File Structure

| File | Responsibility |
|---|---|
| `python/cfb_model_build/model_training/rebuild_cards.py` (create) | Load a `.ubj`, introspect features, rewrite its card. One public function. |
| `python/cfb_model_build/model_training/cli.py` (modify) | Add the `rebuild-cards` subcommand wiring only. |
| `tests/model_training/test_rebuild_cards.py` (create) | Contract tests for the rebuild, incl. the era-consuming models. |

---

### Task 1: Card rebuild from a saved booster

**Files:**
- Create: `python/cfb_model_build/model_training/rebuild_cards.py`
- Test: `tests/model_training/test_rebuild_cards.py`

**Interfaces:**
- Consumes: `write_xgb_model_card` and `_introspect_features` from `model_training.model_card`; `ERA_BOUNDS`, `ERA_ONEHOT_COLS` from `model_training.constants`.
- Produces: `rebuild_card(model_path: Path, *, model_type: str, label: str) -> Path` returning the written card path.

- [ ] **Step 1: Write the failing test**

```python
"""Rebuilding a card from a shipped booster must recover the era contract.

The seven published cards were written 2026-08-02, before `_era_contract()`
landed on 2026-09-07 (#71), so every one of them carries `era_contract: null` --
including fg_model and qbr_model, which consume one-hot era0..era3. Nothing is
retrained here; only the .card.json sibling is rewritten.
"""

from __future__ import annotations

import json

import pytest
import xgboost as xgb

from cfb_model_build.model_training import constants as C
from cfb_model_build.model_training.rebuild_cards import rebuild_card


def _booster(tmp_path, name, features):
    """A minimal real Booster carrying `features` as its feature_names."""
    import numpy as np

    X = np.zeros((4, len(features)), dtype=float)
    y = np.array([0, 1, 0, 1])
    dm = xgb.DMatrix(X, label=y, feature_names=list(features))
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 1}, dm, num_boost_round=1)
    p = tmp_path / f"{name}.ubj"
    bst.save_model(str(p))
    return p


def test_rebuild_recovers_the_one_hot_era_contract(tmp_path):
    p = _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    card_path = rebuild_card(p, model_type="fg", label="made")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    era = card["era_contract"]
    assert era["encoding"] == "one_hot"
    assert era["columns"] == C.ERA_ONEHOT_COLS
    assert era["cuts"] == list(C.ERA_BOUNDS)


def test_rebuild_recovers_the_ordinal_era_contract(tmp_path):
    p = _booster(tmp_path, "xpass_model", ["down", "distance", "era"])
    card = json.loads(rebuild_card(p, model_type="xpass", label="pass").read_text(encoding="utf-8"))
    assert card["era_contract"]["encoding"] == "ordinal"
    assert card["era_contract"]["cuts"] == list(C.ERA_BOUNDS)


def test_a_model_without_era_gets_no_contract(tmp_path):
    """`era_contract: null` is CORRECT for ep_model and the WP pair."""
    p = _booster(tmp_path, "ep_model", ["TimeSecsRem", "yards_to_goal", "distance"])
    card = json.loads(rebuild_card(p, model_type="ep", label="next_score_label").read_text(encoding="utf-8"))
    assert "era_contract" not in card or card["era_contract"] is None


def test_features_are_taken_from_the_booster_in_order(tmp_path):
    """Feature ORDER is load-bearing for XGBoost; the card is what fixes it."""
    feats = ["down", "distance", "yards_to_goal", "era"]
    p = _booster(tmp_path, "xpass_model", feats)
    card = json.loads(rebuild_card(p, model_type="xpass", label="pass").read_text(encoding="utf-8"))
    assert card["features"] == feats


def test_a_booster_without_feature_names_is_rejected(tmp_path):
    """A card with no features would validate nothing downstream."""
    import numpy as np

    dm = xgb.DMatrix(np.zeros((4, 2)), label=np.array([0, 1, 0, 1]))
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 1}, dm, num_boost_round=1)
    p = tmp_path / "nameless.ubj"
    bst.save_model(str(p))
    with pytest.raises(ValueError, match="no feature names"):
        rebuild_card(p, model_type="mystery", label="y")


def test_the_model_file_is_not_modified(tmp_path):
    """This plan rewrites cards only; any .ubj diff is a failure."""
    p = _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    before = p.read_bytes()
    rebuild_card(p, model_type="fg", label="made")
    assert p.read_bytes() == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_rebuild_cards.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cfb_model_build.model_training.rebuild_cards'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Rebuild a model card from a saved booster, without retraining.

The cards in the shipped `cfb_model_artifacts` bundle were written 2026-08-02,
before `_era_contract()` landed (#71), so the four era-consuming models publish
`era_contract: null` and every consumer keeps its own private copy of the era
cuts instead. That duplication is what caused cfb-data#70. Rebuilding the card
from the booster's own `feature_names` republishes the contract without touching
a single model byte.
"""

from __future__ import annotations

from pathlib import Path

import xgboost as xgb

from cfb_model_build.model_training.model_card import _introspect_features, write_xgb_model_card


def rebuild_card(model_path: Path | str, *, model_type: str, label: str) -> Path:
    """Rewrite ``<model_path>.json`` from the booster's own feature names.

    Args:
        model_path: Path to a saved ``.ubj`` booster.
        model_type: Card ``model_type`` (e.g. ``"fg"``, ``"xpass"``).
        label: Training label name recorded on the card.

    Returns:
        Path to the written card.

    Raises:
        ValueError: If the booster carries no feature names — a card with no
            features would validate nothing for a downstream caller.
    """
    model_path = Path(model_path)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    features = _introspect_features(booster)
    if not features:
        raise ValueError(f"{model_path.name} carries no feature names; cannot write a usable card")
    return write_xgb_model_card(
        model_path,
        model_type=model_type,
        label=label,
        features=features,
        model=booster,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/sdv_repos/cfbfastR-cfb-data && uv run pytest tests/model_training/test_rebuild_cards.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Lint and check the lockfile did not drift**

Run: `uv run ruff check python/cfb_model_build/model_training/rebuild_cards.py tests/model_training/test_rebuild_cards.py && git status --short uv.lock`
Expected: `All checks passed!` and no `uv.lock` in the status output. If `uv.lock` changed, run `git checkout uv.lock`.

- [ ] **Step 6: Commit**

```bash
git add python/cfb_model_build/model_training/rebuild_cards.py tests/model_training/test_rebuild_cards.py
git commit -m "feat(model-card): rebuild a card from a saved booster

The seven published cards were written 2026-08-02, before _era_contract()
landed on 2026-09-07 (#71), so all seven carry era_contract: null -- including
fg_model and qbr_model, which consume one-hot era0..era3, and two_pt_model and
xpass_model, which consume ordinal era. Consumers therefore still keep private
copies of the era cuts, which is what caused #70.

rebuild_card() reads the booster's own feature_names and rewrites the card
through the existing writer, so the contract is republished without retraining
and without touching any model bytes."
```

---

### Task 2: `rebuild-cards` CLI subcommand

**Files:**
- Modify: `python/cfb_model_build/model_training/cli.py`
- Test: `tests/model_training/test_rebuild_cards.py` (append)

**Interfaces:**
- Consumes: `rebuild_card()` from Task 1.
- Produces: `python -m cfb_model_build.model_training rebuild-cards --artifacts-dir DIR` rewriting every card in a directory; exits non-zero if any model is unreadable.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_rebuilds_every_booster_in_a_directory(tmp_path):
    """One command must cover all nine bundle models, not one at a time."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    _booster(tmp_path, "xpass_model", ["down", "distance", "era"])
    written = rebuild_all(tmp_path)
    assert len(written) == 2
    names = sorted(p.name for p in written)
    assert names == ["fg_model.card.json", "xpass_model.card.json"]
    for p in written:
        assert json.loads(p.read_text(encoding="utf-8"))["era_contract"] is not None


def test_rebuild_all_reports_a_model_it_cannot_read(tmp_path):
    """A silently skipped model would republish an incomplete bundle."""
    from cfb_model_build.model_training.rebuild_cards import rebuild_all

    _booster(tmp_path, "fg_model", ["yards_to_goal", *C.ERA_ONEHOT_COLS])
    (tmp_path / "broken.ubj").write_bytes(b"not a booster")
    with pytest.raises(RuntimeError, match="broken.ubj"):
        rebuild_all(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_rebuild_cards.py -q -k "cli_rebuilds or cannot_read"`
Expected: FAIL — `ImportError: cannot import name 'rebuild_all'`

- [ ] **Step 3: Write minimal implementation**

Append to `python/cfb_model_build/model_training/rebuild_cards.py`:

```python
#: Card ``model_type`` and training label per bundle asset. Derived from the
#: published cards; `cfb_cp_model` and `fd_model` ship no card at all, which is
#: why their entries are declared here rather than read back.
BUNDLE_MODELS: dict[str, tuple[str, str]] = {
    "ep_model": ("ep", "next_score_label"),
    "fg_model": ("fg", "made"),
    "wp_naive": ("wp_naive", "win"),
    "wp_spread": ("wp_spread", "win"),
    "cfb_cp_model": ("cp", "complete"),
    "xpass_model": ("xpass", "pass"),
    "two_pt_model": ("two_pt", "success"),
    "fd_model": ("fourth_down", "conversion"),
    "qbr_model": ("qbr", "qbr"),
}


def rebuild_all(artifacts_dir: Path | str) -> list[Path]:
    """Rebuild the card for every ``.ubj`` in ``artifacts_dir``.

    Raises:
        RuntimeError: If any booster cannot be read. A skipped model would
            republish an incomplete bundle, which is the failure this exists to
            prevent -- so one bad file fails the run rather than being logged.
    """
    artifacts_dir = Path(artifacts_dir)
    written: list[Path] = []
    for ubj in sorted(artifacts_dir.glob("*.ubj")):
        model_type, label = BUNDLE_MODELS.get(ubj.stem, (ubj.stem, "y"))
        try:
            written.append(rebuild_card(ubj, model_type=model_type, label=label))
        except Exception as exc:  # noqa: BLE001 - re-raised with the file name
            raise RuntimeError(f"could not rebuild card for {ubj.name}: {exc}") from exc
    return written
```

Then add the subcommand in `cli.py`, beside the existing `sub.add_parser(...)` calls:

```python
    rc = sub.add_parser("rebuild-cards", help="rewrite model cards from saved boosters")
    rc.add_argument("--artifacts-dir", required=True, help="directory holding the .ubj files")
```

and in the dispatch body:

```python
    if args.cmd == "rebuild-cards":
        from cfb_model_build.model_training.rebuild_cards import rebuild_all

        for p in rebuild_all(args.artifacts_dir):
            print(f"wrote {p}")
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_rebuild_cards.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Verify the CLI end to end**

Run: `PYTHONPATH=python uv run python -m cfb_model_build.model_training rebuild-cards --help`
Expected: usage text showing `--artifacts-dir`.

- [ ] **Step 6: Commit**

```bash
git add python/cfb_model_build/model_training/rebuild_cards.py python/cfb_model_build/model_training/cli.py tests/model_training/test_rebuild_cards.py
git commit -m "feat(cli): add rebuild-cards for the whole artifacts directory

One command covers all nine bundle models. A booster that cannot be read fails
the run rather than being skipped: a silently skipped model would republish an
incomplete bundle, which is precisely the state this work exists to fix."
```

---

### Task 3: Rebuild the real bundle and verify the contract

**Files:**
- No source changes. Produces regenerated `.card.json` files in a scratch directory.

**Interfaces:**
- Consumes: `rebuild-cards` CLI from Task 2.
- Produces: nine verified cards ready to upload in Task 4.

- [ ] **Step 1: Download the shipped bundle**

```bash
mkdir -p /tmp/cfb_bundle && cd /tmp/cfb_bundle
gh release download cfb_model_artifacts --repo sportsdataverse/sportsdataverse-data --clobber
ls *.ubj | wc -l   # expect 9
```

- [ ] **Step 2: Record the model checksums BEFORE**

```bash
sha256sum /tmp/cfb_bundle/*.ubj | sort > /tmp/ubj_before.txt
cat /tmp/ubj_before.txt
```

- [ ] **Step 3: Rebuild every card**

```bash
cd /mnt/sdv_repos/cfbfastR-cfb-data
# PYTHONPATH=python because the package lives under python/, matching this
# repo's own convention in scripts/cfb_models.sh. Without it the module is
# not importable and the command fails with ModuleNotFoundError.
PYTHONPATH=python uv run python -m cfb_model_build.model_training rebuild-cards --artifacts-dir /tmp/cfb_bundle
```
Expected: nine `wrote ...card.json` lines, including `cfb_cp_model.card.json` and `fd_model.card.json`, which did not exist before.

- [ ] **Step 4: Verify no model byte changed**

```bash
sha256sum /tmp/cfb_bundle/*.ubj | sort > /tmp/ubj_after.txt
diff /tmp/ubj_before.txt /tmp/ubj_after.txt && echo "MODELS UNCHANGED"
```
Expected: `MODELS UNCHANGED`. Any diff is a plan failure — stop and investigate.

- [ ] **Step 5: Verify the era contract is now published where it belongs**

```bash
cd /tmp/cfb_bundle && python3 - <<'PY'
import json, glob, os
# Read from the shipped boosters' own feature_names, not assumed:
# fd_model consumes era0..era3 (one-hot) and cfb_cp_model has no era feature.
ERA_USERS = {"fg_model", "qbr_model", "two_pt_model", "xpass_model", "fd_model"}
NO_ERA = {"ep_model", "wp_naive", "wp_spread", "cfb_cp_model"}
seen = set()
for f in sorted(glob.glob("*.card.json")):
    stem = os.path.basename(f).replace(".card.json", "")
    seen.add(stem)
    card = json.load(open(f))
    era = card.get("era_contract")
    if stem in ERA_USERS:
        assert era, f"{stem}: era_contract still null"
        assert era["cuts"] == [2006, 2013, 2020], f"{stem}: wrong cuts {era['cuts']}"
        print(f"  {stem:14s} {era['encoding']:8s} cuts={era['cuts']}")
    elif stem in NO_ERA:
        assert not era, f"{stem}: unexpected era_contract"
        print(f"  {stem:14s} (no era feature -- correct)")
    else:
        print(f"  {stem:14s} era={'yes' if era else 'no'}")
missing = {"cfb_cp_model", "fd_model"} - seen
assert not missing, f"still cardless: {missing}"
print(f"  cards present: {len(seen)} (expect 9)")
PY
```
Expected: `fg_model one_hot cuts=[2006, 2013, 2020]`, `qbr_model one_hot`, `fd_model one_hot`, `two_pt_model ordinal`, `xpass_model ordinal`, the four no-era models marked correct, and `cards present: 9`. Every one of the nine falls in `ERA_USERS` or `NO_ERA`, so the `else` branch should never print.

- [ ] **Step 6: No commit**

Nothing in the repo changed in this task. The verified cards in `/tmp/cfb_bundle` are the input to Task 4.

---

### Task 4: Republish the bundle with a bumped model_version

**Files:**
- Modify: `/tmp/cfb_bundle/MANIFEST.json` (uploaded, not committed)

**Interfaces:**
- Consumes: verified cards from Task 3.
- Produces: the `cfb_model_artifacts` release carrying nine cards and a new `model_version`, which the Phase B calculators read.

- [ ] **Step 1: Bump the manifest**

```bash
cd /tmp/cfb_bundle && python3 - <<'PY'
import json, datetime
m = json.load(open("MANIFEST.json"))
old = m["model_version"]
m["model_version"] = datetime.date.today().isoformat().replace("-", ".")
m["generated"] = datetime.date.today().isoformat()
m.setdefault("notes", []).append(
    "Cards rebuilt from the shipped boosters to publish era_contract on the four "
    "era-consuming models and to add the previously missing cfb_cp_model and "
    "fd_model cards. No model was retrained; .ubj checksums are unchanged."
)
json.dump(m, open("MANIFEST.json", "w"), indent=1)
print(f"  model_version {old} -> {m['model_version']}")
PY
```

- [ ] **Step 2: Confirm with the user before uploading**

Uploading to a release is outward-facing and cannot be undone silently. Show the user the nine card names, the `model_version` change, and the `MODELS UNCHANGED` result from Task 3 Step 4, and get an explicit go-ahead.

- [ ] **Step 3: Upload the cards and manifest**

```bash
cd /tmp/cfb_bundle
gh release upload cfb_model_artifacts *.card.json MANIFEST.json \
  --repo sportsdataverse/sportsdataverse-data --clobber
```

- [ ] **Step 4: Verify the published contract from a clean fetch**

```bash
curl -sL "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_model_artifacts/fg_model.card.json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  fg era_contract:', json.dumps(d['era_contract']))"
curl -sL -o /dev/null -w "  fd_model.card.json HTTP %{http_code}\n" \
  "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_model_artifacts/fd_model.card.json"
```
Expected: a populated `era_contract` with `cuts: [2006, 2013, 2020]`, and `HTTP 200` for the previously missing card.

- [ ] **Step 5: Record the rebuild in the model registry**

Add a line to `models/REGISTRY.md` under the bundle's row noting the card rebuild date and that no model was retrained. Then:

```bash
git add models/REGISTRY.md
git commit -m "docs(registry): record the model-card contract rebuild

Cards regenerated from the shipped boosters so the four era-consuming models
publish era_contract and cfb_cp_model/fd_model are no longer cardless. No model
retrained; .ubj checksums verified unchanged before upload."
```

- [ ] **Step 6: Open the PR**

```bash
git push -u origin fix/model-card-contract-completion
gh pr create --repo sportsdataverse/cfbfastR-cfb-data --base main \
  --title "fix(model-card): publish a complete contract for all nine CFB models" \
  --body "See docs/superpowers/plans/2026-09-09-cfb-model-card-contract-completion.md.

Adds \`rebuild-cards\`, regenerates every card from the shipped boosters, and republishes the bundle so:

- \`cfb_cp_model\` and \`fd_model\` have cards for the first time
- \`fg_model\`, \`qbr_model\`, \`two_pt_model\` and \`xpass_model\` publish \`era_contract\` (they carried \`era_contract: null\` because the cards were written 2026-08-02, before \`_era_contract()\` landed on 2026-09-07 in #71)

\`ep_model\`, \`wp_naive\` and \`wp_spread\` keep \`era_contract: null\` — correctly, they have no era feature.

No model was retrained. \`.ubj\` checksums verified byte-identical before upload.

Unblocks the CFB calculator surface, where consumers validate against the published card instead of keeping private copies of the era cuts — the duplication that caused #70."
```

---

## Self-Review

**Spec coverage.** This plan implements the spec's "Prerequisites (cfbfastR-cfb-data, first)" section in full: cards for `cfb_cp_model` and `fd_model` (Tasks 2-3), `era_contract` on the four era-consuming cards (Tasks 1, 3), and the bundle republish with a bumped `model_version` (Task 4). The spec's Architecture, Error handling and Testing sections belong to Plans 2 and 3 and are deliberately not covered here.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable code; every verification step carries the exact command and its expected output.

**Type consistency.** `rebuild_card(model_path, *, model_type, label) -> Path` is defined in Task 1 and used unchanged in Task 2's `rebuild_all`. `rebuild_all(artifacts_dir) -> list[Path]` is defined in Task 2 and used by the CLI dispatch in the same task. `BUNDLE_MODELS` is defined once and read once.

**Known deviation from TDD, stated deliberately.** Tasks 3 and 4 are operational (download, verify, upload) and carry verification commands with expected output rather than unit tests. The unit-testable logic all lives in Tasks 1-2.

## Follow-on plans

- **Plan 2 (sportsdataverse-py):** feature-derivation layer extracted from `cfb_pbp.py`, `predict_from_card`, ten `calculate_*` functions. Blocked on this plan, because the derivation layer reads `era_contract` from the card.
- **Plan 3 (cfbfastR):** the R port against the same cards, plus the R↔Python parity fixture.
