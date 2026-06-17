from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelArtifact:
    model_type: str
    model_path: Path
    card_path: Path
    card: dict


def discover_models(artifacts_dir: str | Path) -> list[ModelArtifact]:
    out = []
    for card_path in sorted(Path(artifacts_dir).rglob("*.json")):
        try:
            card = json.loads(card_path.read_text())
        except Exception:  # noqa: BLE001 — skip unreadable cards
            continue
        ubj = card_path.with_suffix(".ubj")
        pkl = card_path.with_suffix(".pkl")
        if ubj.exists():
            model_path = ubj
        elif pkl.exists():
            model_path = pkl
        else:
            continue  # card with no model sibling -> skip
        model_type = card.get("model_type") or ("rb_eval" if model_path.suffix == ".pkl" else model_path.stem)
        out.append(ModelArtifact(model_type=model_type, model_path=model_path, card_path=card_path, card=card))
    return out
