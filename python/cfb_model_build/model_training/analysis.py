"""Per-model analysis frames: play ids beside the EXACT engineered feature matrix each
trainer fits (``export-analysis``).

``docs/models/deepdive.qmd`` attributes the promoted boosters with TreeSHAP. EP's features
are raw ``pbp_full`` columns, but WP/CP/xPass fit engineered ones (``spread_time``,
``adj_TimeSecsRem``, ``score_diff``, ``era``, ...) that only exist inside this package's
feature builders -- so the trainer exports them once, through the same ``*_matrix`` /
``extract_pass_features`` code paths the fits use, and the page reads the frames instead
of re-deriving anything. Output is build-tree only (``python/artifacts/analysis/``, never
committed; ``analysis_manifest.json`` records the source frame, row counts and column
order per model).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import constants as C
from .features import ep_matrix, wp_matrix, xpass_frame
from .ingest import add_winner

MODELS: tuple[str, ...] = ("ep", "wp", "xpass", "cp")
ID_COLS: list[str] = ["game_id", "id", "season", "week", "pos_team"]  # no feature names here (period is a WP/xpass/CP feature)

# Columns the four builders read from pbp_full (a 478-column frame): read only these.
_CP_SOURCE = [
    "start.down",
    "start.distance",
    "start.yardsToEndzone",
    "pos_score_diff_start",
    "start.TimeSecsRem",
    "start.is_home",
    "period",
    "passing_down",
    "type.text",
    "completion",
]
_XPASS_SOURCE = [
    "rush",
    "pass",
    "start.down",
    "start.distance",
    "start.yardsToEndzone",
    "pos_score_diff_start",
    "start.TimeSecsRem",
    "season",
    "period",
]
_WINNER_SOURCE = ["homeScore", "awayScore", "homeTeamName", "awayTeamName", "start.pos_team.name"]


def needed_columns() -> list[str]:
    cols = ID_COLS + ["label", "ScoreDiff_W"] + list(C.EP_SOURCE.values()) + list(C.WP_SOURCE.values())
    cols += _CP_SOURCE + _XPASS_SOURCE + _WINNER_SOURCE
    return list(dict.fromkeys(cols))


def read_source(pbp: str | Path) -> pl.DataFrame:
    """The training frame, restricted to the columns the builders touch."""
    lf = pl.scan_parquet(pbp)
    have = set(lf.collect_schema().names())
    df = lf.select([c for c in needed_columns() if c in have]).collect()
    # The play id is Int64 everywhere else in the chain (model_pbp, load_cfb_pbp) but String in
    # pbp_full -- pin it once here, strictly: a non-numeric id is a parser regression to surface.
    return df.with_columns(pl.col("id").cast(pl.Int64)) if "id" in df.columns else df


def _ids(df: pl.DataFrame) -> list[str]:
    return [c for c in ID_COLS if c in df.columns]


def _beside(df: pl.DataFrame, X, label: pl.Series) -> pl.DataFrame:
    """ids | trainer matrix (row-aligned, unfiltered builders) | label."""
    # hstack raises on a height mismatch where concat(how="horizontal") would pad with nulls
    assert X.shape[0] == df.height, f"builder is not row-aligned: {X.shape[0]} vs {df.height}"
    return df.select(_ids(df)).hstack(pl.from_pandas(X)).with_columns(label.alias("label"))


def build_frames(df: pl.DataFrame, models: tuple[str, ...] = MODELS) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    if "ep" in models:
        X, y, _ = ep_matrix(df)
        out["ep"] = _beside(df, X, pl.Series(y))
    if "wp" in models:
        if "winner" not in df.columns:
            df = add_winner(df)
        # one frame serves both variants: naive = the spread set minus spread_time
        X, y, _ = wp_matrix(df, variant="spread")
        out["wp"] = _beside(df, X, pl.Series(y))
    if "xpass" in models:
        f = xpass_frame(df)
        out["xpass"] = f.select(_ids(f) + C.XPASS_FEATURES + [pl.col("pass").cast(pl.Int32).alias("label")])
    if "cp" in models:
        from cfb_model_build.cpoe.constants import FEATURE_COLS, TARGET_COL
        from cfb_model_build.cpoe.features import extract_pass_features

        feats = pl.from_pandas(extract_pass_features(df))  # pass plays only; keeps game_id/id/season
        ids = [c for c in ID_COLS if c in feats.columns]
        out["cp"] = feats.select(ids + FEATURE_COLS + [pl.col(TARGET_COL).cast(pl.Int32).alias("label")])
    return out


def export_analysis_frames(pbp: str | Path, out_dir: str | Path, models: tuple[str, ...] = MODELS) -> dict[str, int]:
    """Write ``analysis_<model>.parquet`` per model + ``analysis_manifest.json``; return row counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = build_frames(read_source(pbp), models)
    h = hashlib.sha256()
    with open(pbp, "rb") as fh:  # ties the frames to the exact bytes the boosters' cards name
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    manifest: dict[str, object] = {
        "source_frame": str(pbp),
        "source_sha256": h.hexdigest(),
        "source_bytes": Path(pbp).stat().st_size,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {},
    }
    for name, fr in frames.items():
        fr.write_parquet(out_dir / f"analysis_{name}.parquet")
        ids = [c for c in ID_COLS if c in fr.columns]
        manifest["models"][name] = {  # type: ignore[index]
            "n_rows": fr.height,
            "id_cols": ids,
            "features": [c for c in fr.columns if c not in ids and c != "label"],
            "label": "label",
        }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {k: v.height for k, v in frames.items()}
