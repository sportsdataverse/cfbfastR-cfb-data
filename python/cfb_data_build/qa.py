"""Report-only QA stage: per-game validation + pre-publish drift, published as ``espn_cfb_qa``.

Two gates, neither of which can fail a build in this revision
(:data:`BLOCKING` is ``False`` -- V2 ships report-only and ratchets):

* **per game** -- :func:`sportsdataverse.validation.validate_game` scores the
  enriched plays of each ``final.json`` (this repo reshapes, it does not
  enrich: ``cfbfastR-cfb-raw`` runs ``CFBPlayProcess``) against the packaged
  invariant rules. The report's
  :meth:`~sportsdataverse.validation.GameReport.to_row` is the row shape.
* **per season, before publish** -- :func:`drift_findings` compares the finished
  season parquet against the **previously published** asset: column set and
  dtypes (``schema_contract``), null-rate rises (``null_rate``), columns that
  went constant (``constant_column``) and numeric mean shifts
  (``rate_anomaly``). The previously published asset is the contract on
  purpose: it is what the loader reads today, and sdv-py's declared loader
  schemas live under ``tools/``, which the wheel does not ship.

Both land in the season's ``espn_cfb_qa_{season}`` asset -- the per-game rows in
the parquet, the aggregate + drift findings in the ``_summary.json`` sidecar
published beside it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

LEAGUE = "cfb"
SOURCE = "espn"

#: Share of a season's games allowed to carry an ``error`` finding.
#: Seeded from the V1 gate measurement on sdv-py ``main`` @1686f904f -- CFB
#: 21/45 games error-free (47%), so 0.53 -- and NOT from the spec's 0.5%
#: target, which every season would fail today. Lower it only with a ledger
#: entry as the open rules (``score.delta_value``, ``score.monotone``,
#: ``timeouts.*``, ``ep.*_range``, ...) close.
#:
#: Measured here on 20 real 2026 finals from ``cfbfastR-cfb-raw``: 0/20
#: error-free, dominated by ``timeouts.second_half_reset`` (20/20). Those
#: finals were enriched at the raw repo's own sdv-py pin, which is older than
#: the gate's -- the asset makes that lag visible, which is the point.
MAX_ERROR_SHARE = 0.53

#: Report-only. The build logs the summary and publishes the asset; it never
#: fails on QA. Flipping this to ``True`` is the ratchet step, its own PR.
BLOCKING = False

#: One row per validated game. ``failed_rule_ids`` / ``warned_rule_ids`` are
#: comma-joined rather than ``List(Utf8)`` so the same frame serialises to
#: parquet, csv and rds unchanged across the three release families.
QA_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64,
    "league": pl.Utf8,
    "source": pl.Utf8,
    "season": pl.Int64,
    "processing_version": pl.Utf8,
    "n_rows": pl.Int64,
    "ok": pl.Boolean,
    "n_errors": pl.Int64,
    "n_warnings": pl.Int64,
    "failed_rule_ids": pl.Utf8,
    "warned_rule_ids": pl.Utf8,
    "built_at": pl.Utf8,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def qa_row(
    plays: pl.DataFrame,
    *,
    processing_version: str,
    summary: dict[str, Any] | None = None,
    box: dict[str, Any] | None = None,
    header: dict[str, Any] | None = None,
    source: str = SOURCE,
    league: str = LEAGUE,
) -> dict[str, Any]:
    """Validate one processed game and flatten the report into a QA row."""
    from sportsdataverse.validation import validate_game

    report = validate_game(plays, league, header=header, source=source, summary=summary, box=box)
    row = report.to_row()
    row["failed_rule_ids"] = ",".join(row["failed_rule_ids"])
    row["warned_rule_ids"] = ",".join(row["warned_rule_ids"])
    # the library reports the installed sdv-py version; the asset records the
    # stamp the final itself carries, so a row says which library state
    # produced the plays it judged
    row["processing_version"] = processing_version
    row["built_at"] = _utc_now()
    return row


def game_qa_row(game: dict[str, Any], *, processing_version: str) -> dict[str, Any]:
    """Validate one ``final.json`` payload; an empty frame still yields a row."""
    plays = game.get("plays") or []
    frame = pl.from_dicts(plays, infer_schema_length=None) if plays else pl.DataFrame()
    return qa_row(
        frame,
        processing_version=processing_version,
        summary=game,
        box=game.get("advBoxScore"),
        header=game.get("header"),
    )


def _finding(check: str, severity: str, message: str, **kw: Any) -> dict[str, Any]:
    from sportsdataverse.validation.findings import Finding, Severity

    return Finding(
        check, Severity(severity), LEAGUE, kw.pop("dataset", "pbp"), message, **kw
    ).to_dict()


def drift_findings(new: pl.DataFrame, prev: pl.DataFrame | None) -> list[dict[str, Any]]:
    """Report-only drift of a finished season frame against the published one.

    ``prev is None`` (a first-published season, or an unreachable release) is
    not a finding: there is nothing to drift from. Returns the findings as
    dicts, in the harness ``Finding`` shape, for the summary sidecar.
    """
    if prev is None or new.height == 0:
        return []
    # The tolerances are sdv-py's packaged ``validation/thresholds.yaml``, read
    # through the constants #555 added so a caller needs no YAML parser. The
    # import is deferred, as every other sportsdataverse import in this module is.
    from sportsdataverse.validation.thresholds import MEAN_SHIFT_WARN, NULL_RATE_WARN

    null_warn, shift_warn = NULL_RATE_WARN, MEAN_SHIFT_WARN
    out: list[dict[str, Any]] = []
    new_s = {c: str(t) for c, t in new.schema.items()}
    prev_s = {c: str(t) for c, t in prev.schema.items()}

    for col in sorted(set(prev_s) - set(new_s)):
        out.append(
            _finding("schema_contract", "error", f"column {col!r} dropped since the last release",
                     locator={"column": col}, expected=prev_s[col], actual=None)
        )
    for col in sorted(set(new_s) - set(prev_s)):
        out.append(
            _finding("schema_contract", "warn", f"column {col!r} is new since the last release",
                     locator={"column": col}, expected=None, actual=new_s[col])
        )
    for col in sorted(set(new_s) & set(prev_s)):
        if new_s[col] != prev_s[col]:
            out.append(
                _finding("schema_contract", "error", f"dtype changed for {col!r}",
                         locator={"column": col}, expected=prev_s[col], actual=new_s[col])
            )

    shared = sorted(set(new_s) & set(prev_s))
    if shared:
        n_null = new.select([pl.col(c).null_count().alias(c) for c in shared]).row(0)
        p_null = prev.select([pl.col(c).null_count().alias(c) for c in shared]).row(0)
        for col, nn, pn in zip(shared, n_null, p_null):
            rate, prior = nn / new.height, pn / prev.height
            if rate > null_warn >= prior:
                out.append(
                    _finding("null_rate", "warn",
                             f"{col!r} is {rate:.1%} null, was {prior:.1%}",
                             locator={"column": col}, expected=round(prior, 4),
                             actual=round(rate, 4), metric=round(rate - prior, 4))
                )
        n_uniq = new.select([pl.col(c).n_unique().alias(c) for c in shared]).row(0)
        p_uniq = prev.select([pl.col(c).n_unique().alias(c) for c in shared]).row(0)
        for col, nu, pu in zip(shared, n_uniq, p_uniq):
            if nu <= 1 < pu:
                out.append(
                    _finding("constant_column", "warn",
                             f"{col!r} went constant ({pu} distinct values last release)",
                             locator={"column": col}, expected=pu, actual=nu)
                )

    numeric = [c for c in shared if new.schema[c].is_numeric() and prev.schema[c].is_numeric()]
    if numeric:
        n_mean = new.select([pl.col(c).mean().alias(c) for c in numeric]).row(0)
        p_mean = prev.select([pl.col(c).mean().alias(c) for c in numeric]).row(0)
        for col, nm, pm in zip(numeric, n_mean, p_mean):
            if nm is None or pm is None:
                continue
            denom = max(abs(pm), 1e-9)
            shift = abs(nm - pm) / denom
            if shift > shift_warn:
                out.append(
                    _finding("rate_anomaly", "warn",
                             f"mean of {col!r} moved {shift:.1%} ({pm:.4g} -> {nm:.4g})",
                             locator={"column": col}, expected=pm, actual=nm,
                             metric=round(shift, 4))
                )
    return out


def season_summary(
    qa: pl.DataFrame,
    season: int,
    *,
    processing_version: str,
    drift: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Games, error-free share, per-rule counts and the drift findings."""
    games = qa.height
    error_free = int(qa.get_column("ok").sum()) if games else 0
    share = error_free / games if games else None
    counts: dict[str, int] = {}
    for col in ("failed_rule_ids", "warned_rule_ids"):
        for cell in qa.get_column(col).drop_nulls().to_list() if games else []:
            for rule in filter(None, cell.split(",")):
                counts[rule] = counts.get(rule, 0) + 1
    error_share = 1 - share if share is not None else None
    return {
        "league": LEAGUE,
        "source": SOURCE,
        "season": int(season),
        "processing_version": processing_version,
        "built_at": _utc_now(),
        "games": games,
        "games_error_free": error_free,
        "error_free_share": round(share, 4) if share is not None else None,
        "error_share": round(error_share, 4) if error_share is not None else None,
        "max_error_share": MAX_ERROR_SHARE,
        "threshold_exceeded": (
            None if error_share is None else error_share > MAX_ERROR_SHARE
        ),
        "blocking": BLOCKING,
        "counts_by_rule": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "drift": drift or [],
    }


#: Where a published season asset lives (the drift gate's reference).
RELEASE_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "{tag}/{stem}_{season}.parquet"
)


def published_url(tag: str, stem: str, season: int) -> str:
    return RELEASE_URL.format(tag=tag, stem=stem, season=int(season))


def summary_path(parquet_path: str | Path) -> Path:
    """The ``_summary.json`` sidecar beside a season's QA parquet."""
    p = Path(parquet_path)
    return p.with_name(f"{p.stem}_summary.json")


def published_frame(url: str, columns: list[str] | None = None) -> pl.DataFrame | None:
    """The previously published season asset, or ``None`` when it cannot be read.

    Best-effort by design: the drift gate is report-only, so a release that is
    not there yet (a first publish) or a network hiccup must not fail a build.
    """
    try:
        return pl.read_parquet(url, columns=columns)
    except Exception as exc:  # noqa: BLE001 -- no release yet / offline / transient
        log.info("drift gate: previous release unreadable (%s): %r", url, exc)
        return None


def write_summary(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def log_summary(summary: dict[str, Any]) -> None:
    """The one line a build leaves behind (and the drift count)."""
    share = summary["error_free_share"]
    log.info(
        "qa %s %s: %d games, %d error-free (%s), threshold %.2f%s, blocking=%s, %d drift finding(s)",
        summary["league"],
        summary["season"],
        summary["games"],
        summary["games_error_free"],
        "n/a" if share is None else f"{share:.1%}",
        summary["max_error_share"],
        " EXCEEDED" if summary["threshold_exceeded"] else "",
        summary["blocking"],
        len(summary["drift"]),
    )
