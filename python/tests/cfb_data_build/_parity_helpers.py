"""Shared frame-parity assertion for the cfb_data_build oracle tests.

Compares a Python-built frame against the R-released oracle parquet:

* column names: exact-and-ordered by default (``match_order=True``); set
  ``match_order=False`` to require the same column *set* (order-agnostic), used
  for the team_summaries multi-join frames whose column order is a join artifact.
* row count: exact.
* values: type-normalized on the *oracle* dtype, null-aware. Numeric/boolean
  cols -> Float64 with 1e-9 tolerance; other cols -> Utf8 exact. Columns listed
  in ``corr_cols`` are instead held to a Pearson-correlation bar
  (``corr_threshold``) -- for model outputs that cannot byte-match (the
  glmnet-vs-sklearn ridge-adjusted EPA columns).

Row alignment: pass ``sort_keys`` (stable id columns) for frames whose row order
is not positional; both frames are sorted by those keys before comparison.
``sort=True`` sorts by the full column tuple (only safe when no ``corr_cols``).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def assert_frame_parity(
    py: pl.DataFrame,
    oracle: pl.DataFrame,
    *,
    name: str,
    sort: bool = False,
    sort_keys: list[str] | None = None,
    match_order: bool = True,
    corr_cols: set[str] | None = None,
    corr_threshold: float = 0.95,
) -> None:
    corr_cols = corr_cols or set()
    if match_order:
        assert py.columns == oracle.columns, (
            f"{name}: column name/order mismatch\n"
            f"  only in py     : {[c for c in py.columns if c not in set(oracle.columns)]}\n"
            f"  only in oracle : {[c for c in oracle.columns if c not in set(py.columns)]}\n"
            f"  first order div: "
            + str(
                [
                    (i, a, b)
                    for i, (a, b) in enumerate(zip(py.columns, oracle.columns))
                    if a != b
                ][:5]
            )
        )
    else:
        assert set(py.columns) == set(oracle.columns), (
            f"{name}: column SET mismatch\n"
            f"  only in py     : {sorted(set(py.columns) - set(oracle.columns))}\n"
            f"  only in oracle : {sorted(set(oracle.columns) - set(py.columns))}"
        )
        py = py.select(oracle.columns)  # align order for positional comparison
    assert py.height == oracle.height, (
        f"{name}: row count {py.height} != {oracle.height}"
    )

    if sort_keys:
        oracle = oracle.sort(sort_keys, nulls_last=True)
        py = py.sort(sort_keys, nulls_last=True)
    elif sort and oracle.columns:
        oracle = oracle.sort(oracle.columns, nulls_last=True)
        py = py.sort(py.columns, nulls_last=True)

    mismatches: list[str] = []
    for col in oracle.columns:
        o = oracle[col]
        p = py[col]
        odt = oracle.schema[col]
        if col in corr_cols:
            ov = o.cast(pl.Float64, strict=False).to_numpy()
            pv = p.cast(pl.Float64, strict=False).to_numpy()
            mask = ~(np.isnan(ov) | np.isnan(pv))
            if mask.sum() < 3:
                continue
            r = float(np.corrcoef(ov[mask], pv[mask])[0, 1])
            if not (r >= corr_threshold):
                mismatches.append(
                    f"  {col}: correlation {r:.4f} < {corr_threshold} ({mask.sum()} pts)"
                )
            continue
        if odt.is_numeric() or odt == pl.Boolean:
            o2 = o.cast(pl.Float64, strict=False)
            p2 = p.cast(pl.Float64, strict=False)
            # R mean(.., na.rm=TRUE) of an all-NA group is NaN; polars gives null.
            # Treat NaN and null as the same "missing" value.
            o_miss = o2.is_null() | o2.is_nan().fill_null(False)
            p_miss = p2.is_null() | p2.is_nan().fill_null(False)
            close = ((o2 - p2).abs() < 1e-9).fill_null(False)
            eq = (o_miss & p_miss) | (close & ~o_miss & ~p_miss)
        else:
            o2 = o.cast(pl.Utf8, strict=False)
            p2 = p.cast(pl.Utf8, strict=False)
            eq = (o2 == p2).fill_null(False) | (o2.is_null() & p2.is_null())
        n_bad = int((~eq).sum())
        if n_bad:
            bad_idx = [i for i, ok in enumerate(eq.to_list()) if not ok][:3]
            samples = [(i, o2[i], p2[i]) for i in bad_idx]
            mismatches.append(
                f"  {col} ({odt}): {n_bad} mismatched; sample (idx, oracle, py)={samples}"
            )
    assert not mismatches, f"{name}: value mismatches:\n" + "\n".join(mismatches)
