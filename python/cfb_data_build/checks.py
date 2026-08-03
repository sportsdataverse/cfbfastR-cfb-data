"""Build-time gates for silent no-ops.

WHY THIS EXISTS
---------------
On 2026-08-01 `cfb_team_summaries_weekly` was published with
``adj_off_epa`` correlating **0.9928** with its own unadjusted ``EPAplay_off``.
The opponent adjustment had done nothing: ``cfb_adjusted_epa``'s default ridge
penalty was still the glmnet-scale 325, which under sklearn's
``alpha = lambda * n`` convention crushes every team effect to ~0.

Nothing failed. The build ran green, the columns were present, the values were
plausible, every downstream consumer got numbers. The dataset shipped and sat
live for two days. It was found only because someone measured the output.

That is the defining failure mode of this codebase -- components reporting
success while doing nothing (the same shape as the ridge no-op itself, a
`fill_null(0.0)` that is a no-op on Boolean columns, and a verify pass that
only checked the rows it had targeted). The durable fix is not "remember to
check the lambda"; it is to assert on the OUTPUT, so a transform that fails to
transform cannot pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Above this, an "adjusted" column is indistinguishable from its raw input.
#: A real CFB opponent adjustment lands near 0.6-0.7 (measured on
#: cfb_ratings_weekly: 0.628); the no-op measured 0.993. 0.95 sits in the
#: empty space between, so it catches the failure without tripping on a season
#: that genuinely had little schedule variance.
NOOP_CORR_THRESHOLD = 0.95

#: (adjusted column, the raw column it is derived from)
ADJUSTMENT_PAIRS = (
    ("adj_off_epa", "EPAplay_off"),
    ("adj_def_epa", "EPAplay_def"),
)


def adjustment_report(df: pl.DataFrame, pairs=ADJUSTMENT_PAIRS) -> dict[str, float]:
    """corr(adjusted, raw) for each pair present. Lower = more adjustment."""
    out: dict[str, float] = {}
    for adj, raw in pairs:
        if adj not in df.columns or raw not in df.columns:
            continue
        d = df.select(adj, raw).drop_nulls()
        if d.height < 50:
            continue
        x, y = d[adj].to_numpy(), d[raw].to_numpy()
        if x.std() == 0 or y.std() == 0:
            out[adj] = float("nan")
            continue
        out[adj] = float(np.corrcoef(x, y)[0, 1])
    return out


def assert_adjustment_is_real(
    df: pl.DataFrame, *, threshold: float = NOOP_CORR_THRESHOLD, label: str = ""
) -> dict[str, float]:
    """Raise if an opponent-adjusted column is ~identical to its raw input.

    Call this on any frame carrying ``adj_*`` columns before writing or
    publishing it.

    Raises:
        ValueError: if any adjusted column correlates above ``threshold`` with
            its raw source, i.e. the adjustment did not happen.
    """
    rep = adjustment_report(df)
    bad = {k: v for k, v in rep.items() if v == v and v > threshold}
    if bad:
        detail = ", ".join(f"{k} vs raw corr={v:.4f}" for k, v in bad.items())
        raise ValueError(
            f"opponent adjustment is a NO-OP{' in ' + label if label else ''}: "
            f"{detail} (threshold {threshold}). The ridge penalty is almost "
            f"certainly on the wrong scale -- sklearn uses alpha = lambda * n, "
            f"so a glmnet-scale lambda (e.g. 325) shrinks every team effect to "
            f"zero. Check sportsdataverse.cfb.cfb_adjusted_epa._RIDGE_LAMBDA "
            f"(expected ~0.035) and that no caller overrides it."
        )
    return rep
