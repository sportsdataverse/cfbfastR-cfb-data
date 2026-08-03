"""Field position and net punting -- the "hidden yardage" side of the game.

The special-teams composite is a single number that mixes field goals, punting
and returns, and it was found to be ~18x overstated in apparent size relative
to what it predicts (see points_scale). Collapsing three unrelated skills into
one figure also makes the useful part unreachable: punting and starting field
position are persistent team traits, while field-goal outcomes are close to
noise year over year.

So this prices the components separately, in POINTS, using the columns the
weekly substrate already carries:

    start_position_off   average starting field position on offence
    start_position_def   average starting field position the defence hands out
    start_position_margin  the difference -- "hidden yardage"

Field position is exactly the kind of quantity a raw EPA composite obscures: a
team that consistently starts on its own 35 instead of its own 25 has a real,
compounding edge that never shows up as a highlight, and it is separable from
whether the kicker makes 40-yarders.

The point of the module is the ATTRIBUTION, not another rating. Everything is
expressed on the same points scale as offence and defence so the numbers can
be compared without a mental conversion -- which is what went wrong with
special teams in the first place.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Field-position columns present in the weekly summaries substrate.
FP_COLS = ("start_position_off", "start_position_def", "start_position_margin")
#: Punting-adjacent columns. The substrate has no dedicated net-punt metric, so
#: field position handed to the defence is the closest available proxy -- named
#: honestly rather than dressed up as "net punting".
PUNT_PROXY = "start_position_def"


def available(frame: pl.DataFrame) -> list[str]:
    return [c for c in FP_COLS if f"{c}_home" in frame.columns]


def price_components(frame: pl.DataFrame, components: list[str]) -> pl.DataFrame:
    """Points-of-margin price for each component, marginal and joint.

    ``pts_alone`` is the component's price when it is the ONLY regressor;
    ``pts_joint`` is its price alongside the others. A component whose alone
    price is large and joint price is ~0 is a proxy for something already in
    the model -- which is the usual fate of field-position metrics, since good
    teams both start well AND score, and only one of those is causal.
    """
    y = frame["margin"].to_numpy().astype(float)
    diffs = {
        c: np.nan_to_num(
            frame[f"{c}_home"].to_numpy() - frame[f"{c}_away"].to_numpy(), nan=0.0
        )
        for c in components
    }
    rows = []
    X_all = np.column_stack([*diffs.values(), np.ones(len(y))])
    joint, *_ = np.linalg.lstsq(X_all, y, rcond=None)
    for i, (c, d) in enumerate(diffs.items()):
        if d.std() == 0:
            continue
        b_alone = float(np.polyfit(d, y, 1)[0])
        rows.append(
            {
                "component": c,
                "pts_alone": b_alone,
                "pts_joint": float(joint[i]),
                "sd_diff": float(d.std()),
                "spread_alone": float(abs(b_alone) * d.std()),
                "spread_joint": float(abs(joint[i]) * d.std()),
                "corr": float(np.corrcoef(d, y)[0, 1]),
            }
        )
    return pl.DataFrame(rows).sort("spread_joint", descending=True)


def incremental_value(
    frame: pl.DataFrame, base_components: tuple[str, ...], extra: list[str]
) -> dict[str, float]:
    """Does field position add anything ON TOP of the efficiency ratings?

    The honest test. Field position correlates with winning because good teams
    have it, so its standalone number always looks respectable. What matters is
    the residual improvement once offence and defence are already in the model.
    """
    y = frame["margin"].to_numpy().astype(float)

    def _fit(cols: list[str]) -> float:
        mats = [
            np.nan_to_num(
                frame[f"{c}_home"].to_numpy() - frame[f"{c}_away"].to_numpy(), nan=0.0
            )
            for c in cols
            if f"{c}_home" in frame.columns
        ]
        X = np.column_stack([*mats, np.ones(len(y))]) if mats else np.ones((len(y), 1))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(np.mean(np.abs(y - X @ beta)))

    mae_base = _fit(list(base_components))
    mae_with = _fit(list(base_components) + extra)
    return {
        "mae_base": mae_base,
        "mae_with": mae_with,
        "improvement": mae_base - mae_with,
    }
