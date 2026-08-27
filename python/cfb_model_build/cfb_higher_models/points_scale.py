"""Put every rating on a common POINTS-OF-MARGIN scale.

WHY
---
The shipped ratings are in mixed units that look commensurable and are not.
Measured on 2021-2024 (3,056 games):

    unit             sd     pts per unit   pts of spread
    adj_off_epa   0.1331          71.06           11.57
    adj_def_epa   0.1163         -73.97           10.50
    adj_st_epa    0.5223           3.39            2.48

Special teams' NUMBERS are ~4x larger than offense's while its predictive
WORTH is ~4.7x smaller -- an ~18x mismatch between apparent and real
importance. The cause is structural: ``adj_st_epa`` is the SUM of three
per-unit centered mean EPA/play (field goal, punt, kick return). Summing three
per-play rates triples the scale versus a single per-play rate, and nothing
weights it by play share -- special teams is ~5-8% of snaps. The result is
labelled "true EPA units", which is exactly what makes it look comparable to
offensive EPA/play.

This module stops arguing about units. Each component is regressed on actual
scoring margin, so every rating is expressed in POINTS -- the only scale on
which "how much does this matter" is a well-posed question. A component that
predicts nothing gets a small coefficient automatically; nobody has to
remember to down-weight it.

Coefficients are fit WALK-FORWARD (train on prior seasons, apply to the test
season) so the points scale itself never sees the games it is scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

#: Rating components to price, in the order they enter the design matrix.
#: Each is a home-minus-away difference at prediction time.
#: NOTE the `rt_` prefix on special teams. The summaries substrate has NO
#: special-teams columns at all (it is FBS-vs-FBS pass/rush only), so an
#: unprefixed "adj_st_epa" silently matches nothing and the component vanishes
#: from the fit -- which is exactly what happened on the first run: the worth
#: table came back with two rows and no complaint. Special teams lives only in
#: the cfb_ratings block.
COMPONENTS = ("adj_off_epa", "adj_def_epa", "rt_adj_st_epa")

#: Defensive EPA is "allowed", so a HIGHER value is WORSE. Priced with its own
#: sign rather than flipped by hand -- if the sign convention of a published
#: column ever changes, the fit absorbs it instead of silently inverting the
#: model.
_SIGN_FREE = True


@dataclass
class PointsModel:
    """Component coefficients in points of margin, plus home-field advantage."""

    coef: dict[str, float]
    hfa_points: float
    intercept: float
    resid_sd: float
    n_train: int
    seasons: list[int] = field(default_factory=list)

    def contributions(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Per-component points contribution for each game -- the readable form.

        This is what a preview surface should show: "offense +6.1, defense
        -2.3, special teams +0.4, home +2.8" summing to the predicted margin.
        In raw EPA those numbers are incomparable; in points they are additive
        and honest about relative size.
        """
        exprs = []
        for c, b in self.coef.items():
            h, a = f"{c}_home", f"{c}_away"
            if h in frame.columns and a in frame.columns:
                exprs.append(((pl.col(h) - pl.col(a)) * b).alias(f"pts_{c}"))
        hfa = pl.when(pl.col("neutral_site").cast(pl.Boolean)).then(0.0).otherwise(self.hfa_points).alias("pts_hfa")
        out = frame.with_columns(*exprs, hfa)
        pcols = [e.meta.output_name() for e in exprs] + ["pts_hfa"]
        return out.with_columns((pl.sum_horizontal(pcols) + self.intercept).alias("pred_margin"))

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.contributions(frame)["pred_margin"].to_numpy()

    def summary(self) -> str:
        L = ["component          pts/unit"]
        for c, b in self.coef.items():
            L.append(f"  {c:<18} {b:>8.2f}")
        L.append(f"  {'home field':<18} {self.hfa_points:>8.2f}  (points)")
        L.append(f"  residual sd        {self.resid_sd:>8.2f}")
        return "\n".join(L)


def _design(frame: pl.DataFrame, components=COMPONENTS) -> tuple[np.ndarray, list[str]]:
    cols, used = [], []
    missing = [c for c in components if f"{c}_home" not in frame.columns]
    if missing:
        # Silently dropping a component produces a fit that looks fine and
        # prices the wrong model -- the first run lost special teams this way
        # and reported a clean two-row table.
        raise ValueError(
            f"points_scale: components absent from the frame: {missing}. "
            f"Available *_home rating columns: "
            f"{[c for c in frame.columns if c.endswith('_home') and 'epa' in c][:8]}"
        )
    for c in components:
        h, a = f"{c}_home", f"{c}_away"
        if h in frame.columns and a in frame.columns:
            d = frame[h].to_numpy() - frame[a].to_numpy()
            cols.append(np.nan_to_num(d, nan=0.0))
            used.append(c)
    # home-field indicator: 1 at a home site, 0 at a neutral one
    neutral = frame["neutral_site"].to_numpy().astype(bool)
    cols.append((~neutral).astype(float))
    cols.append(np.ones(frame.height))
    return np.column_stack(cols), used


def fit_points_model(frame: pl.DataFrame, components=COMPONENTS) -> PointsModel:
    """Least-squares price of each rating component, in points of margin."""
    X, used = _design(frame, components)
    y = frame["margin"].to_numpy().astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return PointsModel(
        coef={c: float(b) for c, b in zip(used, beta[: len(used)])},
        hfa_points=float(beta[len(used)]),
        intercept=float(beta[len(used) + 1]),
        resid_sd=float(np.std(resid)),
        n_train=int(frame.height),
        seasons=sorted(int(s) for s in frame["season"].unique().to_list()),
    )


def component_worth(frame: pl.DataFrame, components=COMPONENTS) -> pl.DataFrame:
    """How much each component is WORTH vs how big its numbers LOOK.

    ``pts_of_spread`` = |coef| * sd(component difference): the points of margin
    a one-standard-deviation edge in that component buys. That is the honest
    relative weight. ``sd`` is merely how large the raw numbers are, and the
    ratio between the two columns is the unit-scale distortion.
    """
    m = fit_points_model(frame, components)
    rows = []
    for c, b in m.coef.items():
        d = frame[f"{c}_home"].to_numpy() - frame[f"{c}_away"].to_numpy()
        d = np.nan_to_num(d, nan=0.0)
        y = frame["margin"].to_numpy().astype(float)
        rows.append(
            {
                "component": c,
                "pts_per_unit": b,
                "sd_diff": float(d.std()),
                "pts_of_spread": float(abs(b) * d.std()),
                "corr_with_margin": float(np.corrcoef(d, y)[0, 1]) if d.std() else 0.0,
            }
        )
    out = pl.DataFrame(rows).sort("pts_of_spread", descending=True)
    total = out["pts_of_spread"].sum()
    return out.with_columns((pl.col("pts_of_spread") / total).alias("share"))
