"""Home-field advantage: one league number, and a shrunk number per team.

The shipped closed form carries a single ``hfa_epa = 0.01848``, which lands at
~1.65 points once scaled -- against an empirical home edge nearer 3 points. But
a single league constant is also the wrong SHAPE: altitude (Wyoming, Air Force),
travel burden (Hawaii), and genuine crowd effects are real and team-specific.

The obstacle is sample size. A team plays ~6 home games a season, so a raw
per-team home-minus-expected average is mostly noise -- and the noise is
seductive, because it always produces a plausible-looking ranking with the
usual suspects near the top by luck alone.

So per-team HFA is estimated as a SHRUNK deviation from the league mean:

    hfa_team = hfa_league + (n / (n + k)) * (raw_team_deviation)

with ``k`` fit by out-of-sample skill rather than assumed. k is large when
per-team HFA is mostly noise (everything collapses to the league value) and
small when it is real. Letting the data set k is the honest version of the
question "do some teams really have a bigger home edge?".

The residual used is margin MINUS the rating-based expectation, so a team that
merely plays weak opponents at home does not accrue phantom home-field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .points_scale import PointsModel, fit_points_model


@dataclass
class HFAModel:
    league_points: float
    team_points: dict[int, float]
    k: float
    n_teams: int

    def for_team(self, team_id: int) -> float:
        return self.team_points.get(int(team_id), self.league_points)

    def apply(self, frame: pl.DataFrame) -> np.ndarray:
        """Per-game HFA in points; 0 at neutral sites."""
        home = frame["home_id"].to_numpy()
        neutral = frame["neutral_site"].to_numpy().astype(bool)
        vals = np.array([self.for_team(t) for t in home], dtype=float)
        return np.where(neutral, 0.0, vals)


def _residual_home_edge(frame: pl.DataFrame, model: PointsModel) -> pl.DataFrame:
    """Margin minus the rating-based expectation, at home sites only.

    Subtracting the rating expectation is what separates "this team has a home
    edge" from "this team schedules cupcakes at home".
    """
    pred_no_hfa = model.predict(frame) - np.where(frame["neutral_site"].to_numpy().astype(bool), 0.0, model.hfa_points)
    return frame.select("home_id", "margin").with_columns(pl.Series("resid", frame["margin"].to_numpy() - pred_no_hfa))


def fit_hfa(frame: pl.DataFrame, *, k: float = 40.0) -> HFAModel:
    """League HFA plus shrunk per-team deviations, in points."""
    model = fit_points_model(frame)
    res = _residual_home_edge(frame, model)
    league = float(res["resid"].mean())
    agg = res.group_by("home_id").agg(pl.col("resid").mean().alias("raw"), pl.len().alias("n"))
    shrunk = agg.with_columns((league + (pl.col("n") / (pl.col("n") + k)) * (pl.col("raw") - league)).alias("hfa"))
    return HFAModel(
        league_points=league,
        team_points={int(t): float(v) for t, v in zip(shrunk["home_id"], shrunk["hfa"])},
        k=k,
        n_teams=shrunk.height,
    )


def tune_k(frame: pl.DataFrame, ks=(5.0, 10.0, 20.0, 40.0, 80.0, 200.0, 1e6)) -> tuple[float, pl.DataFrame]:
    """Pick k by out-of-sample margin error, walk-forward by season.

    k = 1e6 is the "no per-team HFA at all" control. If it wins, per-team home
    field is noise at this sample size and the league constant is the right
    model -- a result worth having explicitly rather than assuming either way.
    """
    seasons = sorted(frame["season"].unique().to_list())
    rows = []
    for k in ks:
        errs = []
        for i, test in enumerate(seasons):
            if i < 3:
                continue
            tr = frame.filter(pl.col("season").is_in(seasons[:i]))
            te = frame.filter(pl.col("season") == test)
            if not tr.height or not te.height:
                continue
            m = fit_points_model(tr)
            h = fit_hfa(tr, k=k)
            # rating expectation without HFA, plus this model's HFA
            base = m.predict(te) - np.where(te["neutral_site"].to_numpy().astype(bool), 0.0, m.hfa_points)
            pred = base + h.apply(te)
            errs.append(np.abs(pred - te["margin"].to_numpy()))
        if errs:
            e = np.concatenate(errs)
            rows.append({"k": k, "mae": float(e.mean()), "n": int(e.size)})
    tbl = pl.DataFrame(rows).sort("mae")
    return float(tbl["k"][0]), tbl
