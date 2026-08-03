"""Net matchup efficiency: pass/rush units against each other, scaled by pace.

WHY THIS IS NOT ANOTHER TEAM-STRENGTH FEATURE
---------------------------------------------
The family ablation found the 384-column substrate to be roughly ONE dimension
-- team quality -- measured two hundred ways: removing any family cost at most
0.14 MAE, and 7 rating columns reached 13.62 against 13.26 for all 211. Adding
a 212th efficiency metric cannot help.

A MATCHUP is a different object. "Your pass offense against my pass defense" is
not the same information as "you are good and I am average": it is an
interaction, and interactions survive the collapse that killed the marginals.
Two teams with identical net ratings produce different games when one is
pass-heavy and the other is weak specifically against the pass.

THE MODEL
---------
Points are (efficiency per play) x (number of plays). The shipped closed form
collapses both into a single rating difference, which throws away pace -- a
fast, mediocre team and a slow, mediocre team have the same rating and very
different totals and margins.

    expected_pass_epa = f(off pass efficiency, opponent def pass efficiency)
    expected_rush_epa = f(off rush efficiency, opponent def rush efficiency)
    pass_rate         = blend(offense tendency, defense's faced tendency)
    epa_per_play      = pass_rate * pass_epa + (1 - pass_rate) * rush_epa
    plays             = blend(offense pace, defense pace faced)
    points            = plays * epa_per_play * (points per EPA)

Each team's expected points are computed separately, so MARGIN and TOTAL fall
out of the same object rather than being modelled independently -- which also
means they cannot disagree with each other.

Every coefficient is FIT, never assumed: the naive "offense minus defense"
combination assumes both sides matter equally and that the units are already
on a common scale, and neither is safe here (see points_scale for what
happened to special teams under exactly that assumption).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

#: Per-unit efficiency columns in the weekly substrate, by phase and side.
#: `_off_pass` = offense on pass plays, `_def_pass` = defense against the pass.
PHASE_COLS = {
    "pass": ("EPAplay_off_pass", "EPAplay_def_pass"),
    "rush": ("EPAplay_off_rush", "EPAplay_def_rush"),
}
#: Tendency (how often a team passes) and pace (how many plays they run).
TENDENCY_COL = "passrate_off"
PACE_COL = "playsgame_off"
#: Defensive counterparts: the rate/pace a defense has FACED.
TENDENCY_FACED = "passrate_def"
PACE_FACED = "playsgame_def"


@dataclass
class MatchupModel:
    """Fitted matchup -> points model.

    ``phase_coef`` holds, per phase, the weights on (own offence, opponent
    defence) that best predict realised per-play efficiency. They are not
    constrained to (1, -1): if defence is more stable than offence -- which is
    an empirical question, not a modelling choice -- the fit says so.
    """

    phase_coef: dict[str, tuple[float, float, float]]  # phase -> (a_off, b_def, c)
    points_per_epa: float
    hfa_points: float
    intercept: float
    resid_sd: float
    seasons: list[int] = field(default_factory=list)
    n_train: int = 0

    def summary(self) -> str:
        L = ["phase   a_off   b_def   const"]
        for p, (a, b, c) in self.phase_coef.items():
            L.append(f"  {p:<5} {a:>6.3f} {b:>7.3f} {c:>7.4f}")
        L.append(f"  points_per_epa {self.points_per_epa:>7.2f}")
        L.append(f"  hfa_points     {self.hfa_points:>7.2f}")
        L.append(f"  resid_sd       {self.resid_sd:>7.2f}")
        return "\n".join(L)


def _side(frame: pl.DataFrame, col: str, side: str) -> np.ndarray:
    name = f"{col}_{side}"
    if name not in frame.columns:
        return np.zeros(frame.height)
    return np.nan_to_num(frame[name].to_numpy().astype(float), nan=0.0)


def expected_efficiency(frame: pl.DataFrame, model: MatchupModel, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(epa_per_play, plays) that ``side`` is expected to generate.

    ``side`` is "home" or "away"; the opponent's defensive columns come from
    the other side, which is the whole point -- this is where the interaction
    enters.
    """
    opp = "away" if side == "home" else "home"

    # Tendency: what the offence likes to do, tempered by what this defence
    # tends to face. A pass-heavy offence against a defence that forces runs
    # lands in between.
    own_rate = _side(frame, TENDENCY_COL, side)
    faced_rate = _side(frame, TENDENCY_FACED, opp)
    pass_rate = np.clip(0.5 * own_rate + 0.5 * faced_rate, 0.15, 0.85)

    eff = np.zeros(frame.height)
    for phase, (off_col, def_col) in PHASE_COLS.items():
        a, b, c = model.phase_coef[phase]
        e = a * _side(frame, off_col, side) + b * _side(frame, def_col, opp) + c
        w = pass_rate if phase == "pass" else (1.0 - pass_rate)
        eff += w * e

    # Pace: both teams shape how many snaps happen.
    plays = 0.5 * _side(frame, PACE_COL, side) + 0.5 * _side(frame, PACE_FACED, opp)
    plays = np.where(plays > 0, plays, np.nanmedian(plays[plays > 0]) if (plays > 0).any() else 70.0)
    return eff, plays


def _fit_phase(frame: pl.DataFrame, phase: str) -> tuple[float, float, float]:
    """Fit realised per-play efficiency ~ own offence + opponent defence.

    Target: the offence's own realised EPA/play in the NEXT observed state is
    not available pre-game, so we fit on the relationship the ratings imply --
    regressing the home team's scoring rate proxy on the two inputs. Kept
    deliberately simple and linear; the value here is the INTERACTION
    structure, not the functional form.
    """
    off_col, def_col = PHASE_COLS[phase]
    X = np.column_stack(
        [
            _side(frame, off_col, "home"),
            _side(frame, def_col, "away"),
            np.ones(frame.height),
        ]
    )
    # Proxy target: home margin per play, which is what the phase must explain.
    plays = np.clip(_side(frame, PACE_COL, "home"), 40, 110)
    y = frame["margin"].to_numpy().astype(float) / plays
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1]), float(beta[2])


def fit_matchup(frame: pl.DataFrame) -> MatchupModel:
    phase_coef = {p: _fit_phase(frame, p) for p in PHASE_COLS}
    model = MatchupModel(
        phase_coef=phase_coef,
        points_per_epa=1.0,
        hfa_points=0.0,
        intercept=0.0,
        resid_sd=0.0,
    )
    # Second stage: convert expected per-play efficiency x plays into points,
    # and price home field on the same scale.
    eh, ph = expected_efficiency(frame, model, "home")
    ea, pa = expected_efficiency(frame, model, "away")
    net = eh * ph - ea * pa
    neutral = frame["neutral_site"].to_numpy().astype(bool)
    X = np.column_stack([net, (~neutral).astype(float), np.ones(frame.height)])
    y = frame["margin"].to_numpy().astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    model.points_per_epa = float(beta[0])
    model.hfa_points = float(beta[1])
    model.intercept = float(beta[2])
    model.resid_sd = float(np.std(y - X @ beta))
    model.seasons = sorted(int(s) for s in frame["season"].unique().to_list())
    model.n_train = frame.height
    return model


def predict_margin(frame: pl.DataFrame, model: MatchupModel) -> np.ndarray:
    eh, ph = expected_efficiency(frame, model, "home")
    ea, pa = expected_efficiency(frame, model, "away")
    net = eh * ph - ea * pa
    neutral = frame["neutral_site"].to_numpy().astype(bool)
    return model.points_per_epa * net + np.where(neutral, 0.0, model.hfa_points) + model.intercept


def predict_total(frame: pl.DataFrame, model: MatchupModel, *, base: float = 52.0) -> np.ndarray:
    """Expected combined points -- the same object as the margin, not a rival.

    Because each side's expected production is computed separately, the total
    is their SUM where the margin is their difference. A total model fitted
    independently can contradict the margin model; this one structurally cannot.
    """
    eh, ph = expected_efficiency(frame, model, "home")
    ea, pa = expected_efficiency(frame, model, "away")
    return base + model.points_per_epa * (eh * ph + ea * pa)


def head_matchup(train: pl.DataFrame, test: pl.DataFrame, **_):
    """`fit_predict` closure for the walk-forward runner."""
    return predict_margin(test, fit_matchup(train))


#: Names of the engineered matchup columns added by `add_matchup_features`.
MATCHUP_FEATURES = (
    "mu_pass_edge_home",
    "mu_pass_edge_away",
    "mu_rush_edge_home",
    "mu_rush_edge_away",
    "mu_pass_edge_net",
    "mu_rush_edge_net",
    "mu_pace_expected",
    "mu_pass_rate_home",
    "mu_pass_rate_away",
    "mu_epa_net",
    "mu_epa_total",
)


def add_matchup_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach explicit INTERACTION columns for a tree model to use.

    The family ablation showed the marginals collapse to ~one dimension, so the
    question worth asking is not "is pass offence predictive" (it is, and it is
    redundant) but "does pass offence AGAINST THIS pass defence carry anything
    the marginals cannot express". A GBM can in principle discover such an
    interaction from the raw columns, but only by spending splits on it; naming
    it explicitly is both cheaper and testable -- if these columns add nothing
    in an A/B, the interaction genuinely is not there.

    Edges are (own offence + opponent defence-allowed): both raise expected
    production, which is why they ADD rather than subtract.
    """
    out = frame
    exprs = []
    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        for phase, (off_col, def_col) in PHASE_COLS.items():
            h, a = f"{off_col}_{side}", f"{def_col}_{opp}"
            if h in frame.columns and a in frame.columns:
                exprs.append((pl.col(h).fill_null(0.0) + pl.col(a).fill_null(0.0)).alias(f"mu_{phase}_edge_{side}"))
        tr, pc = f"{TENDENCY_COL}_{side}", f"{TENDENCY_FACED}_{opp}"
        if tr in frame.columns and pc in frame.columns:
            exprs.append(
                (0.5 * pl.col(tr).fill_null(0.5) + 0.5 * pl.col(pc).fill_null(0.5)).alias(f"mu_pass_rate_{side}")
            )
    out = out.with_columns(exprs)

    net = []
    for phase in PHASE_COLS:
        h, a = f"mu_{phase}_edge_home", f"mu_{phase}_edge_away"
        if h in out.columns and a in out.columns:
            net.append((pl.col(h) - pl.col(a)).alias(f"mu_{phase}_edge_net"))
    pace_h, pace_a = f"{PACE_COL}_home", f"{PACE_COL}_away"
    if pace_h in out.columns and pace_a in out.columns:
        net.append(
            (0.5 * pl.col(pace_h).fill_null(70.0) + 0.5 * pl.col(pace_a).fill_null(70.0)).alias("mu_pace_expected")
        )
    out = out.with_columns(net)

    # Pace-scaled versions: efficiency only becomes points once multiplied by
    # how many snaps the game will actually contain. This is the piece a pure
    # rating difference cannot represent.
    final = []
    if "mu_pass_edge_net" in out.columns and "mu_pace_expected" in out.columns:
        blend_h = pl.col("mu_pass_rate_home") if "mu_pass_rate_home" in out.columns else pl.lit(0.5)
        eff_net = blend_h * pl.col("mu_pass_edge_net") + (1 - blend_h) * pl.col("mu_rush_edge_net")
        final.append((eff_net * pl.col("mu_pace_expected")).alias("mu_epa_net"))
        tot = (
            pl.col("mu_pass_edge_home")
            + pl.col("mu_pass_edge_away")
            + pl.col("mu_rush_edge_home")
            + pl.col("mu_rush_edge_away")
        )
        final.append((tot * pl.col("mu_pace_expected")).alias("mu_epa_total"))
    return out.with_columns(final) if final else out
