"""Is a model difference real, or is it noise?

Every comparison this package has reported so far was a point estimate with no
variance attached: "60 features beat 244" rests on 12.966 vs 13.027, a
difference of 0.061 MAE. "Rest doesn't help" rests on 0.014. "Matchup
interactions add nothing" rests on 0.015. None of those came with an interval,
so none of them was actually a result -- in either direction. A difference that
cannot be distinguished from zero is not evidence of absence any more than it
is evidence of presence.

TWO TESTS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS

1. PAIRED, GAME-LEVEL. Both models predict the same games, so their absolute
   errors are strongly correlated and the paired difference has far smaller
   variance than either MAE alone. This is the sensitive test -- but it treats
   5,089 games as 5,089 independent observations, which they are not: games
   share teams, weeks and seasons, so it is ANTI-CONSERVATIVE and will call
   small differences significant.

2. CLUSTERED BY SEASON. Nine walk-forward folds, one mean difference each,
   paired t-test on those nine numbers. Far less powerful, but it respects the
   dependence structure -- a model that wins because one season suited it will
   fail here. This is the honest test for "would this hold next year".

Report both. When they disagree, the season-level answer is the one to act on,
and the disagreement itself is informative: it means the edge is not stable
across seasons.

A bootstrap CI is included because MAE differences are not normally distributed
(absolute errors are skewed) and the t-test's normality assumption is doing
real work at these effect sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class Comparison:
    """Result of comparing model A against model B on identical games."""

    name_a: str
    name_b: str
    mae_a: float
    mae_b: float
    diff: float  # positive => A is WORSE (higher error) than B
    n: int
    # game-level paired (anti-conservative)
    t_stat: float
    p_game: float
    ci_lo: float
    ci_hi: float
    # season-clustered (honest)
    n_seasons: int
    p_season: float
    seasons_won: int

    def verdict(self, alpha: float = 0.05) -> str:
        """Three-way call, with an explicit UNDERPOWERED state.

        The season-clustered test has 9 folds. That is almost no power: the
        positive control (GBM vs ridge, a substantial +0.242 MAE gap that IS
        game-level significant at p=0.005) still returns p=0.26 season-
        clustered. So a failed season test does NOT mean "no effect" -- it
        usually means the design cannot see one. Calling that "unstable" would
        be a false negative dressed as a finding, which is the same error as
        the false positives this module exists to catch.
        """
        if self.p_season < alpha:
            return "REAL (survives season clustering)"
        if self.p_game < alpha:
            return (
                "GAME-LEVEL ONLY (season test underpowered at n=9 -- "
                "suggestive, not confirmed)"
            )
        return "NOT ESTABLISHED (cannot be distinguished from zero)"

    def __str__(self) -> str:
        better = self.name_b if self.diff > 0 else self.name_a
        return (
            f"{self.name_a} ({self.mae_a:.3f}) vs {self.name_b} ({self.mae_b:.3f})\n"
            f"  diff {self.diff:+.4f} MAE  95% CI [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}]\n"
            f"  p(game-level, n={self.n}) = {self.p_game:.4f}   "
            f"p(season-clustered, n={self.n_seasons}) = {self.p_season:.4f}\n"
            f"  seasons won by better model: {self.seasons_won}/{self.n_seasons}\n"
            f"  --> {self.verdict()}  (better: {better})"
        )


def _bootstrap_ci(
    d: np.ndarray, *, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI on the mean paired difference.

    Absolute errors are right-skewed, so the t-interval's normality assumption
    is load-bearing at these effect sizes. The bootstrap does not need it.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare_oof(
    oof_a: pl.DataFrame, oof_b: pl.DataFrame, *, name_a: str, name_b: str
) -> Comparison:
    """Paired comparison of two out-of-fold prediction frames.

    Both frames must carry game_id, season, margin, pred_margin. They are
    INNER-JOINED on game_id so the comparison is strictly like-for-like -- if
    one model scored fewer games, only the shared ones count.
    """
    from scipy import stats

    a = oof_a.select("game_id", "season", "margin", pl.col("pred_margin").alias("pa"))
    b = oof_b.select("game_id", pl.col("pred_margin").alias("pb"))
    j = a.join(b, on="game_id", how="inner")
    if not j.height:
        raise ValueError("no shared games between the two OOF frames")

    y = j["margin"].to_numpy().astype(float)
    ea = np.abs(j["pa"].to_numpy() - y)
    eb = np.abs(j["pb"].to_numpy() - y)
    d = ea - eb  # positive => A has more error => A worse

    t_stat, p_game = stats.ttest_rel(ea, eb)
    lo, hi = _bootstrap_ci(d)

    # Season-clustered: one number per fold, then a paired test on those.
    per = (
        j.with_columns(pl.Series("d", d))
        .group_by("season")
        .agg(pl.col("d").mean())
        .sort("season")
    )
    ds = per["d"].to_numpy()
    if len(ds) > 1:
        t_s, p_season = stats.ttest_1samp(ds, 0.0)
    else:
        p_season = float("nan")
    better_is_b = float(np.mean(d)) > 0
    seasons_won = int(np.sum(ds > 0)) if better_is_b else int(np.sum(ds < 0))

    return Comparison(
        name_a=name_a,
        name_b=name_b,
        mae_a=float(ea.mean()),
        mae_b=float(eb.mean()),
        diff=float(d.mean()),
        n=j.height,
        t_stat=float(t_stat),
        p_game=float(p_game),
        ci_lo=lo,
        ci_hi=hi,
        n_seasons=len(ds),
        p_season=float(p_season),
        seasons_won=seasons_won,
    )


def mae_standard_error(oof: pl.DataFrame) -> dict[str, float]:
    """How precise is a single model's MAE, unpaired?

    Context for reading any headline number: if the SE is 0.17, a reported
    improvement of 0.06 means nothing on its own. The paired test is what
    rescues comparisons at that scale -- not a larger sample.
    """
    y = oof["margin"].to_numpy().astype(float)
    e = np.abs(oof["pred_margin"].to_numpy() - y)
    n_seasons = oof["season"].n_unique()
    per_season = (
        oof.with_columns(pl.Series("e", e)).group_by("season").agg(pl.col("e").mean())
    )["e"].to_numpy()
    return {
        "mae": float(e.mean()),
        "se_naive": float(e.std(ddof=1) / np.sqrt(len(e))),
        "se_clustered": float(per_season.std(ddof=1) / np.sqrt(n_seasons)),
        "n": int(len(e)),
        "n_seasons": int(n_seasons),
    }
