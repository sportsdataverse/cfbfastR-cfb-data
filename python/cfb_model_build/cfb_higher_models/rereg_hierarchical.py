"""Re-registration test of the hierarchical team-strength model against the NEW shipped surface.

Pre-registered on 2026-11-26 in the maintainers' private notes, which was committed before this script was ever run. It scores 2005-2013 (2004 only seeds the
carryover prior). No fit or gate scored those seasons for this question.

    cd python && CFB_HM_CACHE=<fresh dir> python -m cfb_model_build.cfb_higher_models.rereg_hierarchical

Baseline: the served sportsdataverse-py formula (``cfb_game_predict``), i.e. the games-played slope
curve plus ``hfa_points`` off neutral sites, from ``CFB_CONSTANTS["modern"]`` via
``fit_pregame.from_constants(...).predict(use_curve=True)``. For continuity with the original gate,
``backtest.shipped_margin`` is reported too; it does not decide anything. Prints PASS or FAIL per the
spec's rule and exits 0 either way. A FAIL is a result, not an error.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

SEASONS = list(
    range(2004, 2014)
)  # 2004 fit-only (seeds the carryover); 2005-2013 scored
DELTA_MAX = -0.15  # hierarchical MAE - served MAE must be <= this
P_SEASON_MAX = 0.05  # season-clustered paired t-test
SEASONS_WON_MIN = 8  # of 9 scored seasons
CAL_SLOPE = (0.80, 1.10)
MAX_CAL_ERR = 0.10


def main() -> int:
    from sportsdataverse.cfb.cfb_prediction_constants import CFB_CONSTANTS

    from .backtest import shipped_margin
    from .data import build_game_frame, load_games
    from .fit_pregame import from_constants
    from .hierarchical import build_corpus, walk_forward
    from .metrics import margin_metrics, norm_cdf, wp_metrics
    from .significance import compare_oof

    preds, _diags, voids, _ratings = walk_forward(
        build_corpus(load_games(SEASONS)), SEASONS, method="eb"
    )
    hist = build_game_frame(SEASONS)
    joined = hist.filter(pl.col("season") > min(SEASONS)).join(
        preds.select("game_id", "pred_margin"), on="game_id", how="inner"
    )
    served = from_constants(CFB_CONSTANTS["modern"]).predict(joined, use_curve=True)
    flat = shipped_margin(joined)
    oof = joined.select("game_id", "season", "margin", "home_won", "pred_margin")

    y = oof["margin"].to_numpy().astype(float)
    p = oof["pred_margin"].to_numpy().astype(float)
    m = margin_metrics(p, y)
    w = wp_metrics(norm_cdf(p, float(np.std(y - p))), oof["home_won"].to_numpy())

    def vs(base: np.ndarray, name: str):
        return compare_oof(
            oof.select("game_id", "season", "margin").with_columns(
                pl.Series("pred_margin", base)
            ),
            oof,
            name_a=name,
            name_b="hierarchical",
        )

    cmp = vs(served, "served")
    print(
        f"scored games {oof.height} in seasons {sorted(oof['season'].unique().to_list())}; voids {voids.height}"
    )
    print(
        f"hierarchical MAE {m['mae']:.4f}, cal slope {m['calibration_slope']:.3f}, max cal err {w['max_cal_err']:.3f}"
    )
    print(
        f"served MAE {np.mean(np.abs(served - y)):.4f}; flat shipped_margin MAE {np.mean(np.abs(flat - y)):.4f}"
    )
    print("PRIMARY (vs served):", cmp)
    print(
        "SECONDARY (vs backtest.shipped_margin, decides nothing):",
        vs(flat, "shipped_margin"),
    )

    # Comparison.diff = MAE(served) - MAE(hierarchical): positive when hierarchical is better
    delta = -cmp.diff
    # Comparison.seasons_won counts the BETTER model's seasons; take the hierarchical side
    hier_won = cmp.seasons_won if cmp.diff > 0 else cmp.n_seasons - cmp.seasons_won
    checks = {
        f"delta MAE {delta:+.4f} <= {DELTA_MAX}": delta <= DELTA_MAX,
        f"season-clustered p {cmp.p_season:.4g} < {P_SEASON_MAX}": cmp.p_season
        < P_SEASON_MAX,
        f"hierarchical won {hier_won}/{cmp.n_seasons} seasons >= {SEASONS_WON_MIN}": hier_won
        >= SEASONS_WON_MIN,
        f"cal slope {m['calibration_slope']:.3f} in {CAL_SLOPE}": CAL_SLOPE[0]
        <= m["calibration_slope"]
        <= CAL_SLOPE[1],
        f"max cal err {w['max_cal_err']:.3f} <= {MAX_CAL_ERR}": w["max_cal_err"]
        <= MAX_CAL_ERR,
    }
    for k, ok in checks.items():
        print(("  ok   " if ok else "  FAIL ") + k)
    print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
