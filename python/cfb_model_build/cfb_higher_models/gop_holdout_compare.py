"""Near-holdout 2024-25: GOP sdv.ts constants today (48.459 / 2.711 / 17.1697) vs the refit (gop_net_adj_epa),
both flat on the republished net_adj_epa. Paired 2,000-game bootstrap, seed 0.

Run from python/ with CFB_HM_CACHE pointing at the fit's (fresh) cache:
  python -m cfb_model_build.cfb_higher_models.gop_holdout_compare ../models/pregame_fit.json
"""

import json
import sys
from math import erf, sqrt

import numpy as np
import polars as pl

from cfb_model_build.cfb_higher_models.data import build_game_frame

fit = json.load(open(sys.argv[1]))["gop_net_adj_epa"]
seasons = list(range(2014, 2026))
ho = build_game_frame(seasons).filter(pl.col("season").is_in([2024, 2025]))
ho = ho.drop_nulls(["net_adj_epa_home", "net_adj_epa_away"])
diff = (ho["net_adj_epa_home"] - ho["net_adj_epa_away"]).to_numpy()
neutral = ho["neutral_site"].to_numpy().astype(bool)
y = ho["margin"].to_numpy()
won = ho["home_won"].to_numpy().astype(float)
phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))


def score(scale, hfa, sd):
    m = scale * diff + np.where(neutral, 0.0, hfa)
    p = phi(m / sd)
    slope = np.polyfit(m, y, 1)[0]
    return m, np.abs(m - y), (p - won) ** 2, slope


rng = np.random.default_rng(0)
idx = rng.integers(0, len(y), size=(2000, len(y)))
old = score(48.4590, 2.7110, 17.1697)
new = score(fit["net_points_scale"], fit["hfa_points"], fit["margin_sd"])
print(f"n={len(y)}")
for lab, (m, ae, br, sl) in (("sdv.ts today", old), ("refit", new)):
    print(
        f"{lab:13s} MAE {ae.mean():.4f}  Brier {br.mean():.4f}  calibration slope {sl:.3f}"
    )
for lab, a, b in (("dMAE", new[1], old[1]), ("dBrier", new[2], old[2])):
    d = a - b
    bs = d[idx].mean(axis=1)
    print(
        f"{lab} (refit - today) {d.mean():+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]"
    )
