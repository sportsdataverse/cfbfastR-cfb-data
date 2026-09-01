# CFB opponent-adjusted team ratings (`cfb_ratings`)

*Hand-authored (not produced by `cfb_model_reports`; kept current manually).*

## Overview

Per-team-season ratings — opponent-adjusted offensive/defensive EPA plus a
net — computed by sdv-py's `cfb_ratings()` over the released `espn_cfb_pbp`
and published per season with an oracle card. `adj_net` is offense minus
defense only; special teams is a separate column (the 2026-08 scale fixes:
netted `adj_*`, true-EPA special teams).

## Methodology

Iterative opponent adjustment over the season-to-date EPA game matrix — a
team's rating is its EPA margin adjusted for the adjusted strength of every
opponent, solved to a fixed point. The team set is every team with
competitive scrimmage plays, FCS included; only ~133 join the FBS-only
published oracles.

## Evaluation (from the published oracle card, measured on released pbp)

| comparison | Spearman |
|---|---|
| `adj_net` vs ESPN FPI | **0.9259** |
| `adj_net` vs SP+ overall | **0.9355** |
| `adj_off_epa` vs SP+ offense | 0.8464 |
| `adj_def_epa` vs SP+ defense | 0.7929 |
| `fei_net` vs FEI | **0.9644** |

Spearman gates are deliberately scale-blind, which is why the level-band
checks from the 2026-08 audit (netted components, ST magnitude) exist
alongside them.

## Reproducibility

`scripts/cfb_models.sh 60` → `cfb_model_publish ratings`
(`cfb_ratings_cron.yml`, in-season). Card sidecar published per run.

## Limitations

Retrodictive season-to-date ratings — no injury/roster awareness; early-season
values lean on thin game matrices. FCS teams are rated but un-oracled.

## Avenues for improvement & open issues

- **Priors and uncertainty** — early-season ratings would benefit from the
  recruiting projection as a prior, and the fixed point yields no intervals.
- **Known issue:** Spearman gates are scale-blind (the 2026-08 audit's
  lesson); the level-band checks must travel with any methodology change.
