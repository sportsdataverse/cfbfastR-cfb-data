# CFB recruiting projection (`cfb_recruiting_proj`)

*Hand-authored (not produced by `cfb_model_reports`; kept current manually).*

## Overview

A strictly **as-of** preseason projection: team-season outcomes projected from
roster construction before the season starts. Projecting season S trains only
on seasons < S — no leakage by construction.

## Features (from the published card)

`talent_composite`, `blue_chip_ratio`, `off_returning`, `def_returning`,
`prior_wins` — recruiting talent plus roster continuity plus a prior-results
anchor, fit with ridge regression.

## Evaluation (T2.2 oracle gates, FBS 2018-2023 backtest window)

| gate | value |
|---|---|
| talent Spearman | **0.896** |
| retention Spearman | 0.229 |
| wins MAE floor | **2.35** |

`pred_net_epa` is null **by design** — the adjusted-EPA target's hosted pbp
source 404s; that is a documented data block, not a faked column. Early
seasons (2016-2017) train on fewer than the validated six prior seasons and
sit outside the gated window.

## Reproducibility

`cfb_model_publish recruiting` (`cfb_recruiting_proj_cron.yml`); inputs are
live at build time (247 RDB recruit feed + returning-production).

## Limitations

Retention is a weak ordinal signal (0.229) — stated, not hidden; the model is
a preseason prior, not an in-season predictor, and coaching changes /
transfer-portal shocks after the build date are invisible to it.

## Avenues for improvement & open issues

- **Unblock `pred_net_epa`** — the adjusted-EPA target's hosted source 404s
  (documented block); re-pointing it at this repo's own model_pbp would
  restore the richest target.
- **Known issue:** retention Spearman 0.229 is a weak ordinal signal, stated
  as such; transfer-portal-era roster churn is the suspected cause and a
  portal-aware returning-production feature the obvious fix.
