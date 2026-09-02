# Scored model-PBP (`espn_cfb_model_pbp`)

*Hand-authored (not produced by `cfb_model_reports`; kept current manually).*

## Overview

`cfb_model_pbp_full.parquet` is the integration surface: the full 2004-present
play-by-play scored with the shipped model suite — EP + class probabilities,
WP (spread + naive), CPOE inputs, and the decision surfaces — one row per
play. Downstream consumers (ratings, higher models, external users) read this
instead of re-scoring 2.2M plays themselves.

## Construction

`python -m cfb_model_build.cfb_model_pbp` (stage 10) rebuilds the frame from
the `cfbfastR-cfb-raw` finals and applies the bundled models; the weekly
in-season cron republishes it so the scored frame always tracks both the
current data AND the current promoted models. The CP model used for CPOE is
promoted inside the CPOE stage (see [cpoe.md](cpoe.md)).

**Athlete columns (additive, 2026-09-01).** Beside `passer_player_name` the frame now
carries `passer_player_id`, `rusher_player_name`, `rusher_player_id`,
`receiver_player_name`, `receiver_player_id` — the ESPN athlete ids sdv-py's
participants module emits and every final.json already carried (the builder used to
drop them). Ids are `Int64`, null where ESPN tagged no participant. Measured in `pbp_full` 2025: a
passer id on 42.2% of all plays vs a passer name on 43.1% (id/name 0.979); rusher 43.4% vs
44.3% (0.980); receiver 37.6% vs 38.7% (0.972) — the 2–3% residue is regex-fallback names
that carry no ESPN id. Before 2005 ESPN ships no passer ids (2004: 0.0% vs names 37.0%).
The build refuses (`check_athlete_ids`) when the newest season ≥ 2005 falls below 0.9 on
any role, so a dropped key can never publish an all-null id column. They key headshots (`https://a.espncdn.com/i/headshots/college-football/players/full/{id}.png`)
and stop same-name passers from merging in leader tables.

## Evaluation

The frame inherits its models' LOSO evaluations (see the per-model reports:
[ep.md](ep.md), [wp_spread.md](wp_spread.md), [cpoe.md](cpoe.md), …); its own
correctness check is the parity/identity suite in the pbp build (EPA sums,
half-boundary resets) plus the R↔Python parity gates in this repo.

## Reproducibility

`scripts/cfb_models.sh 10`; published to `espn_cfb_model_pbp` by stage 60.

## Avenues for improvement & open issues

- **Column-stability contract** — downstream consumers would benefit from a
  schema-versioned guarantee on the scored frame (the loader-schema gate
  covers the release; an in-repo schema doc would close the loop).
- **Known issue:** the frame is only as current as the last cron run against
  the last promoted models; a model promotion without a republish leaves a
  window where the frame carries the previous suite's scores.
