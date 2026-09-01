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
