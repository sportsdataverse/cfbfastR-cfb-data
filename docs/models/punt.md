# Punt Outcome Distribution

*Hand-authored (not produced by `cfb_model_reports`; kept current manually).*

## Overview

The punt model is deliberately **not a booster**: it is an empirical
distribution of punt outcomes (`punt_distribution.parquet`) keyed by yardline,
consumed by the fourth-down decision surface to price the "punt" branch
against go/FG. A distribution is the right object here — the decision layer
needs the full spread of net outcomes (touchbacks, returns, shanks), not a
point estimate.

## Data & construction

Built by `python -m cfb_model_build.model_training.punt_distribution` over the
full 2004-present finals corpus: for each punting yardline, the observed
distribution of opponent start position, with the sparse tails pooled so a
rarely-punted-from yardline borrows its neighbors' mass.

## Evaluation

As an empirical distribution it is validated by construction sampling — the
fourth-down report's decision calibration (see
[fourth_down.md](fourth_down.md)) is the end-to-end check, since punt EV
errors would surface there as go/punt boundary miscalibration.

## Reproducibility

Stage 30 (`scripts/cfb_models.sh 30`, subcommand family in
`cfb_model_build.model_training`); artifact ships in the sdv-py bundle.
Registry row: `models/REGISTRY.md`.

## Avenues for improvement & open issues

- **Returner/coverage identity** — the distribution conditions on yardline
  only; team-level punt/coverage quality would sharpen 4th-down pricing.
- **Known issue:** rarely-punted-from yardlines borrow neighboring mass —
  fine for EV, but tail probabilities there are smoothed, not observed.
