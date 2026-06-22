# RB Evaluation (xREPA)

## Overview

The RB-evaluation (xREPA) model maps a running back's per-play efficiency to an **expected** rushing EPA, isolating the back's contribution from team/context. It is an **analytic artifact** used for player evaluation and is **not bundled into sdv-py** (it ships only here, with the model program).


## Recipe & lineage

A `pygam` **LinearGAM(s(0) + s(1))** — two smooth splines — mapping `epa_per_play` and `success` (rate) to `unadjusted_epa`. Fit on **897 rushers across 2015-2025**. The smooth, monotone-ish response surface is deliberately simple and interpretable rather than a high-variance tree ensemble.


## Metrics

_No metrics available._


## Discussion

Validation is leave-one-season-out: bin the predicted EPA, compare to mean realized unadjusted EPA per bin, and report a weighted R² and weighted calibration error (the `rb_eval.validate` recipe, a port of the R `show_calibration_chart`). The calibration figure plots binned predicted EPA/play against realized EPA. Metrics are emitted only when the `xrepa_loso.parquet` is present (regenerate with `python -m rb_eval train`).


## Limitations

With only 897 rushers the sample is small, so season-to-season bins can be thin at the EPA extremes. The two-feature GAM intentionally ignores offensive-line, scheme, and opponent adjustments — it estimates the *unadjusted* expected EPA, leaving those adjustments to downstream layers. Because it is an analytic artifact (pickled GAM, not an XGBoost booster), it is excluded from the sdv-py model bundle.


## Provenance

| metric | value |
|---|---|
| `features` | epa_per_play, success |
| `hyperparameters` | {} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-17 |
| `xgboost_version` | n/a |


## Notes

- rb_eval LOSO metrics require xrepa_loso.parquet (run `python -m rb_eval train`).
