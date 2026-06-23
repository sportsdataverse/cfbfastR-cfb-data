# RB Evaluation (xREPA)

## Overview

The RB-evaluation (xREPA) model maps a running back's per-play efficiency to an **expected** rushing EPA, isolating the back's contribution from team/context. It is an **analytic artifact** used for player evaluation and is **not bundled into sdv-py** (it ships only here, with the model program).


## Model features

**2 features**; one row per qualifying rusher-season. The target is the rusher's `unadjusted_epa`.

| Feature | Type | What it encodes |
|---|---|---|
| `epa_per_play` | smooth `s(0)` | The rusher's mean EPA per rush — the primary efficiency signal. |
| `success` | smooth `s(1)` | The rusher's success rate (share of positive-EPA rushes) — the consistency signal. |



## Recipe & lineage

A `pygam` **LinearGAM(s(0) + s(1))** — two smooth splines — mapping `epa_per_play` and `success` (rate) to `unadjusted_epa`. Fit on **897 rushers across 2015-2025**. The smooth, monotone-ish response surface is deliberately simple and interpretable rather than a high-variance tree ensemble.


## The model

**Algorithm.** A `pygam` **`LinearGAM(s(0) + s(1))`** — two penalized smoothing splines — deliberately simple and interpretable rather than a high-variance tree ensemble. Fit on **897 rushers across 2015-2025**.

**Evaluation.** Leave-one-season-out: bin the predicted EPA, compare to mean realized unadjusted EPA per bin, and report a weighted R² and weighted calibration error (the `rb_eval.validate` recipe, a port of the R `show_calibration_chart`). Metrics + the calibration figure are emitted only when `xrepa_loso.parquet` is present (regenerate with `python -m rb_eval train`).


## Metrics

_No metrics available._


## Calibration Results


## Discussion

Validation is leave-one-season-out: bin the predicted EPA, compare to mean realized unadjusted EPA per bin, and report a weighted R² and weighted calibration error (the `rb_eval.validate` recipe, a port of the R `show_calibration_chart`). The calibration figure plots binned predicted EPA/play against realized EPA. Metrics are emitted only when the `xrepa_loso.parquet` is present (regenerate with `python -m rb_eval train`).


## Feature importance

Both splines contribute monotone-increasing surfaces: higher `epa_per_play` and higher `success` both raise expected EPA, with `epa_per_play` carrying the steeper response. There is no tree-gain importance for a GAM; the spline partial-dependence shapes are the interpretable analogue.


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
