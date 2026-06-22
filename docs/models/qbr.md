# QBR

## Overview

The QBR model reconstructs an ESPN-Total-QBR-style 0-100 quarterback rating from EPA components, so a QBR can be produced for any game in the corpus without an ESPN QBR feed. It is a per-(quarterback, game) regression onto ESPN's published raw QBR.


## Recipe & lineage

A 6-feature XGBoost regression, **45 trees**, full-history retrain. Features are the per-game weighted-mean EPA components that drive QBR: `qbr_epa`, `sack_epa`, `pass_epa`, `rush_epa`, `pen_epa`, plus the posteam `spread`. The EPA components come from the same EP model documented above, so QBR sits one layer above EP/EPA.


## Metrics

| metric | value |
|---|---|
| `n` | 22833 |
| `rmse` | 17.8726 |
| `mae` | 13.9038 |
| `r2` | 0.5853 |
| `corr` | 0.7651 |


## Figures

![](figures/qbr_calibration.png)


## Discussion

LOSO pooled over 22,833 quarterback-games (2005-2025): RMSE 17.87, MAE 13.90, R² 0.585, correlation 0.765. It **decisively beats the legacy 2020 model**: on the 2021-25 holdout, RMSE 16.1 vs 23.2 and R² 0.66 vs 0.29. The earlier (pre-2014) seasons carry most of the residual error — fold RMSEs run ~19-24 before 2014 and ~16 after — consistent with sparser / noisier early ESPN QBR labels. The scatter figure plots predicted QBR against ESPN raw QBR with the y=x reference.


## Limitations

QBR is a **bounded 0-100** target, so an RMSE of ~18 points is large relative to the scale and the model cannot perfectly reproduce ESPN's proprietary formula (which uses clutch weighting and charting inputs we do not have). The 2004 fold has no joined rows (no ESPN QBR labels), and pre-2014 error is materially higher. Treat the output as a faithful reconstruction of the *EPA-explainable* part of QBR, not a byte-exact ESPN replica.


## Provenance

| metric | value |
|---|---|
| `features` | qbr_epa, sack_epa, pass_epa, rush_epa, pen_epa, spread |
| `hyperparameters` | {} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-17 |
| `xgboost_version` | 3.2.0 |
