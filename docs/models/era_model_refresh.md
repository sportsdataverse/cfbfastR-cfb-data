# Era Model Refresh — 2026-06

## Overview

In June 2026 the full CFB play-by-play corpus (2004–2025, ~18.6k games) was
**re-reprocessed** with two material changes to the modeling inputs:

1. **Consensus betting odds.** `odds_override` (the EPA/WPA pregame spread/total)
   is now sourced from the [`cfb_line_odds`](#) multi-book **median consensus**
   (2006–2025) instead of ESPN's single pickcenter — keyed by the real ESPN
   `game_id`, validated at `corr = 1.000, MAE = 0.00` against the frame.
2. **Roster-backed player IDs + pre-2014 names.** Play-text player-name
   extraction (sdv 0.0.68) and team-aware `{type}_player_id` attribution
   (sdv 0.0.69) make the pre-2014 era usable for player-level aggregation
   (QBR), which previously had no structured participants.

The era-keeper models were **retrained on this refreshed corpus** and validated
under honest leave-one-season-out (LOSO) CV (baseline features vs `+era0–3`
one-hot rule-era dummies) over all 22 seasons. This report records which models
were promoted into the shipped `cfb/models/` bundle and why.

## Validation (full 22-season LOSO, reprocessed corpus)

| Model | Primary metric | baseline → era | era wins? | Decision |
|---|---|---|---|---|
| **qbr** | RMSE | 17.604 → **17.294** (−0.31; r² 0.598 → 0.612) | ✅ yes | **Promoted** (`qbr_era`) |
| **fg** | logloss | 0.5265 → **0.5247** (−0.0018; cal 0.0085 → 0.008) | ✅ yes | **Promoted** (`fg_era`) |
| **wp_spread** | logloss | 0.3486 → 0.3485 (−0.0001) | ✅ marginal | **Promoted** (`wp_spread_backfilled`, non-era) |
| **fourth_down** | 1st-down cal-MAE | 0.00272 → 0.0029 (+0.00018) | ❌ no | **Not promoted** (kept shipped `fd_model`) |

## Key finding — consensus odds reshaped era candidacy

The `era0–3` one-hot dummies were originally a proxy for era-varying signal the
**noisier single-book ESPN odds** could not capture. The multi-book consensus
now captures that signal **directly** in the spread features, so for the
**spread-dependent** models the baseline improved enough that era became
redundant or harmful:

- **wp_spread** baseline logloss dropped **0.3616 → 0.3486** — the new baseline is
  *better than the old era model* (0.3518). The consensus odds did the work; era
  is now a −0.0001 rounding effect. The promoted `wp_spread_backfilled` keeper is
  the canonical 13-feature recipe retrained on the consensus-backfilled frame
  (no era term needed).
- **fourth_down** 1st-down cal-MAE dropped **0.0035 → 0.00272** — also now better
  than the old era model (0.0027), so `fd_model_era` is *actively worse* than the
  refreshed baseline and was **not** promoted.

For the **non-spread** models (`qbr`, `fg`) the era signal is not in the odds, so
the era dummies still help and those era variants were promoted.

## Promotion + integration

The shipped sdv-py CFB models were **already era-aware** — `qbr_model` and
`fg_model` carry `era0–3` in their feature_names, and `cfb_pbp` /
`model_vars` already compute and feed those one-hots during model application.
The refreshed keepers have **byte-identical feature contracts** to the shipped
models, so promotion was a clean drop-in `.ubj` replacement (no code changes):

| sdv-py slot | replaced with |
|---|---|
| `cfb/models/qbr_model.ubj` | `qbr_era.ubj` |
| `cfb/models/fg_model.ubj` | `fg_era.ubj` |
| `cfb/models/wp_spread.ubj` | `wp_spread_backfilled.ubj` |
| `cfb/models/fd_model.ubj` | *(unchanged)* |

## Notes & limitations

- The per-model report pages (`qbr.md`, `fg.md`, `wp_spread.md`) render the
  **non-era** LOSO calibration (the long-standing reports convention); the
  era-aware verdicts above are the authoritative validation for the shipped
  (era) models.
- `wp_spread`'s near-zero era delta is a *good* result: it reflects the
  consensus-odds upgrade absorbing the era signal, not a loss of model quality.
- Reproduce: `python -m era_experiment --pbp artifacts/pbp_full_spreadfilled.parquet --only fg,qbr,wp_spread,fourth_down --espn-qbr <espn_qbr.parquet>` → `artifacts/era_results_new.json`.
