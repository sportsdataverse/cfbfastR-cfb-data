# CFB model roadmap — avenues for improvement & open issues

*Hand-authored companion to the generated per-model reports (the
`cfb_model_reports` generator owns those files, so forward-looking notes live
here where regeneration cannot clobber them).*

| model | avenues | open issues |
|---|---|---|
| ep | era x rule interactions (clock rules, OT formats); weather | class-point map is fixed; no 2-pt modeling beyond safety classes |
| wp_spread | line movement; market-implied totals as a feature | `spread_time` sign fix (2026-06) must travel with retrains |
| wp_naive | fallback only — invest in spread variant | large-early-lead miscalibration vs spread variant undocumented as a figure |
| cpoe | receiver/coverage proxies from participants data | CP promotion happens inside the CPOE stage — a partial run can strand the cp model |
| qbr | opponent adjustment; era interactions | qbr_era promotion notes live only in era_model_refresh.md |
| fg | altitude/weather; kicker pooling across seasons | thin pre-2014 attempts data |
| fourth_down | coach-indexed decision layer | 76-class head shares xYAC machinery — explainability needs 3-D pred_contribs handling |
| two_pt | pool with NFL two-point data (transfer) | sparse sample; era coverage starts 2010 |
| xpass | tempo/formation features from participants | era one-hot reference-class convention |
| punt | returner/coverage identity | smoothed tails at rare yardlines |
| rb_eval | GAM -> monotone boosted comparison | xREPA depends on model_pbp vintages |
| pregame_wp | market blend (12.97 vs 12.27 market MAE gap is one dimension of substrate — 60 features beat 244) | CFBD dependency for Five Factors inputs (opt-in secret) |
| model_pbp | schema-version contract for consumers | promotion/republish window (scores lag a promotion until the next cron) |
| cfb_ratings | recruiting prior for early season; intervals | Spearman gates are scale-blind — keep the level-band checks |
| cfb_recruiting_proj | unblock pred_net_epa via this repo's model_pbp; portal-aware retention | retention Spearman 0.229 (weak ordinal, stated) |

Suite-level EDA / SHAP / identified results live in the compiled Quarto
companion [deepdive.md](deepdive.md) (source `deepdive.qmd`, rendered by
`scripts/render_model_docs.sh`).
