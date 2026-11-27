# hierarchical_forward

The 2026 forward check of the hierarchical team-strength model against the served pregame formula, fixed by its
pre-registration of 2026-11-27 before any snapshot here was made.

There is one parquet per slot `t`: the regular-season week, or 100 + the postseason week. Each one is written by
`python -m cfb_model_build.cfb_higher_models.forward_snapshot --season 2026 --t <t>` and pushed to `main` before the
slot's first remaining kickoff.
- The push time is the snapshot's publish time.
- A game counts only when its `kickoff` is later than that time.
- A slot that missed its snapshot is not filled in afterwards.

Columns:
- `hier_margin` / `hier_home_win_prob` come from the hierarchical arm, `fit_eb` on every completed game before the
  slot.
- `served_margin` / `served_home_win_prob` come from sportsdataverse-py `cfb_predict_games` on
  `cfb_ratings_weekly` at `through_week = t - 1`.
- `sdv_commit` is the sportsdataverse-py the snapshot ran on.

After the CFP final, `forward_check.py` scores the window against `cfb_schedules`.
