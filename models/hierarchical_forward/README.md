# hierarchical_forward

The 2026 forward check of the hierarchical team-strength model against the served pregame formula, fixed by its
pre-registration of 2026-11-27 before any snapshot here was made.

There is one parquet per slot `t`: the regular-season week, or 100 + the postseason week. It sits beside
`<stem>_corpus.parquet`, the completed games before the slot that it was fit on. Each is written by
`python -m cfb_model_build.cfb_higher_models.forward_snapshot --season 2026 --t <t> [--part k]` and pushed to `main`
before the slot's first remaining kickoff.
- **Parts:** CFP rounds share slot 101 with the bowls, so a matchup decided later gets a later `--part` of the same
  slot. A game with an undetermined team is skipped and listed.
- **Publish time:** GitHub's push time for the commit that added the file. A game counts only when its `kickoff` is
  later.
- **No refills:** a snapshot is never rewritten, and a slot that missed its snapshot is not filled in afterwards.

Columns:
- `hier_margin` / `hier_home_win_prob` come from the hierarchical arm, `fit_eb` on every completed game before the
  slot.
- `served_margin` / `served_home_win_prob` come from sportsdataverse-py `cfb_predict_games` on
  `cfb_ratings_weekly` at `through_week = t - 1`.
- `sdv_commit` is the sportsdataverse-py the snapshot ran on.

After the CFP final, `forward_check.py` scores the window against `cfb_schedules`.
