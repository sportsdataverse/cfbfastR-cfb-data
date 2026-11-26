# hierarchical_gate fixture

`frame_2024_sample.parquet`: 400 real 2024 games (`sample(n=400, seed=0)`) from
`cfb_higher_models.data.build_game_frame(range(2014, 2026))`. It was built on 2026-11-26 from the assets
republished on sportsdataverse-py `bcce2e681e` (cfb-data `da4c5ae93`).

Only the columns the gate's baseline needs are kept: game_id, season, week, margin, home_won, neutral_site,
rt_adj_net_home/away and valid_games_home/away.

It is used by `tests/test_hierarchical.py` to pin that the gate reads the served formula, not the flat one.
