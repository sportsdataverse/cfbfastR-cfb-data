# adjustment_ledger fixtures

`sec_2025_plays.parquet`: every FBS pass / rush play with EPA (kneels dropped) of the 67 SEC-vs-SEC games of 2025,
8,854 rows, down-selected to the 14 columns `cfb_adjusted_epa` / `cfb_adjusted_epa_ledger` read. Source: the released
`play_by_play_2025` asset and `load_cfb_schedule([2025])`, read 2026-11-25, through the same FBS-vs-FBS filter and kneel
drop as `prepare_plays_input`. A closed league (16 teams, every opponent inside it), so every team has 8 games and
the leave-one-game-out refits run. Nothing is edited. Used by `tests/cfb_data_build/test_adjustment_ledger.py`.
