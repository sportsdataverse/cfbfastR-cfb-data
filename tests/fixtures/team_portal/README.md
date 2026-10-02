# team_portal fixtures

Real 2024 inputs to `build_team_portal`, read on 2026-10-01 with sportsdataverse-py
at `ec17d62e` (#655). They back `test_built_2024_portal_ranks_colorado_first_among_fbs`
in `tests/test_recruiting.py`.

| File | Rows | Source |
| --- | ---: | --- |
| `moves_2024.parquet` | 6,282 | `cfb_transfer_moves(2024)`, all columns (`season, team_id, player_id, direction, prior_team_id, talent_points`; Utf8 ids) |
| `rosters_2024.parquet` | 27,477 | `load_cfb_rosters(2024)` trimmed to `season, team_id, athlete_id, division` (Int64 ids; 134 FBS + 102 FCS teams) |

Expected from these rows: Colorado (38) leads FBS at 41/111, and Miami (2390) has
15 arrivals, one of them athlete 4688380 from Washington State (265).

Re-create (network):

```python
from sportsdataverse.cfb import cfb_transfer_moves, load_cfb_rosters

cfb_transfer_moves(2024).write_parquet("moves_2024.parquet", compression="zstd")
load_cfb_rosters(2024).select("season", "team_id", "athlete_id", "division").sort(
    "team_id", "athlete_id"
).write_parquet("rosters_2024.parquet", compression="zstd")
```
