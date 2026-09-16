# Matchup reference tables — import notes

`cfb_matchup_{coordinators,returning_production,roster_talent}.csv` are imported
values (see `cfb_data_build.matchup_reference` for what they are and are not).

## Rows removed at import, with the evidence

The returning-production source carried three superseded rows: a team-season
appeared twice, once under a second spelling, with different values. The
delivered lines settle each one, so the stale row was dropped rather than
resolved by sort order:

| row dropped | kept | evidence |
|---|---|---|
| `2024,Western Kentucky,0.72,0.54` | `0.59,0.64` | both delivered snapshots carry 0.59 / 0.64 for every 2024 Western Kentucky game |
| `2025,Uconn,0.68,0.46` | `2025,Connecticut,0.68,0.42` | the final delivered line (week 17) carries 0.42; an earlier run (week 3) still carried 0.46, so the `Connecticut` row supersedes the `Uconn` one |
| `2026,Uconn,0.61,0.55` | `2026,Connecticut,0.25,0.37` | same pair, same precedence |

`match_team_names` now raises when two spellings of one school carry different
values, so a future import cannot reintroduce this silently.
