# Matchup reference tables — import notes

`cfb_matchup_{coordinators,returning_production,roster_talent}.csv` are imported
values (see `cfb_data_build.matchup_reference` for what they are and are not).

## Rows removed at import, with the evidence

The returning-production source carried four superseded rows: a team-season
appeared twice, once under a second spelling, with different values. The
delivered lines settle each one, so the stale row was dropped rather than
resolved by sort order:

| row dropped | kept | evidence |
|---|---|---|
| `2024,Western Kentucky,0.72,0.54` | `0.59,0.64` | both delivered snapshots carry 0.59 / 0.64 for every 2024 Western Kentucky game |
| `2025,Uconn,0.68,0.46` | `2025,Connecticut,0.68,0.42` | the final delivered line (week 17) carries 0.42; an earlier run (week 3) still carried 0.46, so the `Connecticut` row supersedes the `Uconn` one |
| `2026,Uconn,0.61,0.55` | `2026,Connecticut,0.25,0.37` | same pair, same precedence |
| `2026,Umass,0.26,0.33` | `2026,UMass,0.49,0.43` | same precedence: no delivered 2026 line exists, but `UMass` is the source's spelling in every season 2015-2025 and in the delivered 2025 line, and the source's odd-cased duplicate (`Uconn`) is the superseded row wherever a delivered line settles it. `matchup_line` 2026 refused to build on this pair |

`match_team_names` now raises when two spellings of one school carry different
values, so a future import cannot reintroduce this silently.

## `cfb_matchup_coordinators_wikipedia.csv` — provenance

Head coach and offensive / defensive coordinators for every FBS school-season
from 2002, read from each season's English Wikipedia article (the
`Infobox NCAA team season` / `Infobox college sports team season` infobox)
through the MediaWiki API. The school-season universe is CFBD `/coaches`.

Every row records `source_title` and `source_revid`, the exact article
revision it was read from. `python -m cfb_data_build.coordinators_wiki pinned`
re-reads those revisions and reproduces this file byte for byte (verified
2026-09-16), so later edits to the articles cannot change it silently.

Measured against the imported coordinator table on the seasons both cover
(2013-2016): head coach 0.994, offensive coordinators 0.940 and defensive
0.926 any-overlap. Several disagreements were errors in the imported table,
not in Wikipedia: "Rich Skorsky" (Skrosky), "John Regan" (Reagan), "Kirk
Ciarocca" (Ciarrocca), USC 2013 recorded as "Vacant" (Clay Helton), South
Carolina 2013 missing its co-coordinator.

The names are facts; the articles they came from are credited per row above,
under Wikipedia's CC BY-SA licence.

## `cfb_coach_seasons.csv` — 2003 added

2003 was appended with `python -m cfb_data_build.coaches -s 2003 -e 2003`
(CFBD `/coaches`); every existing season is unchanged. Six 2003 rows are not
cleanly attributed because the school changed head coach mid-season (Army,
Duke, Nebraska, UCF).
