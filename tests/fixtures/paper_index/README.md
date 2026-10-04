# paper_index fixtures

Real released rows for `python/cfb_data_build/paper_index.py` (stage 67 and the luck
columns on `team_summaries` / `team_summaries_weekly`), cut on 2026-10-04 from this
repo's committed tree at `origin/main` `531fa7404`. The 2024 pbp file is the one
sdv-py's own Paper Index fixture was cut from (same sha256 as
`sportsdataverse-py/tests/fixtures/paper_index/README.md` records).

| fixture | rows | source | source sha256 |
| --- | --- | --- | --- |
| `play_by_play_slice.parquet` | 771 (533 from 2024, 238 from 2021) | `cfb/pbp/parquet/play_by_play_2024.parquet` and `play_by_play_2021.parquet` (`espn_cfb_pbp`) | 2024 `9b4bfe6a8241cf496f7c43096c305269abc32431fc96f080e4a12302009994e3`, 2021 `2e6583fe991a0691259c35f1828738bb20998fe06ee087d678d02b7dd8b08b91` |
| `cfb_schedules_2024_slice.parquet` | 3 | `cfb/cfb_schedules/parquet/cfb_schedules_2024.parquet` (`cfb_schedules`) | `7feb11b7ac2eccd92afd741a59efeb7f1fd26bb51d15675ab1164cd01985a153` |

## Selection

**pbp**: every play of five games, projected to
`sportsdataverse.paper_index.PBP_COLUMNS` (23 columns), sorted by
`game_id, game_play_number`. Ids are as released: `game_id`, `pos_team_id`,
`homeTeamId`, `awayTeamId` all Int64.

| `game_id` | season | type / week | game | final | plays | why it is here |
| --- | --- | --- | --- | --- | --- | --- |
| 401636915 | 2024 | regular, 10 | Arizona State (9) at Oklahoma State (197) | 42-21 ASU | 164 | one of sdv-py's 12 oracle games; week W - 1 of the snapshot test |
| 401636917 | 2024 | regular, 11 | UCF (2116) at Arizona State (9) | 35-31 ASU | 170 | week W of the snapshot test |
| 401677182 | 2024 | postseason, 1 | Peach Bowl, Texas (251) vs Arizona State (9, listed home) | 39-31 Texas (2OT) | 199 | a bowl: ESPN numbers the postseason from week 1 again |
| 401309543 | 2021 | regular, 1 | Norfolk State (2450) at Toledo (2649) | 49-10 Toledo | 49 | under the snap floor: the feed holds 11 Toledo and 18 Norfolk State scrimmage snaps |
| 401282705 | 2021 | regular, 2 | Toledo (2649) at Notre Dame (87) | 32-29 Notre Dame | 189 | Toledo's next game, scored, so its count is 1 and not 2 |

Arizona State is the three-game team: two wins and a loss, with a regular-season week
pair and a bowl. No 2024 game is under the 20-snap floor (all 945 decided games in the
2024 file are scored), so the floor case comes from 2021, where the committed pbp has
two such games (401309543 and 401282178).

**schedule**: the three 2024 games' rows, columns `game_id`, `season`, `week`,
`season_type`, `season_type_id`, `home_id`, `home_team`, `home_points`, `away_id`,
`away_team`, `away_points`. The snapshot filter reads `season_type_id` and `week`; the
scores are the hand check on the win count.

## Hand numbers

Shares are `sportsdataverse.paper_index.paper_index_games` at sdv-py `21549eec` on the
fixture rows. The week-10 share is also Game on Paper's own module on the same rows:
sdv-py's `tests/fixtures/paper_index/gop_compute_cfb.json` gives Oklahoma State's
`homeShare` as `0.2907412420199904`, so Arizona State's is `0.7092587579800096`.

| team | game | won | `paper_share` |
| --- | --- | --- | --- |
| Arizona State | 401636915 (week 10) | yes | 0.70925875798001 |
| Arizona State | 401636917 (week 11) | yes | 0.12240098401069241 |
| Arizona State | 401677182 (Peach Bowl) | no | 0.11377000178908554 |
| Toledo | 401282705 (at Notre Dame) | no | 0.5748213286851482 |

Arizona State, season: `deserved_wins` = 0.70925875798001 + 0.12240098401069241 +
0.11377000178908554 = 0.945429743779788; wins = 2; `luck_wins` = 2 - 0.945429743779788
= 1.054570256220212; variance = 0.206211 + 0.107419 + 0.100826 = 0.414456, so `luck_z`
= 1.054570 / sqrt(0.414456) = 1.638084.

Snapshots (regular season, `week <= W`, inclusive): `through_week` 1-9 hold none of
Arizona State's fixture games, 10 holds week 10, 11-16 hold weeks 10 and 11. The bowl
is in the season table only.

Toledo 2021: one game counted. The 49-10 win is not scored, so it adds neither a game
nor a win: `paper_index_games_n` = 1, `luck_wins` = 0 - 0.5748213286851482.
