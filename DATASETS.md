# cfbfastR-cfb-data — Dataset Schemas (expected return per compiled-season dataset)

Expected `col_name | col_type | col_description` for each per-game-compiled **season** dataset produced by `cfbfastR-cfb-data` (e.g. the 2026 season). The repo **reshapes** the enriched per-game `final` JSON from `cfbfastR-cfb-raw` — it does not re-enrich — so these schemas are season-invariant; the tables below were generated from a real completed game (ESPN game `401628455`, 2024) and adversarially verified against the live JSON. `col_type` is the R type after `data.table::rbindlist(fill = TRUE)` (character / integer / double / logical / list).

> Datasets are published to `sportsdataverse-data` (and committed in-repo) under the `espn_cfb_*` release tags shown below. `play_by_play` uses `espn_cfb_pbp`. (`cfbfastR::load_cfb_pbp()` currently reads the legacy `cfbfastR_cfb_pbp` tag — see the Plan 2 cutover note.) `officials` and `propbets` are intentionally not produced (unavailable for CFB, §12.8); `power_index`/`linescores` are recent-seasons-only.

## Summary

| dataset | grain | n_cols | release tag |
| --- | --- | --- | --- |
| [play_by_play](#play-by-play) | one row per play | 380 | `espn_cfb_pbp` |
| [team_box](#team-box) | one row per team (2 per game) | 21 | `espn_cfb_team_box` |
| [player_box](#player-box) | one row per player per stat category | 56 | `espn_cfb_player_box` |
| [adv_team](#adv-team) | one row per team | 77 | `espn_cfb_adv_team` |
| [adv_passing](#adv-passing) | one row per passer | 23 | `espn_cfb_adv_passing` |
| [adv_rushing](#adv-rushing) | one row per rusher | 15 | `espn_cfb_adv_rushing` |
| [adv_receiving](#adv-receiving) | one row per receiver | 15 | `espn_cfb_adv_receiving` |
| [adv_defensive](#adv-defensive) | one row per defensive team unit | 21 | `espn_cfb_adv_defensive` |
| [adv_turnover](#adv-turnover) | one row per team | 14 | `espn_cfb_adv_turnover` |
| [adv_drives](#adv-drives) | one row per team | 10 | `espn_cfb_adv_drives` |
| [adv_situational](#adv-situational) | one row per team | 73 | `espn_cfb_adv_situational` |
| [play_participants](#play-participants) | one row per (play, athlete, role) | 56 | `espn_cfb_play_participants` |
| [drives](#drives) | one row per drive | 15 | `espn_cfb_drives` |
| [game_rosters](#game-rosters) | one row per rostered athlete per game | 70 | `espn_cfb_game_rosters` |
| [rosters](#rosters) | one row per rostered athlete per season (deduped) | ~63 | `espn_cfb_rosters` |
| [betting](#betting) | one row per game | 9 | `espn_cfb_betting` |
| [schedules](#schedules) | one row per game | 34 | `espn_cfb_schedules` |
| [linescores](#linescores) | one row per (team, period) | 5 | `espn_cfb_linescores` |
| [power_index](#power-index) | one row per team (or per game) | 22 | `espn_cfb_power_index` |
| [injuries](#injuries) | one row per injury entry | 12 | `espn_cfb_injuries` |

---

### play_by_play

One row per play (one row per enriched ESPN play dict in `g["plays"]`, bound across every game in the compiled season).

| col_name | col_type | col_description |
| --- | --- | --- |
| game_play_number | integer | Sequential play index within the game (1..N). |
| id | double | ESPN unique play id (e.g. 401628455102926001); exceeds 32-bit int range so it is a double. |
| sequenceNumber | integer | ESPN play sequence ordering value within the game. |
| text | character | Full play description text. |
| awayScore | integer | Away team score after this play. |
| homeScore | integer | Home team score after this play. |
| scoringPlay | logical | TRUE if the play resulted in a score (ESPN flag). |
| priority | logical | ESPN priority/highlight flag for the play. |
| modified | character | ISO timestamp of when ESPN last modified the play record. |
| wallclock | character | ISO 8601 wall-clock timestamp the play occurred. |
| teamParticipants | list | List of participant objects ({team $ref, id, order, type=offense/defense}). |
| isPenalty | logical | TRUE if ESPN classifies the play as a penalty. |
| statYardage | integer | ESPN stat yardage for the play. |
| isTurnover | logical | TRUE if ESPN classifies the play as a turnover. |
| type.id | character | ESPN play-type id code. |
| type.text | character | ESPN play-type label (e.g. "Rush", "Pass Reception", "Field Goal Good"). |
| type.abbreviation | character | ESPN play-type abbreviation (sparse/null on many plays). |
| period.number | integer | Quarter/period number of the play. |
| clock.displayValue | character | Game clock at play start, MM:SS display string. |
| start.down | integer | Down at the start of the play. |
| start.distance | integer | Yards to go for a first down at play start. |
| start.yardLine | integer | Yard line (ESPN 0-100 field) at play start. |
| start.yardsToEndzone | integer | Yards to the opponent end zone at play start. |
| start.downDistanceText | character | Down-and-distance text at play start (e.g. "1st & 10"); null on non-scrimmage plays. |
| start.shortDownDistanceText | character | Abbreviated down-and-distance text at play start; null on non-scrimmage plays. |
| start.possessionText | character | Field-position possession text at play start; null on non-scrimmage plays. |
| start.team.id | integer | ESPN team id in possession reported at play start. |
| end.down | integer | Down at the end of the play. |
| end.distance | integer | Yards to go for a first down at play end. |
| end.yardLine | integer | Yard line at play end. |
| end.yardsToEndzone | integer | Yards to the opponent end zone at play end. |
| end.downDistanceText | character | Down-and-distance text at play end; null on non-scrimmage plays. |
| end.shortDownDistanceText | character | Abbreviated down-and-distance text at play end; null on non-scrimmage plays. |
| end.possessionText | character | Field-position possession text at play end; null on non-scrimmage plays. |
| end.team.id | integer | ESPN team id in possession reported at play end. |
| scoringType.name | character | Scoring type machine name (e.g. "touchdown","field-goal"); null on non-scoring plays. |
| scoringType.displayName | character | Scoring type display name; null on non-scoring plays. |
| scoringType.abbreviation | character | Scoring type abbreviation (TD, FG, etc.); null on non-scoring plays. |
| pointAfterAttempt.id | double | PAT attempt id; null unless the play was a try. |
| pointAfterAttempt.text | character | PAT attempt description; null unless a try. |
| pointAfterAttempt.abbreviation | character | PAT attempt abbreviation; null unless a try. |
| pointAfterAttempt.value | double | PAT point value (1/2); null unless a try. |
| drive.id | character | ESPN drive id this play belongs to. |
| drive.displayResult | character | Human-readable drive result (e.g. "Touchdown","Punt"). |
| drive.isScore | logical | TRUE if the drive ended in a score. |
| drive.team.shortDisplayName | character | Offensive team short display name for the drive. |
| drive.team.displayName | character | Offensive team full display name for the drive. |
| drive.team.name | character | Offensive team nickname for the drive. |
| drive.team.abbreviation | character | Offensive team abbreviation for the drive. |
| drive.yards | integer | Net yards gained on the drive. |
| drive.offensivePlays | integer | Number of offensive plays in the drive. |
| drive.result | character | Drive result code/label. |
| drive.description | character | Drive summary description (plays, yards, time). |
| drive.shortDisplayResult | character | Abbreviated drive result. |
| drive.timeElapsed.displayValue | character | Time elapsed during the drive (MM:SS). |
| drive.start.period.number | integer | Period the drive started. |
| drive.start.period.type | character | Period type at drive start (e.g. "quarter"). |
| drive.start.yardLine | integer | Yard line where the drive started. |
| drive.start.clock.displayValue | character | Clock when the drive started (MM:SS). |
| drive.start.text | character | Drive-start field-position text. |
| drive.end.period.number | integer | Period the drive ended. |
| drive.end.period.type | character | Period type at drive end. |
| drive.end.yardLine | integer | Yard line where the drive ended. |
| drive.end.clock.displayValue | character | Clock when the drive ended (MM:SS); may be null. |
| game_id | integer | ESPN game id (stamped identity; embedded in every play). |
| season | integer | Season year (stamped identity). |
| seasonType | integer | ESPN season-type code (1=pre, 2=regular, 3=post). |
| week | integer | Season week number (stamped identity). |
| status_type_completed | logical | TRUE if the game status is completed. |
| homeTeamId | integer | ESPN home team id. |
| awayTeamId | integer | ESPN away team id. |
| homeTeamName | character | Home team location/name. |
| awayTeamName | character | Away team location/name. |
| homeTeamMascot | character | Home team mascot/nickname. |
| awayTeamMascot | character | Away team mascot/nickname. |
| homeTeamAbbrev | character | Home team abbreviation. |
| awayTeamAbbrev | character | Away team abbreviation. |
| homeTeamNameAlt | character | Alternate home team name. |
| awayTeamNameAlt | character | Alternate away team name. |
| gameSpread | double | Pregame point spread for the game. |
| homeFavorite | logical | TRUE if the home team is the betting favorite. |
| gameSpreadAvailable | logical | TRUE if a spread was available for the game. |
| overUnder | double | Pregame over/under total. |
| homeTeamSpread | double | Spread expressed from the home team's perspective. |
| clock.minutes | integer | Minutes component of the game clock at play start. |
| clock.seconds | integer | Seconds component of the game clock at play start. |
| half | integer | Half of play (1 or 2; OT continues numbering). |
| lag_half | integer | Half value of the previous play; null on first play. |
| lead_half | integer | Half value of the next play. |
| start.TimeSecsRem | integer | Seconds remaining in the half at play start. |
| start.adj_TimeSecsRem | integer | Adjusted seconds remaining in game at play start (EPA model input). |
| orig_play_type | character | Original ESPN play-type label before any mid-pipeline reclassification. |
| lead_text | character | Description text of the next play. |
| lead_start_team | character | Possessing team text on the next play. |
| lead_start_yardsToEndzone | integer | Yards to end zone at the start of the next play. |
| lead_start_down | integer | Down at the start of the next play. |
| lead_start_distance | integer | Distance to go at the start of the next play. |
| lead_scoringPlay | logical | Whether the next play is a scoring play. |
| text_dupe | logical | TRUE if the play text duplicates an adjacent play (dedupe flag). |
| start.pos_team.id | integer | Offensive (possession) team id at play start. |
| start.def_pos_team.id | integer | Defensive team id at play start. |
| end.def_pos_team.id | integer | Defensive team id at play end. |
| end.pos_team.id | integer | Offensive (possession) team id at play end. |
| start.pos_team.name | character | Offensive team name at play start. |
| start.def_pos_team.name | character | Defensive team name at play start. |
| end.pos_team.name | character | Offensive team name at play end. |
| end.def_pos_team.name | character | Defensive team name at play end. |
| start.is_home | logical | TRUE if the possession team at play start is the home team. |
| end.is_home | logical | TRUE if the possession team at play end is the home team. |
| homeTimeoutCalled | logical | TRUE if the home team called a timeout on this play. |
| awayTimeoutCalled | logical | TRUE if the away team called a timeout on this play. |
| end.homeTeamTimeouts | integer | Home timeouts remaining at play end. |
| end.awayTeamTimeouts | integer | Away timeouts remaining at play end. |
| start.homeTeamTimeouts | integer | Home timeouts remaining at play start. |
| start.awayTeamTimeouts | integer | Away timeouts remaining at play start. |
| end.TimeSecsRem | integer | Seconds remaining in the half at play end. |
| end.adj_TimeSecsRem | integer | Adjusted seconds remaining in game at play end. |
| start.posTeamTimeouts | integer | Offensive team timeouts remaining at play start. |
| start.defPosTeamTimeouts | integer | Defensive team timeouts remaining at play start. |
| end.posTeamTimeouts | integer | Offensive team timeouts remaining at play end. |
| end.defPosTeamTimeouts | integer | Defensive team timeouts remaining at play end. |
| firstHalfKickoffTeamId | integer | Team id that received/kicked the opening kickoff (used for 2H possession logic). |
| period | integer | Quarter/period number (duplicate of period.number, flattened). |
| start.yard | integer | Absolute yard (0-100) at play start. |
| end.yard | integer | Absolute yard (0-100) at play end. |
| lag_scoringPlay | logical | Whether the previous play was a scoring play; null on first play. |
| end_of_half | logical | TRUE if the play ends the half; null where undefined. |
| down_1 | logical | TRUE if the play started on 1st down. |
| down_2 | logical | TRUE if the play started on 2nd down. |
| down_3 | logical | TRUE if the play started on 3rd down. |
| down_4 | logical | TRUE if the play started on 4th down. |
| down_1_end | logical | TRUE if the play ended on 1st down. |
| down_2_end | logical | TRUE if the play ended on 2nd down. |
| down_3_end | logical | TRUE if the play ended on 3rd down. |
| down_4_end | logical | TRUE if the play ended on 4th down. |
| scoring_play | logical | Derived scoring-play flag. |
| td_play | logical | TRUE if the play involved a touchdown. |
| touchdown | logical | TRUE if a touchdown was scored on the play. |
| td_check | logical | Internal touchdown-detection helper flag. |
| safety | logical | TRUE if a safety occurred. |
| fumble_vec | logical | TRUE if a fumble occurred on the play. |
| forced_fumble | logical | TRUE if the fumble was forced. |
| kickoff_play | logical | TRUE if the play is a kickoff. |
| kickoff_tb | logical | TRUE if the kickoff resulted in a touchback. |
| kickoff_onside | logical | TRUE if the kickoff was an onside attempt. |
| kickoff_oob | logical | TRUE if the kickoff went out of bounds. |
| kickoff_fair_catch | logical | TRUE if the kickoff was fair-caught. |
| kickoff_downed | logical | TRUE if the kickoff was downed. |
| kick_play | logical | TRUE if the play is a kick (kickoff or punt) play. |
| kickoff_safety | logical | TRUE if the kickoff resulted in a safety. |
| punt | logical | TRUE if the play is a punt. |
| punt_play | logical | TRUE if the play is a punt play. |
| punt_tb | logical | TRUE if the punt resulted in a touchback. |
| punt_oob | logical | TRUE if the punt went out of bounds. |
| punt_fair_catch | logical | TRUE if the punt was fair-caught. |
| punt_downed | logical | TRUE if the punt was downed. |
| punt_safety | logical | TRUE if the punt resulted in a safety. |
| punt_blocked | logical | TRUE if the punt was blocked. |
| penalty_safety | logical | TRUE if a penalty resulted in a safety. |
| rush | logical | TRUE if the play is a rush. |
| pass | logical | TRUE if the play is a pass. |
| sack_vec | logical | TRUE if a sack occurred (vector form; canonical column is `sack`). |
| pos_team | integer | Offensive (possession) team id for the play. |
| def_pos_team | integer | Defensive team id for the play. |
| is_home | logical | TRUE if the possession team is the home team. |
| lag_HA_score_diff | integer | Home-minus-away score differential on the previous play; null on first. |
| HA_score_diff | integer | Home-minus-away score differential after this play. |
| net_HA_score_pts | integer | Net change in home-minus-away points on this play; null where undefined. |
| H_score_diff | integer | Home score differential; null where undefined. |
| A_score_diff | integer | Away score differential; null where undefined. |
| lag_homeScore | integer | Home score on the previous play. |
| lag_awayScore | integer | Away score on the previous play. |
| start.homeScore | integer | Home score at play start. |
| start.awayScore | integer | Away score at play start. |
| end.homeScore | integer | Home score at play end. |
| end.awayScore | integer | Away score at play end. |
| pos_team_score | integer | Offensive team score for the play. |
| def_pos_team_score | integer | Defensive team score for the play. |
| start.pos_team_score | integer | Offensive team score at play start. |
| start.def_pos_team_score | integer | Defensive team score at play start. |
| start.pos_score_diff | integer | Offense-minus-defense score differential at play start. |
| end.pos_team_score | integer | Offensive team score at play end. |
| end.def_pos_team_score | integer | Defensive team score at play end. |
| end.pos_score_diff | integer | Offense-minus-defense score differential at play end. |
| lag_pos_team | integer | Possession team id on the previous play. |
| lead_pos_team | integer | Possession team id on the next play; null at game end. |
| lead_pos_team2 | integer | Possession team id two plays ahead; null near game end. |
| pos_score_diff | integer | Offense-minus-defense score differential for the play. |
| lag_pos_score_diff | integer | Possession score differential on the previous play. |
| pos_score_pts | integer | Points scored on the play from the possession team's perspective (e.g. -7,+3,+7). |
| pos_score_diff_start | integer | Possession score differential at play start. |
| start.pos_team_receives_2H_kickoff | logical | TRUE if the possession team receives the second-half kickoff (at play start). |
| end.pos_team_receives_2H_kickoff | logical | TRUE if the possession team receives the second-half kickoff (at play end). |
| change_of_poss | logical | TRUE if possession changed on the play. |
| penalty_flag | logical | TRUE if a penalty was flagged. |
| penalty_declined | logical | TRUE if the penalty was declined. |
| penalty_no_play | logical | TRUE if the penalty negated the play. |
| penalty_offset | logical | TRUE if offsetting penalties occurred. |
| penalty_1st_conv | logical | TRUE if the penalty produced a first down. |
| penalty_in_text | logical | TRUE if a penalty is mentioned in the play text. |
| penalty_detail | character | Parsed penalty detail; null when no penalty. |
| penalty_text | character | Penalty text extracted from the play description; null when none. |
| yds_penalty | character | Penalty yardage (string as enriched; null when none). |
| sack | logical | TRUE if the play was a sack. |
| int | logical | TRUE if the play was an interception. |
| int_td | logical | TRUE if the interception was returned for a touchdown. |
| completion | logical | TRUE if the pass was completed. |
| pass_attempt | logical | TRUE if the play was a pass attempt. |
| target | logical | TRUE if a receiver was targeted. |
| pass_breakup | logical | TRUE if a pass breakup occurred. |
| pass_td | logical | TRUE if the play was a passing touchdown. |
| rush_td | logical | TRUE if the play was a rushing touchdown. |
| turnover_vec | logical | TRUE if a turnover occurred (vector form). |
| offense_score_play | logical | TRUE if the offense scored on the play. |
| defense_score_play | logical | TRUE if the defense scored on the play. |
| downs_turnover | logical | TRUE if possession changed on downs. |
| yds_punted | integer | Yards the ball was punted; null when not a punt. |
| yds_punt_gained | integer | Net punt yards gained; null when not a punt. |
| fg_attempt | logical | TRUE if the play was a field-goal attempt. |
| fg_made | logical | TRUE if the field goal was made. |
| yds_fg | integer | Field-goal distance in yards; null when not a FG. |
| pos_unit | character | Offensive unit label (e.g. offense/special teams). |
| def_pos_unit | character | Defensive unit label. |
| lead_play_type | character | Play type of the next play; null at game end. |
| sp | logical | TRUE if the play is a special-teams play. |
| play | logical | TRUE if the row is an actual play (vs administrative). |
| scrimmage_play | logical | TRUE if the play is a scrimmage (offensive) play. |
| change_of_pos_team | logical | TRUE if the possession team changed on the play. |
| pos_score_diff_end | integer | Possession score differential at play end. |
| fumble_lost | logical | TRUE if a fumble was lost. |
| fumble_recovered | logical | TRUE if a fumble was recovered. |
| yds_rushed | integer | Rushing yards on the play; null when not a rush. |
| yds_receiving | integer | Receiving yards on the play; null when not a reception. |
| yds_int_return | integer | Interception return yards; null when no INT. |
| yds_kickoff | integer | Kickoff distance in yards; null when not a kickoff. |
| yds_kickoff_return | integer | Kickoff return yards; null when no return. |
| yds_punt_return | integer | Punt return yards; null when no return. |
| yds_fumble_return | character | Fumble return yards; null in this fixture (no fumble return). |
| yds_sacked | integer | Yards lost on a sack; null when not a sack. |
| sack_players | character | Names of players credited with the sack; null when not a sack. |
| passer_player_name | character | Passer name; null when not a pass. |
| rusher_player_name | character | Rusher name; null when not a rush. |
| receiver_player_name | character | Targeted receiver name; null when not a reception. |
| sack_player_name | character | Primary sacking player name; null when not a sack. |
| sack_player_name2 | character | Second sacking player name; null in this fixture. |
| pass_breakup_player_name | character | Player credited with the pass breakup; null in this fixture. |
| interception_player_name | character | Intercepting player name; null in this fixture (no INT). |
| fg_kicker_player_name | character | Field-goal kicker name; null in this fixture. |
| fg_block_player_name | character | Player who blocked the field goal; null in this fixture. |
| fg_return_player_name | character | Player returning a blocked/missed field goal; null in this fixture. |
| kickoff_player_name | character | Kicker name on the kickoff; null when not a kickoff. |
| kickoff_return_player_name | character | Kickoff returner name; null in this fixture. |
| punter_player_name | character | Punter name; null when not a punt. |
| punt_block_player_name | character | Player who blocked the punt; null in this fixture. |
| punt_return_player_name | character | Punt returner name; null when not a return. |
| punt_block_return_player_name | character | Player returning a blocked punt; null in this fixture. |
| fumble_player_name | character | Player who fumbled; null when no fumble. |
| fumble_forced_player_name | character | Player who forced the fumble; null in this fixture. |
| fumble_recovered_player_name | character | Player who recovered the fumble; null when no fumble. |
| new_down | integer | Resulting down after the play. |
| new_distance | integer | Resulting distance to go after the play. |
| middle_8 | logical | TRUE if the play falls in the "middle 8" (last 4 min H1 / first 4 min H2). |
| rz_play | logical | TRUE if the play is in the red zone. |
| under_2 | logical | TRUE if under two minutes remain in the half. |
| goal_to_go | logical | TRUE if it is a goal-to-go situation. |
| scoring_opp | logical | TRUE if the play is in a scoring-opportunity area. |
| stuffed_run | logical | TRUE if a rush gained <= 0 yards. |
| stopped_run | logical | TRUE if a rush was stopped short (stuffed/limited). |
| opportunity_run | logical | TRUE if a rush reached the line of scrimmage cleanly (opportunity-rate metric). |
| highlight_run | logical | TRUE if the rush qualifies as a highlight (breakaway) run. |
| adj_rush_yardage | integer | Adjusted/capped rush yardage used in line-yard decomposition; null when not a rush. |
| line_yards | double | Offensive-line-credited rushing yards (line yards); null when not a rush. |
| second_level_yards | double | Rushing yards gained 5-10 yards past the line; null when not a rush. |
| open_field_yards | integer | Rushing yards gained 10+ yards past the line; null when not a rush. |
| highlight_yards | double | Highlight (breakaway) rushing yards credited to the back; null when not a rush. |
| opp_highlight_yards | double | Opponent-perspective highlight rushing yards; null when not applicable. |
| short_rush_success | logical | TRUE if a short-yardage rush succeeded; null when not applicable. |
| short_rush_attempt | logical | TRUE if the play was a short-yardage rush attempt; null when not applicable. |
| power_rush_success | logical | TRUE if a power-situation rush converted; null when not applicable. |
| power_rush_attempt | logical | TRUE if the play was a power-situation rush attempt; null when not applicable. |
| early_down | logical | TRUE if the play occurred on an early down (1st/2nd). |
| late_down | logical | TRUE if the play occurred on a late down (3rd/4th). |
| early_down_pass | logical | TRUE if an early-down pass. |
| early_down_rush | logical | TRUE if an early-down rush. |
| late_down_pass | logical | TRUE if a late-down pass. |
| late_down_rush | logical | TRUE if a late-down rush. |
| standard_down | logical | TRUE if the play is a standard down. |
| passing_down | logical | TRUE if the play is a passing down. |
| TFL | logical | TRUE if the play was a tackle for loss. |
| TFL_pass | logical | TRUE if a TFL on a pass play (e.g. sack). |
| TFL_rush | logical | TRUE if a TFL on a rush play. |
| havoc | logical | TRUE if the play was a havoc event (TFL, PBU, forced fumble, INT). |
| start.pos_team_spread | double | Possession-team spread at play start. |
| start.elapsed_share | double | Share of game time elapsed at play start (0-1). |
| start.spread_time | double | Spread x time-remaining interaction term at play start (WP model input). |
| end.pos_team_spread | double | Possession-team spread at play end. |
| end.elapsed_share | double | Share of game time elapsed at play end (0-1). |
| end.spread_time | double | Spread x time-remaining interaction term at play end. |
| down | integer | Down for the play (canonical). |
| distance | integer | Distance to go for the play (canonical). |
| start.yardsToEndzone.touchback | integer | Yards to end zone assuming a touchback at play start. |
| EP_start_touchback | double | Expected points at the start assuming a touchback. |
| EP_start | double | Expected points at the start of the play. |
| EP_end | double | Expected points at the end of the play. |
| lag_EP_end | double | Expected points at the end of the previous play; null on first. |
| lag_change_of_pos_team | logical | Whether possession changed on the previous play. |
| EP_between | double | Expected-points adjustment between plays; null where undefined. |
| EPA | double | Expected points added on the play. |
| def_EPA | double | Defensive EPA (negated offensive EPA). |
| EPA_scrimmage | double | EPA restricted to scrimmage plays; null otherwise. |
| EPA_rush | double | EPA on rush plays; null otherwise. |
| EPA_pass | double | EPA on pass plays; null otherwise. |
| EPA_explosive | logical | TRUE if the play was an explosive play by EPA. |
| EPA_non_explosive | double | EPA on non-explosive plays; null otherwise. |
| EPA_explosive_pass | logical | TRUE if an explosive pass by EPA. |
| EPA_explosive_rush | logical | TRUE if an explosive rush by EPA. |
| first_down_created | logical | TRUE if the play created a first down. |
| EPA_success | logical | TRUE if the play was successful by EPA (>0). |
| EPA_success_early_down | logical | TRUE if a successful early-down play by EPA. |
| EPA_success_early_down_pass | logical | TRUE if a successful early-down pass by EPA. |
| EPA_success_early_down_rush | logical | TRUE if a successful early-down rush by EPA. |
| EPA_success_late_down | logical | TRUE if a successful late-down play by EPA. |
| EPA_success_late_down_pass | logical | TRUE if a successful late-down pass by EPA. |
| EPA_success_late_down_rush | logical | TRUE if a successful late-down rush by EPA. |
| EPA_success_standard_down | logical | TRUE if a successful standard-down play by EPA. |
| EPA_success_passing_down | logical | TRUE if a successful passing-down play by EPA. |
| EPA_success_pass | logical | TRUE if a successful pass by EPA. |
| EPA_success_rush | logical | TRUE if a successful rush by EPA. |
| EPA_success_EPA | double | EPA value on successful plays; null otherwise. |
| EPA_success_standard_down_EPA | double | EPA value on successful standard-down plays; null otherwise. |
| EPA_success_passing_down_EPA | double | EPA value on successful passing-down plays; null otherwise. |
| EPA_success_pass_EPA | double | EPA value on successful passes; null otherwise. |
| EPA_success_rush_EPA | double | EPA value on successful rushes; null otherwise. |
| EPA_middle_8_success | logical | TRUE if a successful middle-8 play by EPA. |
| EPA_middle_8_success_pass | logical | TRUE if a successful middle-8 pass by EPA. |
| EPA_middle_8_success_rush | logical | TRUE if a successful middle-8 rush by EPA. |
| EPA_penalty | double | EPA attributed to penalty plays; null otherwise. |
| EPA_sp | double | EPA on special-teams plays. |
| EPA_fg | double | EPA on field-goal plays; null otherwise. |
| EPA_punt | double | EPA on punt plays; null otherwise. |
| EPA_kickoff | double | EPA on kickoff plays; null otherwise. |
| start.ExpScoreDiff_touchback | double | Expected score differential at play start assuming a touchback. |
| start.ExpScoreDiff | double | Expected score differential at play start (WP model input). |
| start.ExpScoreDiff_Time_Ratio_touchback | double | ExpScoreDiff/time ratio at play start assuming a touchback. |
| start.ExpScoreDiff_Time_Ratio | double | ExpScoreDiff/time ratio at play start. |
| end.ExpScoreDiff | double | Expected score differential at play end. |
| end.ExpScoreDiff_Time_Ratio | double | ExpScoreDiff/time ratio at play end. |
| wp_before | double | Win probability for the possession team before the play. |
| wp_touchback | double | Win probability assuming a touchback. |
| wp_after | double | Win probability for the possession team after the play. |
| def_wp_before | double | Defensive team win probability before the play. |
| home_wp_before | double | Home team win probability before the play. |
| away_wp_before | double | Away team win probability before the play. |
| lead_wp_before | double | Win probability before the next play; null at game end. |
| lead_wp_before2 | double | Win probability before two plays ahead; null near game end. |
| def_wp_after | double | Defensive team win probability after the play. |
| home_wp_after | double | Home team win probability after the play. |
| away_wp_after | double | Away team win probability after the play. |
| wpa | double | Win probability added on the play. |
| drive_start | double | Drive-start field-position / time marker. |
| drive_stopped | logical | TRUE if the drive was stopped (no score). |
| drive_play_index | integer | Index of this play within its drive. |
| drive_offense_plays | integer | Number of offensive plays in the drive. |
| prog_drive_EPA | double | Cumulative (progressive) EPA over the drive up to this play; null early. |
| prog_drive_WPA | double | Cumulative (progressive) WPA over the drive up to this play. |
| drive_offense_yards | integer | Cumulative offensive yards in the drive. |
| drive_total_yards | integer | Total drive yardage. |
| qbr_epa | double | QBR-style EPA contribution for the play. |
| weight | double | Play weight used in QBR/EPA aggregation. |
| non_fumble_sack | logical | TRUE if a sack occurred without a fumble. |
| sack_epa | double | EPA attributed to sacks; null otherwise. |
| pass_epa | double | EPA attributed to passes (QBR split); null otherwise. |
| rush_epa | double | EPA attributed to rushes (QBR split); null otherwise. |
| pen_epa | double | EPA attributed to penalties (QBR split); null otherwise. |
| sack_weight | double | QBR weight for sack component; null otherwise. |
| pass_weight | double | QBR weight for pass component; null otherwise. |
| rush_weight | double | QBR weight for rush component; null otherwise. |
| pen_weight | double | QBR weight for penalty component; null otherwise. |
| action_play | logical | TRUE if the row is an action (non-administrative) play. |
| athlete_name | character | Primary athlete name associated with the play; null when none. |

_Release tag: `espn_cfb_pbp`_

---

### team_box

One row per team (2 per game) — `g["boxscore"]["teams"][i]` reshaped so each ESPN team stat `name` becomes a column holding that stat's `displayValue`, plus team identity, home/away, and stamped identity columns.

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game identifier, stamped from the source JSON (`gameId`/`header.id`); identical for both team rows of a game. |
| season | integer | Season year, stamped from `season.year` / `header.season.year` (e.g. 2024). |
| week | integer | Season week number, stamped from the game's `week` (e.g. 1). |
| team_id | integer | ESPN team id from `team.id` (integer-valued string in JSON, e.g. 2006, 194). |
| team_display_name | character | Team display name from `team.displayName` (e.g. "Akron Zips", "Ohio State Buckeyes"). |
| home_away | character | Whether this team is "home" or "away" for the game, from `homeAway`. |
| first_downs | character | Total first downs (stat `firstDowns`, label "1st Downs"); displayValue is an integer-as-string. |
| third_down_eff | character | Third-down conversion efficiency (stat `thirdDownEff`, label "3rd down efficiency") as a "made-attempts" string, e.g. "4-16". |
| fourth_down_eff | character | Fourth-down conversion efficiency (stat `fourthDownEff`, label "4th down efficiency") as a "made-attempts" string, e.g. "2-3". |
| total_yards | character | Total offensive yards (stat `totalYards`, label "Total Yards"); integer-as-string. |
| net_passing_yards | character | Net passing yards (stat `netPassingYards`, label "Passing"); integer-as-string. |
| completion_attempts | character | Pass completions over attempts (stat `completionAttempts`, label "Comp/Att") as a "comp/att" string, e.g. "18/29". |
| yards_per_pass | character | Yards gained per pass attempt (stat `yardsPerPass`, label "Yards per pass") as a decimal string, e.g. "4.5". |
| rushing_yards | character | Total rushing yards (stat `rushingYards`, label "Rushing"); integer-as-string. |
| rushing_attempts | character | Total rushing attempts (stat `rushingAttempts`, label "Rushing Attempts"); integer-as-string. |
| yards_per_rush_attempt | character | Yards gained per rush attempt (stat `yardsPerRushAttempt`, label "Yards per rush") as a decimal string, e.g. "1.3". |
| total_penalties_yards | character | Penalties committed and penalty yards (stat `totalPenaltiesYards`, label "Penalties") as a "count-yards" string, e.g. "5-35". |
| turnovers | character | Total turnovers committed (stat `turnovers`, label "Turnovers"); integer-as-string. |
| fumbles_lost | character | Fumbles lost (stat `fumblesLost`, label "Fumbles lost"); integer-as-string. |
| interceptions | character | Interceptions thrown (stat `interceptions`, label "Interceptions thrown"); integer-as-string. |
| possession_time | character | Time of possession (stat `possessionTime`, label "Possession") as an "MM:SS" string, e.g. "34:03". |

_Release tag: `espn_cfb_team_box`_

---

### player_box

One row per player per stat category (athlete x category) per game, bound across all games in a season.

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game id, stamped from `g$header$id` (e.g. 401628455). Same for every row of one game. |
| season | integer | Season year, stamped from `g$header$season$year` (e.g. 2024). |
| week | integer | Season week, stamped from `g$header$week` (e.g. 1). Added by season-compile loaders; expected per endpoint shape. |
| team_id | integer | ESPN team id of the team the player belongs to, from `players[].team$id` (e.g. 2006). |
| category | character | Stat group name from `statistics[]$name`: one of passing, rushing, receiving, fumbles, defensive, interceptions, kickReturns, puntReturns, kicking, punting. Identifies which key block this row populates. |
| athlete_id | integer | ESPN athlete id from `$athlete$id`. Integer-valued though delivered as a string upstream. |
| athlete_name | character | Athlete display name from `$athlete$displayName`. |
| completions_passing_attempts | character | passing C/ATT — completions/attempts as a single slash-joined string (e.g. "9/13"). |
| passing_yards | character | passing YDS — passing yards (e.g. "68"). |
| yards_per_pass_attempt | character | passing AVG — yards per pass attempt (e.g. "5.2"). |
| passing_touchdowns | character | passing TD — passing touchdowns. |
| interceptions | character | Shared key: passing INT (interceptions thrown by passer) OR interceptions category INT (interceptions made on defense), depending on the row's category. |
| adj_qbr | character | passing QBR — ESPN Adjusted Total QBR for the passer (e.g. "74.0"). |
| rushing_attempts | character | rushing CAR — rushing attempts (carries). |
| rushing_yards | character | rushing YDS — rushing yards. |
| yards_per_rush_attempt | character | rushing AVG — yards per rush attempt. |
| rushing_touchdowns | character | rushing TD — rushing touchdowns. |
| long_rushing | character | rushing LONG — longest run, in yards. |
| receptions | character | receiving REC — receptions. |
| receiving_yards | character | receiving YDS — receiving yards. |
| yards_per_reception | character | receiving AVG — yards per reception. |
| receiving_touchdowns | character | receiving TD — receiving touchdowns. |
| long_reception | character | receiving LONG — longest reception, in yards. |
| fumbles | character | fumbles FUM — total fumbles. |
| fumbles_lost | character | fumbles LOST — fumbles lost to the opponent. |
| fumbles_recovered | character | fumbles REC — fumbles recovered. |
| total_tackles | character | defensive TOT — total tackles. |
| solo_tackles | character | defensive SOLO — solo (unassisted) tackles. |
| sacks | character | defensive SACKS — sacks (may be fractional, e.g. "1.5"). |
| tackles_for_loss | character | defensive TFL — tackles for loss. |
| passes_defended | character | defensive PD — passes defended/broken up. |
| hurries | character | defensive QB HUR — quarterback hurries. |
| defensive_touchdowns | character | defensive TD — defensive touchdowns scored. |
| interception_yards | character | interceptions YDS — return yards on interceptions. Empty in this fixture (0 athletes in the interceptions category); expected per endpoint shape. |
| interception_touchdowns | character | interceptions TD — touchdowns scored on interception returns. Empty in this fixture; expected per endpoint shape. |
| kick_returns | character | kickReturns NO — number of kick returns. Empty in this fixture (0 athletes); expected per endpoint shape. |
| kick_return_yards | character | kickReturns YDS — kick return yards. Empty in this fixture; expected. |
| yards_per_kick_return | character | kickReturns AVG — yards per kick return. Empty in this fixture; expected. |
| long_kick_return | character | kickReturns LONG — longest kick return. Empty in this fixture; expected. |
| kick_return_touchdowns | character | kickReturns TD — kick return touchdowns. Empty in this fixture; expected. |
| punt_returns | character | puntReturns NO — number of punt returns. Empty in this fixture (0 athletes); expected per endpoint shape. |
| punt_return_yards | character | puntReturns YDS — punt return yards. Empty in this fixture; expected. |
| yards_per_punt_return | character | puntReturns AVG — yards per punt return. Empty in this fixture; expected. |
| long_punt_return | character | puntReturns LONG — longest punt return. Empty in this fixture; expected. |
| punt_return_touchdowns | character | puntReturns TD — punt return touchdowns. Empty in this fixture; expected. |
| field_goals_made_field_goal_attempts | character | kicking FG — field goals made/attempts as a slash-joined string (e.g. "2/2"). |
| field_goal_pct | character | kicking PCT — field goal percentage (e.g. "100.0"). |
| long_field_goal_made | character | kicking LONG — longest field goal made, in yards. |
| extra_points_made_extra_point_attempts | character | kicking XP — extra points made/attempts as a slash-joined string (e.g. "0/0"). |
| total_kicking_points | character | kicking PTS — total kicking points scored. |
| punts | character | punting NO — number of punts. |
| punt_yards | character | punting YDS — total punt yards. |
| gross_avg_punt_yards | character | punting AVG — gross average yards per punt (e.g. "48.2"). |
| touchbacks | character | punting TB — punts resulting in touchbacks. |
| punts_inside20 | character | punting In 20 — punts downed inside the opponent 20-yard line. |
| long_punt | character | punting LONG — longest punt, in yards. |

_Release tag: `espn_cfb_player_box`_

---

### adv_team

One row per team per game (two rows per game), bound across all games in the compiled season.

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game id, stamped onto every row during the season-wide bind (from top-level `id`/`gameId`). |
| season | integer | Season year, stamped during the bind (from top-level `season`). |
| week | integer | Week number, stamped during the bind (from `header.week`/top-level `week`). |
| pos_team | integer | ESPN team id of the possessing/offensive team this row summarizes (the group-by key; cast to Int32 upstream). |
| rushing_highlight_yards_per_opp | double | Mean opportunity-adjusted highlight yards per opportunity rush (over rushes that were opportunity runs). |
| total_pen_yards | integer | Total penalty yardage (sum of statYardage on penalty-flagged plays). |
| EPA_penalty | double | Sum of EPA attributable to penalties. |
| penalty_first_downs_created | integer | Count of penalty-induced first-down conversions. |
| penalty_first_downs_created_rate | double | Mean (rate) of penalty-induced first-down conversions over penalty plays. |
| special_teams_plays | integer | Count of special-teams plays. |
| EPA_sp | double | Total special-teams EPA. |
| EPA_special_teams | double | Total special-teams EPA (alias of EPA_sp). |
| field_goals | integer | Count of field-goal attempts. |
| EPA_fg | double | Total EPA from field-goal attempts. |
| punt_plays | integer | Count of punt plays. |
| EPA_punt | double | Total EPA from punt plays. |
| kickoff_plays | integer | Count of kickoff plays. |
| EPA_kickoff | double | Total EPA from kickoff plays. |
| rushes | integer | Count of rushing scrimmage plays. |
| rush_yards | double | Total rushing yards (sum of yds_rushed). |
| yards_per_rush | double | Mean rushing yards per rush attempt. |
| rushing_power_rate | double | Rate of rushes that were power-run attempts. |
| rushing_first_downs_created | integer | First downs created on rushing scrimmage plays. |
| rushing_first_downs_created_rate | double | Mean (rate) of first downs created per rush. |
| EPA_rushing_overall | double | Total EPA on rushing scrimmage plays. |
| EPA_rushing_per_play | double | Mean EPA per rush. |
| EPA_explosive_rushing | integer | Count of explosive rushing plays (sum of EPA_explosive flag on rushes). |
| EPA_explosive_rushing_rate | double | Rate of explosive rushes. |
| EPA_non_explosive_rushing | double | Total EPA on non-explosive rushing plays. |
| EPA_non_explosive_rushing_per_play | double | Mean EPA per non-explosive rush. |
| passes | integer | Count of passing scrimmage plays. |
| pass_yards | double | Total receiving/passing yards (sum of yds_receiving). |
| yards_per_pass | double | Mean passing yards per pass play. |
| passing_first_downs_created | integer | First downs created on passing scrimmage plays. |
| passing_first_downs_created_rate | double | Mean (rate) of first downs created per pass. |
| EPA_passing_overall | double | Total EPA on passing scrimmage plays. |
| EPA_passing_per_play | double | Mean EPA per pass. |
| EPA_explosive_passing | integer | Count of explosive passing plays. |
| EPA_explosive_passing_rate | double | Rate of explosive passes. |
| EPA_non_explosive_passing | double | Total EPA on non-explosive passing plays. |
| EPA_non_explosive_passing_per_play | double | Mean EPA per non-explosive pass. |
| scrimmage_plays | integer | Count of scrimmage plays (offensive plays from scrimmage). |
| EPA_overall_off | double | Total offensive EPA on scrimmage plays. |
| EPA_overall_offense | double | Total offensive EPA on scrimmage plays (alias of EPA_overall_off). |
| EPA_per_play | double | Mean EPA per scrimmage play. |
| EPA_non_explosive | double | Total EPA on non-explosive scrimmage plays. |
| EPA_non_explosive_per_play | double | Mean EPA per non-explosive scrimmage play. |
| EPA_explosive | integer | Count of explosive scrimmage plays. |
| EPA_explosive_rate | double | Rate of explosive scrimmage plays. |
| passes_rate | double | Pass rate over scrimmage plays (mean of pass flag). |
| off_yards | integer | Total offensive yards on scrimmage plays (sum of statYardage). |
| total_off_yards | integer | Total offensive yards on scrimmage plays (alias of off_yards). |
| yards_per_play | double | Mean yards per scrimmage play. |
| EPA_plays | integer | Count of EPA-eligible plays (sum of play flag, all plays). |
| total_yards | integer | Total yardage across all plays (sum of statYardage, base box). |
| EPA_overall_total | double | Total EPA across all plays. |
| rushes_rate | double | Rush rate over scrimmage plays (mean of rush flag). |
| first_downs_created | integer | Total first downs created over scrimmage plays. |
| first_downs_created_rate | double | Mean (rate) of first downs created per scrimmage play. |
| EPA_rushing_power | double | Total EPA on power-run rushing attempts. |
| EPA_rushing_power_per_play | double | Mean EPA per power-run rushing attempt. |
| rushing_power_success | integer | Count of successful power-run rushes. |
| rushing_power_success_rate | double | Power-run success rate. |
| rushing_power | integer | Count of power-run rushing attempts. |
| rushing_stuff | integer | Count of rushes stuffed (no/negative gain at the line). |
| rushing_stuff_rate | double | Stuffed-run rate over rushes. |
| rushing_stopped | integer | Count of rushes stopped (short of expected gain). |
| rushing_stopped_rate | double | Stopped-run rate over rushes. |
| rushing_opportunity | integer | Count of opportunity runs (rushes reaching the opportunity-yardage threshold). |
| rushing_opportunity_rate | double | Opportunity-run rate over rushes. |
| rushing_highlight | integer | Count of highlight runs (rushes that broke into highlight-yardage territory). |
| rushing_highlight_rate | double | Highlight-run rate over rushes. |
| rushing_highlight_yards | double | Total highlight yards (yards gained beyond the line/second-level on highlight runs). |
| line_yards | double | Total line yards (offensive-line-credited rushing yardage). |
| line_yards_per_carry | double | Mean line yards per rush attempt. |
| second_level_yards | double | Total second-level rushing yards (5-10 yards past LOS). |
| open_field_yards | double | Total open-field rushing yards (beyond 10 yards past LOS). |

_Release tag: `espn_cfb_adv_team`_

---

### adv_passing

_Grain: one row per (team, passer) within a game; bound across a season, one row per passer per game._

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game id, stamped onto every row during the season-wide bind (source `gameId`, e.g. 401628455). |
| season | integer | Season year the game belongs to, stamped during the bind (e.g. 2024). |
| week | integer | Week number within the season, stamped during the bind (e.g. 1). |
| pos_team | integer | ESPN team id of the passer's offense (possession team); cast to Int32 in source. |
| passer_player_name | character | Passer's display name; literal `"TEAM"` for team-charged passing plays (e.g. intentional grounding) not attributed to an individual. |
| Comp | integer | Completions: sum of completed pass attempts for this passer in the game. |
| Att | integer | Pass attempts: sum of pass attempts for this passer. |
| Yds | double | Passing yards for this passer, aggregated from receiving yards on his attempts (can be 0.0). |
| Pass_TD | integer | Passing touchdowns thrown by this passer. |
| Int | integer | Interceptions thrown by this passer. |
| YPA | double | Yards per attempt: mean receiving yards across this passer's attempts. |
| EPA | double | Total Expected Points Added summed over this passer's plays. |
| EPA_per_Play | double | Mean EPA per play for this passer. |
| WPA | double | Total Win Probability Added summed over this passer's plays. |
| SR | double | Success rate: mean of the EPA-based success indicator (share of plays with positive EPA-success) for this passer. |
| Sck | integer | Sacks taken on this passer's dropbacks. |
| qbr_epa | double | Weighted clipped-EPA component used in the QBR computation (per-play qbr_epa weighted by play weight, summed and normalized) for this passer. |
| sack_epa | double | Weighted QBR-EPA contribution from non-fumble sack plays; NA when the passer took no qualifying sacks. |
| pass_epa | double | Weighted QBR-EPA contribution from pass plays for this passer. |
| rush_epa | double | Weighted QBR-EPA contribution from this passer's rush plays; NA when he had no rushes. |
| pen_epa | double | Weighted QBR-EPA contribution from penalty plays; NA when no penalty plays applied (often all-NA in a game). |
| spread | double | Pregame point spread from the passer's team perspective (start.pos_team_spread), used as a QBR model feature. |
| exp_qbr | double | Expected QBR (0-100) predicted by the XGBoost QBR model from the QBR-EPA features for this passer. |

_Release tag: `espn_cfb_adv_passing`_

---

### adv_rushing

One row per rusher (per team, per game).

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game id stamped by the season-compile reshape (from g["id"]/g["gameId"], e.g. 401628455); identifies the game each rusher row belongs to. |
| season | integer | Season year stamped by the reshape (from g["season"], e.g. 2024). |
| week | integer | Week number stamped by the reshape (from g["week"], e.g. 1); present because rows are bound across a full season. |
| pos_team | integer | ESPN team id of the rusher's offense (possession team); cast to Int32 in the source. Distinct per game (e.g. 194, 2006). |
| rusher_player_name | character | Rusher's full name (whitespace-stripped). May be null upstream when the play participant name is missing. |
| Car | integer | Carries: count of rush plays attributed to the rusher (sum of the per-play rush flag). |
| Yds | double | Total rushing yards (sum of per-play yds_rushed). |
| Rush_TD | integer | Rushing touchdowns (sum of the per-play rush_td flag). |
| YPC | double | Yards per carry: mean of per-play yds_rushed, rounded to 2 decimals. |
| EPA | double | Total Expected Points Added on the rusher's rush plays (sum of per-play EPA). |
| EPA_per_Play | double | Mean Expected Points Added per rush play, rounded to 2 decimals. |
| WPA | double | Total Win Probability Added on the rusher's rush plays (sum of per-play wpa). |
| SR | double | Success rate: mean of the per-play EPA_success indicator (fraction of carries that were successful). |
| Fum | integer | Fumbles on rush plays (sum of per-play fumble_vec). |
| Fum_Lost | integer | Fumbles lost on rush plays (sum of per-play fumble_lost). |

_Release tag: `espn_cfb_adv_rushing`_

---

### adv_receiving

One row per receiver (per team, per game) — bound across all games in the compiled season.

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game/event ID, stamped onto every row during the season-wide reshape. |
| season | integer | Season year (e.g. 2024, 2026), stamped during the reshape. |
| pos_team | integer | ESPN team ID of the receiver's (offensive/possession) team for this game. |
| receiver_player_name | character | Receiver's player name (trimmed). May be NA when the target play has no identifiable receiver. |
| Rec | integer | Receptions — sum of completed catches credited to the receiver (`completion`). |
| Tar | integer | Targets — number of pass attempts thrown to this receiver (`target`). |
| Yds | double | Receiving yards — sum of `yds_receiving` over the receiver's targets. |
| Rec_TD | integer | Receiving touchdowns — sum of `pass_td` on plays to this receiver. |
| YPT | double | Yards per target — mean of `yds_receiving` across the receiver's targets, rounded to 2 decimals. |
| EPA | double | Total Expected Points Added — sum of play-level `EPA` over the receiver's targets, rounded to 2 decimals. |
| EPA_per_Play | double | EPA per play — mean play-level `EPA` across the receiver's targets, rounded to 2 decimals. |
| WPA | double | Total Win Probability Added — sum of `wpa` over the receiver's targets, rounded to 2 decimals. |
| SR | double | Success rate — mean of the binary `EPA_success` indicator across the receiver's targets, rounded to 2 decimals. |
| Fum | integer | Fumbles — sum of `fumble_vec` on plays involving this receiver. |
| Fum_Lost | integer | Fumbles lost — sum of `fumble_lost` on plays involving this receiver. |

_Release tag: `espn_cfb_adv_receiving`_

---

### adv_defensive

One row per defensive team unit (per game, bound across all games in a season).

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game ID, stamped onto every row from the top-level game JSON during the season-wide bind. |
| season | integer | Season year, stamped from the top-level game JSON. |
| week | integer | Week number within the season, stamped from the top-level game JSON. |
| def_pos_team | integer | ESPN team ID of the defending team (the unit charting these defensive stats); grouping key, cast to Int32 in source. |
| scrimmage_plays | integer | Count of scrimmage plays this defense was on the field for (denominator for havoc_total_rate). |
| TFL | integer | Tackles for loss recorded by this defense across all scrimmage plays. |
| TFL_pass | integer | Tackles for loss occurring on pass plays. |
| TFL_rush | integer | Tackles for loss occurring on rush plays. |
| havoc_total | integer | Total havoc events (TFLs, pass breakups, forced fumbles, interceptions) generated by this defense across all scrimmage plays. |
| havoc_total_rate | double | Total havoc rate: havoc_total divided by scrimmage_plays (proportion in [0,1]). |
| fumbles | integer | Forced fumbles created by this defense. |
| def_int | integer | Interceptions made by this defense. |
| drive_stopped_rate | double | Percentage of opponent drives this defense stopped, mean of the drive_stopped flag times 100, rounded to 2 decimals. |
| num_pass_plays | integer | Count of opponent pass plays this defense faced (denominator for the pass-havoc and sack rates). |
| havoc_total_pass | integer | Havoc events generated on pass plays. |
| havoc_total_pass_rate | double | Pass havoc rate: havoc_total_pass divided by num_pass_plays (proportion in [0,1]). |
| sacks | integer | Sacks recorded by this defense (sum of the sack vector on pass plays). |
| sacks_rate | double | Sack rate: sacks divided by num_pass_plays (proportion in [0,1]). |
| pass_breakups | integer | Pass breakups recorded by this defense on pass plays. |
| havoc_total_rush | integer | Havoc events generated on rush plays. |
| havoc_total_rush_rate | double | Rush havoc rate: havoc_total_rush divided by the defense's rush-play count faced (proportion in [0,1]). |

_Release tag: `espn_cfb_adv_defensive`_

---

### adv_turnover

One row per team (2 rows per game: index 0 = away, index 1 = home), bound across all games in a season.

| col_name | col_type | col_description |
| --- | --- | --- |
| pos_team | integer | Team id (ESPN team id) for the team this row's turnover stats belong to; the group-by key from the source aggregation (`group_by(["pos_team"])`). |
| pass_breakups | integer | Count of pass breakups by this team's defense (sum of the `pass_breakup` play flag over scrimmage plays). |
| fumbles_lost | integer | Count of fumbles this team lost (sum of the `fumble_lost` play flag); feeds the team's turnover total. |
| fumbles_recovered | integer | Count of fumbles this team recovered (sum of the `fumble_recovered` play flag). |
| total_fumbles | integer | Total fumbles involving this team (sum of the `fumble_vec` play flag); used in the expected-turnovers formula. |
| Int | integer | Interceptions thrown by this team (sum of the `int` play flag), coerced to integer; feeds turnover total and expected turnovers. |
| expected_turnovers | double | Model-expected turnovers for this team: `0.5 * total_fumbles + 0.22 * (pass_breakups + Int)`. |
| expected_turnover_margin | double | Expected turnover margin: the opponent's `expected_turnovers` minus this team's `expected_turnovers`. |
| turnovers | integer | Actual turnovers committed by this team: `fumbles_lost + Int`. |
| turnover_margin | integer | Actual turnover margin: opponent's `turnovers` minus this team's `turnovers`. |
| turnover_luck | double | Turnover luck: `5.0 * (turnover_margin - expected_turnover_margin)`, scaling the gap between actual and expected margin into a points-style figure. |
| game_id | integer | ESPN game id stamped from the source JSON `id`; identifies the game this team row belongs to. |
| season | integer | Season year stamped from the compiled JSON (e.g. 2024); identifies the season the row belongs to. |
| week | integer | Week number stamped from the compiled JSON; identifies the week within the season. |

_Release tag: `espn_cfb_adv_turnover`_

---

### adv_drives

One row per team (per `pos_team`) per game; bound across all games in a season.

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game id, stamped onto every row when the per-game `advBoxScore.drives` block is bound across the season. |
| season | integer | Season (year) of the compiled dataset, stamped onto every row at bind time. |
| pos_team | integer | ESPN team id of the team in possession (drive-owning team); cast to Int32 in the source reshape. |
| drive_total_available_yards | double | Sum of `drive_start` (yards to the opponent end zone at the start of each scrimmage play) across the team's scrimmage plays — the total field-position-derived available yardage. |
| drive_total_gained_yards | integer | Sum of `drive.yards` (yards gained per drive) over the team's scrimmage plays — total yards gained on drives. |
| avg_field_position | double | Mean of `drive_start` across scrimmage plays — the team's average starting field position (yards from the opponent end zone). |
| plays_per_drive | double | Mean of `drive.offensivePlays` — average number of offensive plays per drive. |
| yards_per_drive | double | Mean of `drive.yards` — average yards gained per drive. |
| drives | integer | Count of distinct `drive.id` — number of unique drives by the team. |
| drive_total_gained_yards_rate | double | 100 * `drive_total_gained_yards` / `drive_total_available_yards` — percentage of available drive yardage actually gained. |

_Release tag: `espn_cfb_adv_drives`_

---

### adv_situational

One row per team (per game; the season release binds two rows per game, one per `pos_team`, across all games).

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game id, stamped by the R reshape from the source JSON; join key back to the schedule/other adv box blocks. |
| season | integer | Season year of the game, stamped by the R reshape (e.g. 2026). |
| week | integer | Season week of the game, stamped by the R reshape (where available from the game header). |
| pos_team | integer | ESPN team id of the team in possession (offense) these situational splits are computed for. |
| EPA_success | integer | Count of successful scrimmage plays (EPA > 0) for the team. |
| EPA_success_rate | double | Success rate over all scrimmage plays (mean of the EPA-success indicator). |
| EPA_success_pass | integer | Count of successful pass plays (EPA > 0). |
| EPA_success_pass_rate | double | Success rate over pass plays. |
| EPA_success_rush | integer | Count of successful rush plays (EPA > 0). |
| EPA_success_rush_rate | double | Success rate over rush plays. |
| EPA_success_rz | integer | Count of successful red-zone scrimmage plays (start yard line <= 20, EPA > 0); NA when the team had no red-zone scrimmage plays. |
| EPA_success_rate_rz | double | Red-zone success rate; NA when the team had no red-zone scrimmage plays. |
| EPA_success_third | integer | Count of successful 3rd-down scrimmage plays (EPA > 0). |
| EPA_success_rate_third | double | 3rd-down success rate. |
| EPA_success_early_down | integer | Count of successful early-down (1st/2nd) scrimmage plays (EPA > 0). |
| EPA_success_early_down_rate | double | Early-down success rate. |
| early_downs | integer | Number of early-down (1st/2nd) scrimmage plays. |
| early_down_pass_rate | double | Pass share (pass-play rate) on early downs. |
| early_down_rush_rate | double | Rush share (rush-play rate) on early downs. |
| EPA_early_down | double | Total EPA accumulated on early downs. |
| EPA_early_down_per_play | double | Mean EPA per early-down play. |
| early_down_first_down | integer | Count of early-down plays that created a first down. |
| early_down_first_down_rate | double | Rate of early-down plays that created a first down. |
| early_down_pass | integer | Number of early-down pass plays. |
| EPA_early_down_pass | double | Total EPA on early-down pass plays. |
| EPA_early_down_pass_per_play | double | Mean EPA per early-down pass play. |
| EPA_success_early_down_pass | integer | Count of successful early-down pass plays (EPA > 0). |
| EPA_success_early_down_pass_rate | double | Success rate on early-down pass plays. |
| early_down_rush | integer | Number of early-down rush plays. |
| EPA_early_down_rush | double | Total EPA on early-down rush plays. |
| EPA_early_down_rush_per_play | double | Mean EPA per early-down rush play. |
| EPA_success_early_down_rush | integer | Count of successful early-down rush plays (EPA > 0). |
| EPA_success_early_down_rush_rate | double | Success rate on early-down rush plays. |
| middle_8 | integer | Number of "middle 8" scrimmage plays (final 4 min of Q2 + first 4 min of Q3; adjusted seconds remaining 1560-2040). |
| middle_8_pass_rate | double | Pass share on middle-8 plays. |
| middle_8_rush_rate | double | Rush share on middle-8 plays. |
| EPA_middle_8 | double | Total EPA accumulated in the middle 8. |
| EPA_middle_8_per_play | double | Mean EPA per middle-8 play. |
| EPA_middle_8_success | integer | Count of successful middle-8 plays (EPA > 0). |
| EPA_middle_8_success_rate | double | Success rate in the middle 8. |
| middle_8_pass | integer | Number of middle-8 pass plays. |
| EPA_middle_8_pass | double | Total EPA on middle-8 pass plays. |
| EPA_middle_8_pass_per_play | double | Mean EPA per middle-8 pass play. |
| EPA_middle_8_success_pass | integer | Count of successful middle-8 pass plays (EPA > 0). |
| EPA_middle_8_success_pass_rate | double | Success rate on middle-8 pass plays. |
| middle_8_rush | integer | Number of middle-8 rush plays. |
| EPA_middle_8_rush | double | Total EPA on middle-8 rush plays. |
| EPA_middle_8_rush_per_play | double | Mean EPA per middle-8 rush play. |
| EPA_middle_8_success_rush | integer | Count of successful middle-8 rush plays (EPA > 0). |
| EPA_middle_8_success_rush_rate | double | Success rate on middle-8 rush plays. |
| EPA_success_late_down | integer | Count of successful late-down (3rd/4th) plays (EPA > 0). |
| EPA_success_late_down_pass | integer | Count of successful late-down pass plays (EPA > 0). |
| EPA_success_late_down_rush | integer | Count of successful late-down rush plays (EPA > 0). |
| late_downs | integer | Number of late-down (3rd/4th) scrimmage plays. |
| late_down_pass | integer | Number of late-down pass plays. |
| late_down_rush | integer | Number of late-down rush plays. |
| EPA_late_down | double | Total EPA accumulated on late downs. |
| EPA_late_down_per_play | double | Mean EPA per late-down play. |
| EPA_success_late_down_rate | double | Late-down success rate (mean of the late-down success indicator). |
| EPA_success_late_down_pass_rate | double | Late-down pass success rate. |
| EPA_success_late_down_rush_rate | double | Late-down rush success rate. |
| late_down_pass_rate | double | Pass share on late downs. |
| late_down_rush_rate | double | Rush share on late downs. |
| EPA_success_standard_down | integer | Count of successful standard-down plays (EPA > 0). |
| EPA_success_standard_down_rate | double | Standard-down success rate. |
| EPA_standard_down | double | Total EPA accumulated on standard downs. |
| EPA_standard_down_per_play | double | Mean EPA per standard-down play. |
| standard_downs | integer | Number of standard-down scrimmage plays (down/distance situations favoring the offense). |
| EPA_success_passing_down | integer | Count of successful passing-down plays (EPA > 0). |
| EPA_success_passing_down_rate | double | Passing-down success rate. |
| EPA_passing_down | double | Total EPA accumulated on passing downs. |
| EPA_passing_down_per_play | double | Mean EPA per passing-down play. |
| passing_downs | integer | Number of passing-down scrimmage plays (obvious passing down/distance situations). |

_Release tag: `espn_cfb_adv_situational`_

---

### play_participants

One row per (game, play). Participant types are pivoted wide into per-type columns; this is NOT one-row-per-athlete (the sdv-py long frame is already pivoted). Column families are data-driven — only participant types present in the game appear.

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game/event identifier, present on every participant row. |
| season | integer | Season (year), stamped from the top-level JSON during the season-bind reshape. |
| week | integer | Week number, stamped from the top-level JSON during the season-bind reshape. |
| play_id | double | ESPN play identifier (18-digit; joins to plays.id). Exceeds R 32-bit integer max, so lands as double. |
| passer_player_id | character | Athlete id of the first passer on the play; NA if no passer. |
| passer_player_name | character | Display name of the first passer on the play; NA if no passer/unresolved. |
| passer_player_ids | character | Stringified list of every passer athlete id on the play (e.g. "['123']", "[]"); serialized numpy-repr string in this JSON. |
| passer_player_names | character | Stringified list of every passer display name on the play; serialized numpy-repr string. |
| rusher_player_id | character | Athlete id of the first rusher on the play; NA if none. |
| rusher_player_name | character | Display name of the first rusher; NA if none/unresolved. |
| rusher_player_ids | character | Stringified list of every rusher athlete id on the play. |
| rusher_player_names | character | Stringified list of every rusher display name on the play. |
| receiver_player_id | character | Athlete id of the first receiver/target on the play; NA if none. |
| receiver_player_name | character | Display name of the first receiver; NA if none/unresolved. |
| receiver_player_ids | character | Stringified list of every receiver athlete id on the play. |
| receiver_player_names | character | Stringified list of every receiver display name on the play. |
| tackler_player_id | character | Athlete id of the first tackler on the play; NA if none. |
| tackler_player_name | character | Display name of the first tackler; NA if none/unresolved. |
| tackler_player_ids | character | Stringified list of every tackler athlete id (preserves multi-tackler plays). |
| tackler_player_names | character | Stringified list of every tackler display name on the play. |
| sacked_by_player_id | character | Athlete id of the first sacker on the play; NA if none. |
| sacked_by_player_name | character | Display name of the first sacker; NA if none/unresolved. |
| sacked_by_player_ids | character | Stringified list of every sacker athlete id (preserves split sacks). |
| sacked_by_player_names | character | Stringified list of every sacker display name on the play. |
| pass_defender_player_id | character | Athlete id of the first pass defender (PD/breakup) on the play; NA if none. |
| pass_defender_player_name | character | Display name of the first pass defender; NA if none/unresolved. |
| pass_defender_player_ids | character | Stringified list of every pass defender athlete id on the play. |
| pass_defender_player_names | character | Stringified list of every pass defender display name on the play. |
| kicker_player_id | character | Athlete id of the first kicker (kickoff/FG/XP kicker) on the play; NA if none. |
| kicker_player_name | character | Display name of the first kicker; NA if none/unresolved. |
| kicker_player_ids | character | Stringified list of every kicker athlete id on the play. |
| kicker_player_names | character | Stringified list of every kicker display name on the play. |
| punter_player_id | character | Athlete id of the first punter on the play; NA if none. |
| punter_player_name | character | Display name of the first punter; NA if none/unresolved. |
| punter_player_ids | character | Stringified list of every punter athlete id on the play. |
| punter_player_names | character | Stringified list of every punter display name on the play. |
| returner_player_id | character | Athlete id of the first returner (kick/punt/turnover return) on the play; NA if none. |
| returner_player_name | character | Display name of the first returner; NA if none/unresolved. |
| returner_player_ids | character | Stringified list of every returner athlete id (preserves multi-lateral returns). |
| returner_player_names | character | Stringified list of every returner display name on the play. |
| scorer_player_id | character | Athlete id of the first scorer on the play; NA if none. |
| scorer_player_name | character | Display name of the first scorer; NA if none/unresolved. |
| scorer_player_ids | character | Stringified list of every scorer athlete id on the play. |
| scorer_player_names | character | Stringified list of every scorer display name on the play. |
| pat_scorer_player_id | character | Athlete id of the first PAT (point-after) scorer on the play; NA if none. |
| pat_scorer_player_name | character | Display name of the first PAT scorer; NA if none/unresolved. |
| pat_scorer_player_ids | character | Stringified list of every PAT scorer athlete id on the play. |
| pat_scorer_player_names | character | Stringified list of every PAT scorer display name on the play. |
| penalized_player_id | character | Athlete id of the first penalized player on the play; NA if none. |
| penalized_player_name | character | Display name of the first penalized player; NA if none/unresolved. |
| penalized_player_ids | character | Stringified list of every penalized athlete id on the play. |
| penalized_player_names | character | Stringified list of every penalized display name on the play. |
| assisted_by_player_id | character | Athlete id of the first assisting player (assisted tackle/return) on the play; NA if none. |
| assisted_by_player_name | character | Display name of the first assisting player; NA if none/unresolved. |
| assisted_by_player_ids | character | Stringified list of every assisting athlete id on the play. |
| assisted_by_player_names | character | Stringified list of every assisting display name on the play. |

_Release tag: `espn_cfb_play_participants`_

---

### drives

_Grain: one row per drive (per game), bound across all games in a season._

| col_name | col_type | col_description |
| --- | --- | --- |
| id | character | ESPN drive identifier (string, e.g. "4016284551"); first 10 digits are the game_id followed by the drive sequence number. |
| description | character | Human-readable drive summary, e.g. "10 plays, 47 yards, 5:18". |
| team | list | Nested ESPN team object for the offense on this drive: id, name, abbreviation, displayName, shortDisplayName, and logos[]. Retained as a list-column (not flattened). |
| start | list | Nested drive-start object: period{type, number}, clock{displayValue}, yardLine (int), and text (e.g. "OSU 32"). Retained as a list-column. |
| end | list | Nested drive-end object: period{type, number}, clock{displayValue}, yardLine (int), and text (e.g. "OSU 36"). Retained as a list-column. |
| time_elapsed | list | Nested object holding the drive duration: displayValue (e.g. "0:51"). Retained as a list-column. |
| yards | integer | Net yards gained on the drive. |
| is_score | logical | TRUE if the drive ended in a score (touchdown or field goal), FALSE otherwise. |
| offensive_plays | integer | Number of offensive plays run on the drive. |
| result | character | Uppercase drive-result code, e.g. "PUNT", "FG", "TD", "DOWNS", "INT", "INT TD", "FUMBLE RETURN TD", "END OF HALF", "END OF GAME". |
| short_display_result | character | Short display form of the result, typically the same uppercase code as `result` (e.g. "PUNT", "FG", "TD"). |
| display_result | character | Title-case human-readable result label, e.g. "Punt", "Field Goal", "Touchdown", "Downs". |
| game_id | integer | ESPN game identifier stamped onto every drive row from the top-level gameId (e.g. 401628455). |
| season | integer | Season year stamped from header.season.year (e.g. 2024). |
| week | integer | Season week stamped from header.week (e.g. 1). |

_Release tag: `espn_cfb_drives`_

---

### game_rosters

One row per rostered athlete per game — the season-wide bind of each game's
`g["game_rosters"]` (the full per-game roster compilation).

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game id; present on every roster element and the join key back to the game (e.g. 401628455). |
| season | integer | Season year stamped from top-level `g["season"]` during the season-wide bind (e.g. 2024, 2026). |
| athlete_id | integer | ESPN athlete id (e.g. 4429071). |
| athlete_uid | character | ESPN athlete uid string, e.g. "s:20~l:23~a:4429071". |
| athlete_guid | character | ESPN athlete GUID (UUID form). |
| athlete_type | character | Athlete sport/type tag, e.g. "football". |
| first_name | character | Athlete first name. |
| last_name | character | Athlete last name. |
| full_name | character | Athlete full name, e.g. "Cody Simon". |
| athlete_display_name | character | Display name for the athlete (usually equals full_name). |
| display_name | character | Abbreviated display name, e.g. "C. Simon". |
| short_name | character | Short name, e.g. "C. Simon". |
| slug | character | URL slug for the athlete, e.g. "cody-simon". |
| weight | double | Listed weight in pounds. |
| display_weight | character | Human-readable weight, e.g. "230 lbs". |
| height | double | Listed height in inches. |
| display_height | character | Human-readable height, e.g. "6' 2\""; NaN/null where missing (coerces to character). |
| age | double | Athlete age in years. |
| date_of_birth | character | ISO date-of-birth timestamp, e.g. "2002-04-01T08:00Z"; NaN/null where missing. |
| jersey | character | Jersey number as text, e.g. "0". |
| jersey_right | character | Right-side/duplicate jersey number field as text. |
| linked | logical | Whether the athlete record is linked in ESPN's system. |
| active | logical | ESPN athlete active flag. |
| is_active | logical | Whether the team entry is active for this game. |
| is_all_star | logical | All-star flag (false for CFB). |
| valid | logical | Whether ESPN marked this roster entry valid. |
| starter | logical | Whether the athlete started the game. |
| did_not_play | logical | Whether the athlete did not play. |
| winner | logical | Whether the athlete's team won the game. |
| order | integer | Sort order of the athlete within the team roster listing. |
| home_away | character | "home" or "away" indicating the athlete's team side. |
| alternate_ids_sdr | character | SportsDataR/alternate athlete id, e.g. "4429071". |
| birth_place_city | character | Birthplace city. |
| birth_place_state | character | Birthplace state/province; NaN/null where missing. |
| birth_place_country | character | Birthplace country, e.g. "USA". |
| birth_country_alternate_id | character | Alternate id for the birth country, e.g. "1". |
| birth_country_abbreviation | character | Birth country abbreviation, e.g. "USA". |
| headshot_href | character | URL to the athlete headshot image; NaN/null where missing. |
| headshot_alt | character | Alt text for the headshot image; NaN/null where missing. |
| flag_href | character | URL to the athlete's country flag image. |
| flag_alt | character | Alt text for the flag image, e.g. "USA". |
| flag_rel | character | Stringified relation list for the flag, e.g. "['country-flag']". |
| experience_years | integer | Years of experience. |
| experience_display_value | character | Experience class label, e.g. "Senior". |
| experience_abbreviation | character | Experience abbreviation, e.g. "SR". |
| status_id | character | Athlete status id as text, e.g. "2". |
| status_name | character | Status name, e.g. "Inactive". |
| status_type | character | Status type slug, e.g. "inactive". |
| status_abbreviation | character | Status abbreviation, e.g. "Inactive". |
| hand_type | character | Handedness type; NaN/null where missing. |
| hand_abbreviation | character | Handedness abbreviation; NaN/null where missing. |
| hand_display_value | character | Handedness display value; NaN/null where missing. |
| athlete_href | character | ESPN core API athlete reference URL. |
| position_href | character | ESPN core API position reference URL for the athlete. |
| statistics_href | character | ESPN core API statistics reference URL; NaN/null where missing. |
| team_id | integer | ESPN team id for the athlete's team (e.g. 194). |
| team_guid | character | Team GUID. |
| team_uid | character | Team uid string, e.g. "s:20~l:23~t:194". |
| team_slug | character | Team slug, e.g. "ohio-state-buckeyes". |
| team_location | character | Team location/school name, e.g. "Ohio State". |
| team_name | character | Team mascot/nickname, e.g. "Buckeyes". |
| team_nickname | character | Team nickname field, e.g. "Ohio State". |
| team_abbreviation | character | Team abbreviation, e.g. "OSU". |
| team_display_name | character | Full team display name, e.g. "Ohio State Buckeyes". |
| team_short_display_name | character | Short team display name, e.g. "Ohio State". |
| team_color | character | Primary team color hex (no leading #), e.g. "ba0c2f". |
| team_alternate_color | character | Alternate team color hex, e.g. "a8adb4". |
| team_alternate_ids_sdr | character | Alternate SportsDataR team id, e.g. "7014". |
| logo_href | character | URL to the team logo image. |
| logo_dark_href | character | URL to the dark-mode team logo image. |

_Release tag: `espn_cfb_game_rosters`_

---

### rosters

One row per rostered athlete **per season** — the [game_rosters](#game-rosters)
compilation de-duplicated to one row per `(season, team_id, athlete_id)` (the latest
game's attribute values are kept), with the per-game circumstance fields dropped:
`game_id`, `week`, `starter`, `did_not_play`, `winner`, `order`, `home_away`. All other
athlete/team identity, physical, status, and media columns from `game_rosters` are
retained. Note: this dataset is **ESPN-derived** (it supersedes the prior CFBD-sourced
`espn_cfb_rosters` schema), so it does not carry CFBD-only fields such as hometown/geo,
recruiting ids, or class year.

_Release tag: `espn_cfb_rosters`_

---

### betting

One row per game.

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game/event id, stamped onto every betting row from the enriched game's `id`/`gameId`; the join key back to other release datasets. |
| season | integer | Season year the game belongs to (e.g. 2024, 2026), stamped from the enriched game payload. |
| week | integer | Week number within the season, stamped from the enriched game payload. |
| game_spread | double | Resolved point spread for the game (favorite's line, negative when the favorite is laying points; here -48.5). Single scalar selected by the R reshape from the underlying odds payloads according to `odds_source`. |
| over_under | double | Resolved over/under (total points) line for the game (e.g. 55.5). Selected alongside `game_spread` per `odds_source`. |
| home_favorite | logical | TRUE when the home team is the betting favorite, FALSE when the away team is favored. Derived from the sign/side of the resolved spread. |
| home_team_spread | double | Resolved spread expressed from the home team's perspective (negative when home is favored; matches `game_spread` when `home_favorite` is TRUE). |
| game_spread_available | logical | TRUE when a usable spread/total was resolved for the game, FALSE when no odds were available (in which case the numeric odds fields fall back to defaults / NA). |
| odds_source | character | Provenance of the resolved odds, indicating which upstream payload the scalar fields were taken from: `summary_pickcenter` (ESPN summary pickcenter), `core_odds_api` (ESPN core odds API items), `default` (hardcoded fallback when no odds found), or `injected` (manually supplied/override). |

_Release tag: `espn_cfb_betting`_

---

### schedules

One row per game (one row per compiled game in the season; keyed on `game_id`).

| col_name | col_type | col_description |
| --- | --- | --- |
| game_id | integer | ESPN game/event id, stamped from top-level `id`/`gameId` (e.g. 401628455). Primary key, one per game. |
| season | integer | Season year the game belongs to, stamped from top-level `season` (e.g. 2024). |
| week | integer | Week number within the season, stamped from top-level `week` (e.g. 1). |
| season_type | integer | ESPN season-type code stamped from top-level `season_type` (2 = regular season, 3 = postseason). |
| game_date | character | Kickoff datetime in ISO-8601 UTC from `header.competitions[0].date` (e.g. "2024-08-31T19:30Z"). |
| neutral_site | logical | Whether the game was played at a neutral site, from `header.competitions[0].neutralSite`. |
| conference_competition | logical | Whether the matchup is a conference game, from `header.competitions[0].conferenceCompetition`. |
| home_id | integer | ESPN team id of the home team, from the home competitor in `header.competitions[0].competitors[]` (mirrors top-level `homeTeamId`). |
| home_team | character | Home team display name from the home competitor's `team.displayName` (e.g. "Ohio State Buckeyes"). |
| home_location | character | Home team school/location name from `team.location` (e.g. "Ohio State"). |
| home_nickname | character | Home team mascot/nickname from `team.name` (e.g. "Buckeyes"). Note: `team.nickname` is the school name ("Ohio State"), not the mascot; this column correctly uses `team.name`. |
| home_abbreviation | character | Home team abbreviation from `team.abbreviation` (e.g. "OSU"). |
| home_conference_id | integer | Home team's ESPN conference/group id from `team.groups.id` (e.g. 5). |
| home_score | integer | Final points scored by the home team, from the home competitor's `score`. |
| home_winner | logical | Whether the home team won, from the home competitor's `winner` flag. |
| home_rank | integer | Home team's AP/coaches poll rank entering the game from the home competitor's `rank`; NA when unranked. |
| away_id | integer | ESPN team id of the away team, from the away competitor in `header.competitions[0].competitors[]` (mirrors top-level `awayTeamId`). |
| away_team | character | Away team display name from the away competitor's `team.displayName` (e.g. "Akron Zips"). |
| away_location | character | Away team school/location name from `team.location` (e.g. "Akron"). |
| away_nickname | character | Away team mascot/nickname from `team.name` (e.g. "Zips"). Note: `team.nickname` is the school name ("Akron"), not the mascot; this column correctly uses `team.name`. |
| away_abbreviation | character | Away team abbreviation from `team.abbreviation` (e.g. "AKR"). |
| away_conference_id | integer | Away team's ESPN conference/group id from `team.groups.id` (e.g. 15). |
| away_score | integer | Final points scored by the away team, from the away competitor's `score`. |
| away_winner | logical | Whether the away team won, from the away competitor's `winner` flag. |
| away_rank | integer | Away team's AP/coaches poll rank entering the game from the away competitor's `rank`; NA when unranked. |
| status_type_name | character | Game status enum from `header.competitions[0].status.type.name` (e.g. "STATUS_FINAL"). |
| status_type_completed | logical | Whether the game has completed, from `status.type.completed`. |
| status_type_detail | character | Human-readable status text from `status.type.detail`/`shortDetail` (e.g. "Final"). |
| venue_id | integer | ESPN venue id from `gameInfo.venue.id` (e.g. 3861). |
| venue_full_name | character | Venue name from `gameInfo.venue.fullName` (e.g. "Ohio Stadium"). |
| venue_city | character | Venue city from `gameInfo.venue.address.city` (e.g. "Columbus"). |
| venue_state | character | Venue state from `gameInfo.venue.address.state` (e.g. "OH"). |
| venue_grass | logical | Whether the venue surface is natural grass, from `gameInfo.venue.grass` (e.g. FALSE for Ohio Stadium). |
| attendance | integer | Reported attendance from `gameInfo.attendance` (e.g. 102011). |

_Release tag: `espn_cfb_schedules`_

---

### linescores

One row per (team, period) — per-quarter scoring for each team, reshaped from `g["team_box_extra"][team_id]["linescores"]` long across all games in a season.

NOTE: present only when `team_box_extra` exists (recent seasons); empty otherwise. In this 2024 fixture each linescore element carries only `displayValue` (the per-period points); `value` is parsed from `displayValue` and `period` is derived from the 1-based index within each team's linescore list. The reshape then stamps `team_id`, `game_id`, and `season`.

| col_name | col_type | col_description |
| --- | --- | --- |
| team_id | integer | ESPN team identifier for the team this linescore row belongs to (the `team_box_extra` map key, e.g. 194, 2006). |
| period | integer | 1-based period/quarter index derived from the element position within the team's `linescores` list (1=Q1 ... 4=Q4; >4 for overtime). |
| value | integer | Points scored by the team in that period, parsed from each linescore element's `displayValue` string (e.g. "7" -> 7). |
| game_id | integer | ESPN game/event identifier stamped onto every row (e.g. 401628455); from the JSON top-level `id`. |
| season | integer | Season (calendar year) the game belongs to, stamped onto every row (e.g. 2024). |

_Release tag: `espn_cfb_linescores`_

---

### power_index

One row per team per FPI stat (long format; in this fixture 2 teams x 4 FPI stats = 8 rows per game). Recent-only: empty (zero rows) for pre-~2015 games whose `power_index.items` is unresolved/empty.

| col_name | col_type | col_description |
|:---|:---|:---|
| game_id | character | ESPN event/game id, stamped onto every row (e.g. "401628455"). |
| season | integer | Season year the FPI projection was run for (from the resolved powerindex resource, e.g. 2024). |
| team_id | character | ESPN team id parsed from the per-team powerindex `team.$ref` (e.g. "194", "2006"). |
| team_name | character | Team short name from the ESPN team catalog (team_detail=TRUE only). |
| team_abbreviation | character | Team abbreviation from the ESPN team catalog (team_detail=TRUE only). |
| team_location | character | Team location/school name from the ESPN team catalog (team_detail=TRUE only). |
| team_display_name | character | Full team display name from the ESPN team catalog (team_detail=TRUE only). |
| team_short_display_name | character | Team short display name from the ESPN team catalog (team_detail=TRUE only). |
| team_nickname | character | Team nickname from the ESPN team catalog (team_detail=TRUE only). |
| team_color | character | Primary team color hex from the ESPN team catalog (team_detail=TRUE only). |
| team_alternate_color | character | Alternate team color hex from the ESPN team catalog (team_detail=TRUE only). |
| team_logo_href | character | Team logo URL from the ESPN team catalog (team_detail=TRUE only). |
| team_logo_dark_href | character | Dark-mode team logo URL from the ESPN team catalog (team_detail=TRUE only). |
| stat_name | character | FPI stat machine name: teampredptdiff, gameprojection, matchupquality, or teamadjgamescore. |
| abbreviation | character | ESPN stat abbreviation (e.g. "PRED PT DIFF", "GAME PROJ", "MATCHUP QUALITY", "TEAM ADJ GAMESCORE"). |
| display_name | character | ESPN stat display name (e.g. "PRED PT DIFF", "WIN PROB", "MATCHUP QUALITY", "GAME SCORE"). |
| short_display_name | character | ESPN stat short display name; NA in practice because the powerindex stat objects omit shortDisplayName. |
| value | double | Numeric FPI value for this team/stat: predicted point differential (can be negative), win-probability percentage (0-100), matchup quality (0-100), or adjusted game score. |
| display_value | character | ESPN-formatted display string of the value (e.g. "40.2", "98.7%", "53.8", "65.3"). |
| description | character | ESPN long-form description of the FPI stat (e.g. "Expected margin of victory for the FPI favorite."). |
| powerindex_ref | character | Source `$ref` URL of the per-team powerindex resource that was resolved. |
| team_ref | character | Source `$ref` URL of the team resource (`.../seasons/{season}/teams/{team_id}`) from which team_id was parsed. |

_Release tag: `espn_cfb_power_index`_

---

### injuries

One row per injury entry (one injured athlete per team injury report). Empty (`[]`) in this fixture; schema below is the expected ESPN-injuries shape after the one-row-per-athlete reshape.

| col_name | col_type | col_description |
| --- | --- | --- |
| team_id | character | ESPN team id for the team whose injury report this entry belongs to (from the per-team list element's outer `id` field). Stamped onto each athlete row during the reshape. |
| athlete_name | character | Display name of the injured player (from `athlete.displayName`; may fall back to `athlete.shortName`). |
| status | character | Injury/availability status for the athlete at the entry level (ESPN injury `status` string, e.g. "Active", "Out", "Questionable"). |
| type | character | Injury type category name (from the entry's `type.name`, e.g. "INJURY_STATUS_ACTIVE"; `type.description` gives the human-readable label). |
| date | character | ISO-8601 timestamp of the injury report/update for the athlete (from the entry's `date`), as a character string. |
| position | character | Athlete position abbreviation at time of report, when present in the nested athlete object (`athlete.position.abbreviation`). |
| id | character | Injury record id for the entry (from the entry's top-level `id` field, e.g. "149141"). Identifies the individual injury note. |
| short_comment | character | Short free-text summary of the injury report (from the entry's `shortComment`). The brief version of the narrative. |
| long_comment | character | Full narrative text of the injury report (from the entry's `longComment`). The extended commentary on the athlete's status. |
| game_id | integer | ESPN game/event id stamped onto every row from the enriching game file name (e.g. 401628455). Integer-valued id. |
| season | integer | Season year stamped onto every row during season compilation (e.g. 2024). |
| week | integer | Week number stamped during compilation when available from the game's competition metadata; may be NA for non-regular-season games. |

_Release tag: `espn_cfb_injuries`_

---
