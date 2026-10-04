# defense_vs_position fixtures

Real released 2024 rows for `python/cfb_data_build/defense_vs_position.py` (stage 66),
cut on 2026-10-04 from this repo's committed tree at `origin/main` `bcb96ae7e`. The two
files sdv-py's own fixture was cut from are byte-identical to the
`sportsdataverse-data` release assets (same sha256 as
`sportsdataverse-py/tests/fixtures/defense_vs_position/README.md` records).

| fixture | rows | source | source sha256 |
| --- | --- | --- | --- |
| `play_by_play_2024_slice.parquet` | 1,621 | `cfb/pbp/parquet/play_by_play_2024.parquet` (`espn_cfb_pbp`) | `9b4bfe6a8241cf496f7c43096c305269abc32431fc96f080e4a12302009994e3` |
| `cfb_rosters_2024_slice.parquet` | 203 | `cfb/cfb_rosters/parquet/cfb_rosters_2024.parquet` (`espn_cfb_rosters`) | `5d0f008bc4490ecd4ae0cd9c5224d5da4abc93914bd43321aa9cb854c168ba34` |
| `cfb_schedules_2024_slice.parquet` | 20 | `cfb/cfb_schedules/parquet/cfb_schedules_2024.parquet` (`cfb_schedules`) | `7feb11b7ac2eccd92afd741a59efeb7f1fd26bb51d15675ab1164cd01985a153` |

## Selection

**pbp**: every play (special teams and penalties included) the seven defenses below
defended in the listed games: the semi-join of the season pbp on those
`(def_pos_team_id, game_id)` pairs, sorted by `game_id, game_play_number`. Columns:
`sportsdataverse.defense_vs_position.PBP_COLUMNS["cfb"]` plus `type.text` and `text`
for reading plays by hand.

| defense (`team_id`) | division | games (`game_id`) | why it is here |
| --- | --- | --- | --- |
| Penn State (213) | fbs | 401628457 at West Virginia, 401628470 Bowling Green, 401628493 Kent State | sdv-py's fixture games; the TE cell matches its hand count |
| Georgia (61) | fbs | 401628323 Clemson, 401628339 Tennessee Tech | sdv-py's fixture games; 2 games, so never `qualified` |
| Ohio State (194) | fbs | 401628455 Akron, 401628468 Western Michigan, 401628498 at Michigan State | FBS qualifier |
| Texas (251) | fbs | 401628331 Colorado State, 401628347 at Michigan, 401628361 UTSA | FBS qualifier |
| Notre Dame (87) | fbs | 401628332 at Texas A&M, 401628977 Northern Illinois, 401628978 at Purdue | FBS qualifier |
| Oregon (2483) | fbs | 401628456 Idaho, 401628469 Boise State, 401628483 at Oregon State | FBS qualifier |
| North Dakota State (2449) | fcs | 401634206 at Colorado, 401634207 Tennessee State, 401634135 at East Tennessee State | qualified but not FBS |

The four added FBS defenses and North Dakota State each carry their first three games
by `cfb_schedules.start_date` among the games the season pbp has plays for (the pbp has
no plays for Ohio State vs Marshall, 401628492, so its third game is Michigan State).
Five FBS qualifiers is exactly `MIN_COHORT_TEAMS`, the F5 cohort floor.

**rosters**: `season`, `team_id`, `athlete_id`, `display_name`, `position`,
`position_abbreviation` for every rusher and receiver id in the pbp slice: 204 ids, 203
rows. The missing id, `-5650`, is ESPN's TEAM placeholder.

**schedules**: the 20 games above, with `game_id`, `season`, `week`, `season_type_id`,
`start_date` and the home / away `id`, `team`, `division`, `conference`.

## Hand-computed cells

Counted row by row in plain Python from the fixture files, not through the builder or
the sdv-py function. Population: `EPA_scrimmage` not null, `down` 1-4, `EPA` not null,
`seasonType` 2 or 3.

**Penn State, TE** (sdv-py's cell): 14 targets over 2 games, EPA sum
12.263268947601318, 0.875948 per play. Two games, so not `qualified`.

**RB, EPA per play allowed** (carries by a roster RB or FB, plus targets naming one).
Lower is the better defense, so the lowest ranks first.

| defense | plays | games | EPA sum | EPA/play | carries | rush yards | rank of 5 | `_pct` = 100 * (6 - rank) / 6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ohio State | 63 | 3 | -11.423318937420845 | -0.181323 | 52 | 143 (2.75 a carry) | 1 | 83.33 |
| Texas | 91 | 3 | -10.408038347959518 | -0.114374 | 79 | 328 | 2 | 66.67 |
| Penn State | 75 | 3 | -7.0440216436982155 | -0.093920 | 69 | 255 | 3 | 50.00 |
| Notre Dame | 92 | 3 | 2.9307474493980408 | 0.031856 | 84 | 351 | 4 | 33.33 |
| Oregon | 75 | 3 | 13.624835595488548 | 0.181664 | 63 | 357 | 5 | 16.67 |
| Georgia | 47 | 2 | -6.262705460190773 | -0.133249 | 42 | 139 | none (2 games) | null |
| North Dakota State | 56 | 3 | 6.197718042880297 | 0.110674 | 51 | 267 | none (FCS) | null |

Georgia would rank 2nd of 6 and North Dakota State 5th of 6 if either leaked into the
cohort, which moves Texas, Notre Dame and Oregon off the values above.

**QB, sack rate** (sacks per dropback). The one reversed column: more sacks is the
better defense, so the highest ranks first.

| defense | dropbacks | sacks | sack rate | rank of 5 | `_pct` |
| --- | --- | --- | --- | --- | --- |
| Ohio State | 82 | 12 | 0.146341 | 1 | 83.33 |
| Oregon | 108 | 7 | 0.064815 | 2 | 66.67 |
| Notre Dame | 79 | 5 | 0.063291 | 3 | 50.00 |
| Penn State | 85 | 4 | 0.047059 | 4 | 33.33 |
| Texas | 91 | 4 | 0.043956 | 5 | 16.67 |
| Georgia | 41 | 5 | 0.121951 | none (2 games) | null |
| North Dakota State | 79 | 5 | 0.063291 | none (FCS) | null |

**`unattributed_target_share`** (throws = dropbacks that are not sacks; unattributed =
no `receiver_player_id`).

- Georgia vs Clemson, game 401628323: 29 throws, 3 with no receiver named: play 20
  ("Cade Klubnik pass incomplete"), play 131 (the Malaki Starks interception) and play
  149 ("Cade Klubnik pass incomplete"). 3 / 29 = 0.103448.
- Georgia, both games: 6 of 36. Penn State, three games: 40 of 81.

TE cohort: only 3 of the FBS defenses have a TE row spanning 3 games (Penn State's has
2), under the floor of 5, so every TE `_pct` is null.
