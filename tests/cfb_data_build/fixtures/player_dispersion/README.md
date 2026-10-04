# player_dispersion fixture

`rushers_2024_auburn_wk02_06.csv`: 39 real carries, cut from the committed 2024 pbp, for
`test_player_dispersion.py`.

- **Source:** `cfb/pbp/parquet/play_by_play_2024.parquet` and
  `cfb/cfb_schedules/parquet/cfb_schedules_2024.parquet` on `main` at 34ee6a12a, passed
  through `summaries_input.prepare_plays_input` and `team_summaries.add_derived_metrics`,
  then the rushing table's own filter (`team_off`, `rush == 1`, `rush_player_id` not null).
- **Cut:** Auburn (`pos_team_id == "2"`), its first five games of 2024 (weeks 2-6:
  401628337, 401628352, 401628363, 401628375, 401628381), three rushers:
  - Damari Alston (4685699): 26 carries over 5 games;
  - Jeremiah Cobb (4870642): 10 carries over 4 games;
  - Hank Brown (5089314): 3 carries over 2 games, under the 3-game dispersion floor.
- **Columns:** the ones the helpers read (`game_id`, `pos_team_id`, `rush_player_id`,
  `EPA`, `yds_rushed`, `pos_score_diff_start`) plus `week`, `game_play_number` and
  `rusher_player_name` for reading. Values are written unrounded.

The boundary carries the tests depend on are real: Alston has a 4-yard and an 11-yard
carry, which sit on the two tier cut-points, and both players have 5-yard carries.

Regenerate (from the repo root, with the 2024 inputs above):

```python
import polars as pl
from cfb_data_build.summaries_input import prepare_plays_input
from cfb_data_build.team_summaries import add_derived_metrics

pbp = pl.read_parquet("cfb/pbp/parquet/play_by_play_2024.parquet")
sched = pl.read_parquet("cfb/cfb_schedules/parquet/cfb_schedules_2024.parquet")
plays = add_derived_metrics(prepare_plays_input(pbp, sched, 2024))
games = ["401628337", "401628352", "401628363", "401628375", "401628381"]
(
    plays.filter(
        pl.col("EPA").is_not_null()
        & pl.col("success").is_not_null()
        & pl.col("epa_success").is_not_null()
        & (pl.col("rush") == 1)
        & (pl.col("pos_team_id") == "2")
        & pl.col("game_id").is_in(games)
        & pl.col("rush_player_id").is_in([4685699, 4870642, 5089314])
    )
    .sort("game_id", "game_play_number")
    .select(
        "game_id", "week", "game_play_number", "pos_team_id", "rush_player_id",
        "rusher_player_name", "EPA", "yds_rushed", "pos_score_diff_start",
    )
    .write_csv("tests/cfb_data_build/fixtures/player_dispersion/rushers_2024_auburn_wk02_06.csv")
)
```
