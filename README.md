# cfbfastR-cfb-data

Analysis-ready college-football datasets, **reshaped** from the per-game enriched `final`
JSON in [`cfbfastR-cfb-raw`](https://github.com/sportsdataverse/cfbfastR-cfb-raw). The
heavy lifting (EPA/WPA/QBR, advanced box score) already happened in Python in the `-raw`
repo — this repo does **no re-enrichment**, only rectangularization into parquet/csv/rds.

## What it produces

Per compiled season, one table per dataset (see **[DATASETS.md](DATASETS.md)** for the full
`col_name | col_type | col_description` data dictionary — 19 datasets, ~928 columns):

| dataset | tag | grain |
|---|---|---|
| play_by_play | `espn_cfb_pbp` | one row per play (~380 enriched cols) |
| team_box / player_box | `espn_cfb_team_box` / `espn_cfb_player_box` | ESPN box scores |
| adv_team / adv_passing / adv_rushing / adv_receiving / adv_defensive / adv_turnover / adv_drives / adv_situational | `espn_cfb_adv_*` | advanced box (EPA/success/explosiveness) |
| play_participants | `espn_cfb_play_participants` | per-play participants |
| drives | `espn_cfb_drives` | drive-level |
| game_rosters | `espn_cfb_game_rosters` | per-game rosters (one row per athlete per game) |
| rosters | `espn_cfb_rosters` | season rosters (ESPN-derived, deduped from game_rosters) |
| betting | `espn_cfb_betting` | resolved odds/lines |
| schedules | `espn_cfb_schedules` | game meta |
| linescores | `espn_cfb_linescores` | per-quarter (recent) |
| power_index | `espn_cfb_power_index` | FPI (recent) |
| injuries | `espn_cfb_injuries` | injury reports |

Each dataset is committed in-repo under `cfb/{dataset}/{parquet,rds,csv}/` **and** published
to `sportsdataverse/sportsdataverse-data` releases under its `espn_cfb_*` tag.

> **`load_cfb_pbp()` cutover:** `cfbfastR::load_cfb_pbp()` currently reads the *legacy*
> `cfbfastR_cfb_pbp` release. This pipeline publishes to `espn_cfb_pbp` for now (legacy data
> untouched); a later cutover repoints the loader or promotes the assets.

## Usage

```sh
Rscript R/espn_cfb_01_pbp_creation.R -s 2024 -e 2024     # one dataset, one season
bash scripts/daily_cfb_R_processor.sh -s 2024 -e 2024    # all datasets, season range
Rscript R/releases_init.R                                # one-time: create release tags
```

### Recruiting datasets (backfill / manual)

`scripts/10_build_recruiting.sh` builds `cfb_recruits`, `cfb_team_talent` and
`cfb_returning_production`. The daily processor already builds the current
season; this script is the **backfill** entry point, and it carries each
dataset's measured floor so a below-floor request is clamped rather than
silently producing a thinner number:

| dataset | floor | why |
|---|---|---|
| `cfb_recruits` | 2002 | 247 composite ratings collapse before then (2001: 52% rated on page 1, 0% by page 4) |
| `cfb_team_talent` | 2005 | 2002 + the 4-class window |
| `cfb_returning_production` | 2005 | needs the season S-1 ESPN player box, which floors at 2004 |

```sh
bash scripts/10_build_recruiting.sh                                   # all three, floor..current
bash scripts/10_build_recruiting.sh --publish                         # + upload to the release tags
bash scripts/10_build_recruiting.sh --dataset team_talent --start 2020 --end 2024
```

`recruits` and `team_talent` read the raw 247 store from `cfbfastR-cfb-raw`
(`CFB_RAW_ROOT`, default `../cfbfastR-cfb-raw`); `returning_production` reads
the ESPN player box and needs no store.

## Automation

Triggered by a `repository_dispatch` from `cfbfastR-cfb-raw` on every push (the commit
message carries `Start:/End:` years), plus a cron over the CFB calendar (offset after
`-raw`) and manual `workflow_dispatch`. See `.github/workflows/daily_cfb.yml`.

## Architecture

`-raw` (Python/uv): scrape ESPN → enrich → `cfb/json/final/{id}.json`.
`-data` (this repo, R): read `final` JSON over HTTP → reshape each block → parquet/csv/rds →
piggyback release. Reshape functions are pure and unit-tested offline against a committed
fixture (`tests/testthat/`).

## Automation & status

<!-- BEGIN GENERATED: status -->

| workflow | schedule | last run |
|---|---|---|
| [![cfb_model_pipeline.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_model_pipeline.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_model_pipeline.yml) | day 5 06:00 UTC in Feb | 2026-07-30 |
| [![cfb_playoff_figures.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_playoff_figures.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_playoff_figures.yml) | on dispatch | never run |
| [![cfb_postweek.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_postweek.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_postweek.yml) | Mondays 15:00 UTC in Aug-Dec | 2026-08-24 |
| [![cfb_previews.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_previews.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_previews.yml) | Saturdays 14:00 UTC in Sep-Nov; weekdays 22:00 UTC in Sep-Nov; daily 14:00 UTC in Dec; days 1-20 14:00 UTC in Jan | never run |
| [![cfb_ratings_cron.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_ratings_cron.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_ratings_cron.yml) | daily 13:00 UTC in Aug; daily 13:00 UTC in Sep-Nov; daily 13:00 UTC in Dec; days 1-25 13:00 UTC in Jan | 2026-08-27 |
| [![cfb_recruiting_proj_cron.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_recruiting_proj_cron.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_recruiting_proj_cron.yml) | day 5 12:00 UTC in Dec; day 5 12:00 UTC in Jan-Aug | 2026-08-05 |
| [![daily_cfb.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/daily_cfb.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/daily_cfb.yml) | daily 11:00 UTC in Aug; daily 11:00 UTC in Sep-Nov; daily 11:00 UTC in Dec; days 1-25 11:00 UTC in Jan | 2026-08-27 |
| [![orphan_scripts.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/orphan_scripts.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/orphan_scripts.yml) | on push / PR / dispatch | 2026-08-28 |
| [![tests.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/tests.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/tests.yml) | on PR / push / dispatch | 2026-08-28 |

| release tag | assets | size | last publish |
|---|---:|---:|---|
| [`cfb_crosswalk`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_crosswalk) | 26 | 1.8 MB | 2026-06-13 |
| [`cfb_fpi_weekly`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_fpi_weekly) | 63 | 18.4 MB | 2026-08-01 |
| [`cfb_model_artifacts`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_model_artifacts) | 19 | 27.6 MB | 2026-08-27 |
| [`cfb_ratings`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_ratings) | 67 | 1.0 MB | 2026-08-27 |
| [`cfb_ratings_weekly`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_ratings_weekly) | 66 | 12.7 MB | 2026-08-06 |
| [`cfb_recruiting_proj`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_recruiting_proj) | 11 | 0.1 MB | 2026-08-06 |
| [`cfb_recruits`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_recruits) | 25 | 1.5 MB | 2026-08-19 |
| [`cfb_returning_production`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_returning_production) | 21 | 0.1 MB | 2026-08-06 |
| [`cfb_schedules`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules) | 78 | 4.3 MB | 2026-08-27 |
| [`cfb_team_info`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_team_info) | 52 | 5.0 MB | 2026-08-27 |
| [`cfb_team_summaries_weekly`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_team_summaries_weekly) | 66 | 231.0 MB | 2026-08-06 |
| [`cfb_team_talent`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_team_talent) | 22 | 0.2 MB | 2026-08-19 |
| [`espn_cfb_adv_defensive`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_defensive) | 66 | 6.6 MB | 2026-08-03 |
| [`espn_cfb_adv_defensive_players`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_defensive_players) | 66 | 13.3 MB | 2026-08-03 |
| [`espn_cfb_adv_drives`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_drives) | 66 | 6.0 MB | 2026-08-03 |
| [`espn_cfb_adv_passing`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_passing) | 66 | 23.3 MB | 2026-08-03 |
| [`espn_cfb_adv_receiving`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_receiving) | 66 | 46.2 MB | 2026-08-03 |
| [`espn_cfb_adv_rushing`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_rushing) | 66 | 25.3 MB | 2026-08-03 |
| [`espn_cfb_adv_situational`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_situational) | 66 | 27.4 MB | 2026-08-03 |
| [`espn_cfb_adv_specialists`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_specialists) | 66 | 14.9 MB | 2026-08-03 |
| [`espn_cfb_adv_team`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_team) | 66 | 29.0 MB | 2026-08-03 |
| [`espn_cfb_adv_team_gamelog`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_team_gamelog) | 66 | 31.3 MB | 2026-08-01 |
| [`espn_cfb_adv_turnover`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_turnover) | 66 | 5.0 MB | 2026-08-03 |
| [`espn_cfb_betting`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_betting) | 48 | 0.3 MB | 2026-08-27 |
| [`espn_cfb_drives`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_drives) | 45 | 23.0 MB | 2026-07-18 |
| [`espn_cfb_game_rosters`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_game_rosters) | 45 | 406.8 MB | 2026-07-18 |
| [`espn_cfb_injuries`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_injuries) | 0 | 0.0 MB | — |
| [`espn_cfb_linescores`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_linescores) | 44 | 0.8 MB | 2026-07-18 |
| [`espn_cfb_model_artifacts`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_model_artifacts) | 22 | 27.5 MB | 2026-06-23 |
| [`espn_cfb_model_pbp`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_model_pbp) | 47 | 1,163.3 MB | 2026-07-12 |
| [`espn_cfb_passing`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_passing) | 66 | 5.2 MB | 2026-08-06 |
| [`espn_cfb_pbp`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_pbp) | 70 | 12,088.7 MB | 2026-08-03 |
| [`espn_cfb_percentiles`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_percentiles) | 66 | 1.8 MB | 2026-08-06 |
| [`espn_cfb_play_participants`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_play_participants) | 25 | 85.6 MB | 2026-07-18 |
| [`espn_cfb_player_box`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_player_box) | 45 | 27.3 MB | 2026-07-18 |
| [`espn_cfb_player_boxscores`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_player_boxscores) | 0 | 0.0 MB | — |
| [`espn_cfb_power_index`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_power_index) | 47 | 1.8 MB | 2026-08-27 |
| [`espn_cfb_receiving`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_receiving) | 66 | 13.6 MB | 2026-08-06 |
| [`espn_cfb_rosters`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_rosters) | 70 | 115.0 MB | 2026-08-27 |
| [`espn_cfb_rushing`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_rushing) | 66 | 9.0 MB | 2026-08-06 |
| [`espn_cfb_schedules`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_schedules) | 48 | 1.4 MB | 2026-08-27 |
| [`espn_cfb_team_box`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_team_box) | 48 | 2.3 MB | 2026-08-27 |
| [`espn_cfb_team_boxscores`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_team_boxscores) | 0 | 0.0 MB | — |
| [`espn_cfb_team_summaries`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_team_summaries) | 66 | 21.8 MB | 2026-08-06 |
| [`espn_cfb_teams`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_teams) | 78 | 14.3 MB | 2026-08-27 |

<!-- END GENERATED: status -->

## Reports & explainers

<!-- BEGIN GENERATED: reports -->

| Report | What it is | Last updated |
|---|---|---|
| [Model registry](models/REGISTRY.md) | model | artifact | gates | retrain, one row per published model | 2026-08-28 |
| [Model reports & cards](docs/models/) | 17 files, one per item | 2026-09-01 |

<!-- END GENERATED: reports -->
