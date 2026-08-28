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
| [![cfb_playoff_figures.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_playoff_figures.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_playoff_figures.yml) | on push / PR / dispatch | never run |
| [![cfb_postweek.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_postweek.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_postweek.yml) | daily 15:00 UTC in Aug-Dec, dow 1 | 2026-08-24 |
| [![cfb_previews.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_previews.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_previews.yml) | daily 14:00 UTC in Sep-Nov, dow 6; daily 22:00 UTC in Sep-Nov, dow 1-5; daily 14:00 UTC in Dec; days 1-20 14:00 UTC in Jan | never run |
| [![cfb_ratings_cron.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_ratings_cron.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_ratings_cron.yml) | daily 13:00 UTC in Aug; daily 13:00 UTC in Sep-Nov; daily 13:00 UTC in Dec; days 1-25 13:00 UTC in Jan | 2026-08-27 |
| [![cfb_recruiting_proj_cron.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_recruiting_proj_cron.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/cfb_recruiting_proj_cron.yml) | day 5 12:00 UTC in Dec; day 5 12:00 UTC in Jan-Aug | 2026-08-05 |
| [![daily_cfb.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/daily_cfb.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/daily_cfb.yml) | daily 11:00 UTC in Aug; daily 11:00 UTC in Sep-Nov; daily 11:00 UTC in Dec; days 1-25 11:00 UTC in Jan | 2026-08-27 |
| [![orphan_scripts.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/orphan_scripts.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/orphan_scripts.yml) | on push / PR / dispatch | 2026-08-28 |
| [![tests.yml](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/tests.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-data/actions/workflows/tests.yml) | on push / PR / dispatch | 2026-08-28 |

<!-- END GENERATED: status -->
