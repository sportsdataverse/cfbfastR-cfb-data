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
| rosters | `espn_cfb_rosters` | game rosters |
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

## Automation

Triggered by a `repository_dispatch` from `cfbfastR-cfb-raw` on every push (the commit
message carries `Start:/End:` years), plus a cron over the CFB calendar (offset after
`-raw`) and manual `workflow_dispatch`. See `.github/workflows/daily_cfb.yml`.

## Architecture

`-raw` (Python/uv): scrape ESPN → enrich → `cfb/json/final/{id}.json`.
`-data` (this repo, R): read `final` JSON over HTTP → reshape each block → parquet/csv/rds →
piggyback release. Reshape functions are pure and unit-tested offline against a committed
fixture (`tests/testthat/`).
