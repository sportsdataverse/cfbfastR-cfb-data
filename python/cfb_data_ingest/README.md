# cfb_data_ingest

URL-fetch ingest layer for per-game CFB JSON. Fetches `final.json` files from
`raw.githubusercontent.com` — **no API key required**.

## Quick start

```bash
python -m cfb_data_ingest --seasons 2023 2024 --cache-dir .cache/cfb_final
```

This warms a local cache of per-game JSON files from:

```
https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/json/final/{game_id}.json
```

The schedule master (also at `raw.githubusercontent.com`) is fetched automatically
to enumerate `game_id` values for the requested seasons. No authentication is
needed for either request.

### Options

| Flag | Default | Description |
|---|---|---|
| `--seasons` | all | Space-separated list of seasons to fetch |
| `--cache-dir` | `.cache/cfb_final` | Directory to write `{game_id}.json` files into |
| `--schedule` | (RAW_BASE URL) | Local path override for the schedule master parquet |
| `--refresh` | false | Re-fetch even if a cached file already exists |

## Disk cache layout

```
<cache-dir>/
  <game_id>.json   # per-game final.json payload
```

`fetch_final()` is fail-soft per game — a missing or non-200 response is
counted in the `missing` total and does not abort the batch.

## `CFB_DATA_API_KEY` — not used by cfb_data_ingest

`CFB_DATA_API_KEY` is **not** read by this module. The key is only required by
`pregame_wp`'s separate CFBD ingest (`pregame_wp.data_ingest`), which hits
`api.collegefootballdata.com` directly for plays and drives data. Obtain a free
token at <https://collegefootballdata.com/key> if you need that surface.
