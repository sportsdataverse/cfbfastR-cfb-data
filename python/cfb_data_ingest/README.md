# cfb_data_ingest

Managed ingest layer for College Football Data API (CFBD) payloads used by the
`pregame_wp` modeling pipeline.

## Required secret: `CFB_DATA_API_KEY`

All functions that hit `api.collegefootballdata.com` require a CFBD API bearer
token. Obtain one at <https://collegefootballdata.com/key> (free registration).

Set the variable before running any live-fetch code:

```bash
# shell
export CFB_DATA_API_KEY=<your_token>
```

```ini
# .env file at the project root (loaded automatically by uv run)
CFB_DATA_API_KEY=<your_token>
```

If the key is absent, every CFBD call raises immediately with:

```
EnvironmentError: CFB_DATA_API_KEY not set.
Add it to your .env file or export it before running.
```

The fallback name `CFBD_DATA_API_KEY` is also accepted for backwards
compatibility.

## Test gating

All tests in `tests/pregame_wp/test_data_ingest.py` are marked
`@pytest.mark.integration`. The default test run (`pytest -m "not integration"`)
skips them entirely so CI never needs a live key. To run the live CFBD tests
locally:

```bash
CFB_DATA_API_KEY=<token> uv run pytest tests/pregame_wp/test_data_ingest.py -m integration -v
```

## Disk cache

`pregame_wp.data_ingest.fetch_and_cache(game_id, year, week, raw_dir)` fetches
plays + drives for one game and writes them to:

```
<raw_dir>/<game_id>/plays.json
<raw_dir>/<game_id>/drives.json
```

`load_game_frames(game_id, raw_dir)` reads those cached files back without
touching the network, enabling offline rebuilds of the `pregame_wp` pipeline.
