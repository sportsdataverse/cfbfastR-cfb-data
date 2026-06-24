# cfb_data_build

The **Python half of the cfb-data dual-write**: it reproduces the R producers
(`R/_data_utils.R` + `R/espn_cfb_*_creation.R`) that reshape cfb-raw
`final.json` game payloads into the tidy per-season frames released under the
`espn_cfb_*` piggyback tags on `sportsdataverse/sportsdataverse-data`.

It is a **faithful, parity-tested port** — not a reimplementation. Every dataset
is validated byte-for-value against the R-released output captured from the same
committed game fixture (see "Parity" below).

## Design

Two ways to build a dataset, selected per row in
[`config.py`](config.py)'s `REGISTRY`:

| kind | how | datasets |
|---|---|---|
| **generic flatten** (`block` path) | [`reshape.flat_block_frame`](reshape.py) flattens one (possibly nested) `final.json` block to one row per element, then stamps game identity | `play_participants`, `game_rosters`, `injuries`, `power_index`, and the 8 `adv_*` sections (`("advBoxScore", <sec>)`) |
| **bespoke reshaper** (`reshaper` key) | a hand-ported function in [`reshapers.py`](reshapers.py) | `pbp` (conform), `team_box`, `player_box`, `drives`, `betting`, `schedules`, `linescores`, `rosters` (derive) |

Supporting modules: [`io.py`](io.py) (`write_dataset` → parquet + csv.gz +
manifest), [`build.py`](build.py) (`build_season` / `build_dataset`, reusing
`cfb_data_ingest` for fetch/enumerate), [`publish.py`](publish.py) (generalizes
the `cfb_model_publish` gh-release pattern), [`cli.py`](cli.py)
(`python -m cfb_data_build --dataset <name> -s <yr> -e <yr>`).

### Faithful-port specifics

- **`pbp` conform**: [`pbp.py`](pbp.py) is a verbatim port of cfbfastR's
  `.pbp_apply_output_schema` (`cfbfastR/R/pbp_output_schema.R`) — five static
  column manifests + tiered drop + canonical reorder. `conform_pbp` delegates to
  it in R, so the released `espn_cfb_pbp` is the conformed `"default"` tier
  (371 cols here), not the raw 380-col flatten. **No value recomputation** — EPA/WP
  already live in the enriched `final.json`.
- **List cells → JSON strings**: R reads with `simplifyVector = FALSE` then
  `dict_to_row` unboxes length-1 / boxes length-2+, and `stringify_list_cols`
  JSON-encodes survivors. `reshape._norm_cell` reproduces this; the compact
  `json.dumps(separators=(",",":"))` is **byte-identical** to R
  `jsonlite::toJSON(auto_unbox=TRUE)` for these structures (e.g. pbp
  `teamParticipants`).
- **Frame-level identity stamping**: `game_id`/`season`/`week` are appended
  *after* the row union (`stamp_identity`), matching R's `df$game_id <- ...`.
  Critical for heterogeneous-key frames like `player_box`.
- **`rosters`** is a **season-level** dedup of the `game_rosters` *output* (not
  per game): build `game_rosters` first, then `build_rosters_season` reads its
  parquet and derives one row per `(season, team_id, athlete_id)` (latest game).
- **rds is R-owned**: Python writes parquet + csv.gz + manifest; `.rds` (R's
  native format) stays with the R producer. The dual-write parity bar is the
  parquet.

## Parity

Tests in [`../tests/cfb_data_build/`](../tests/cfb_data_build) compare each
Python builder against an R **oracle parquet** produced by running the *actual*
R `reshape_*` on the committed fixture `final_401628455.json` (game 401628455,
2024 wk 1) under R 4.5.3 with `cfbfastR` installed. The comparison asserts:
column names **and order** exact, row count exact, and values type-normalized on
the oracle dtype (numeric/bool → Float64 @ 1e-9; string exact). 39 tests, all green.

**Coverage — all released datasets at parity:**

Per-game (`final.json` reshape): `play_participants`, `pbp`, `team_box`,
`player_box`, `drives`, `game_rosters`, `rosters`, `betting`, `schedules`,
`linescores`, `power_index`, `injuries` (empty-frame degrade), and the 8
`adv_*` sections (`adv_team` … `adv_situational`).

Season aggregation: `team_summaries` ([team_summaries.py](team_summaries.py))
builds the 5 "Binion Box Score" tables (`percentiles`, `team_summaries`,
`passing`, `rushing`, `receiving`) from a full cfbfastR-schema season pbp. The
deterministic aggregations match R exactly; the opponent-adjusted EPA columns
(`glmnet` ridge in R vs `sklearn` here) are held to a correlation bar. Its
parity test is `integration`-marked (consumes a captured season pbp). **Its CI
step stays on R for now** — the Python season-pbp source
(`sportsdataverse.cfb.load_cfb_pbp`) is stale for recent seasons, so it has no
current-season Python input yet.

## Usage

```sh
# one dataset, season range (fetches missing final.json into the cache first)
python -m cfb_data_build --dataset pbp -s 2024 -e 2024
python -m cfb_data_build --dataset play_participants -s 2014 -e 2024 --publish

# rosters depends on game_rosters being built first for the season
python -m cfb_data_build --dataset game_rosters -s 2024 -e 2024
python -m cfb_data_build --dataset rosters       -s 2024 -e 2024
```
