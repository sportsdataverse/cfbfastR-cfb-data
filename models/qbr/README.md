# models/qbr — xQBR labels, pre-registration and gate record

| file | what it is |
|---|---|
| `espn_qbr_labels.parquet` | ESPN game-level QBR, the training target. One row per (game_id, athlete_id) |
| `PREREG_xqbr_retrain.md` | the 2026-09-30 pre-registration: data, frozen holdout, arms, gate, choice rule |
| `qbr.gate.json` | the gate record `train-qbr` wrote for the shipped model: every arm's holdout numbers, the input checksums, and the sha256 of the model that passed |
| `qbr_partition.parquet` | that fit's train/holdout partition (game_id, athlete_id, season, week, split), so its holdout can be reproduced exactly |

## `espn_qbr_labels.parquet` provenance

- **Source:** ESPN core API
  `sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/{season}/types/{2|3}/weeks/{week}/qbr/10000?limit=1000`
  via sdv-py `espn_cfb_season_qbr_week(..., split=10000)`. It covers regular-season weeks
  1–17 (week 17 is always empty) and postseason week 1, which holds every bowl and CFP game.
- **Captured:** 2026-09-30 (`captured_at`). The capture was sequential with a 0.5 s pause
  between requests, 414 requests in all.
- **Command:** from `python/`: `python -m cfb_model_build.model_training capture-qbr-labels --seasons 2004 ... 2026`.
  It replaces just the named seasons in this file.
- **Size:** 34,430 rows, 18,300 games, 26 columns. sha256 at capture:
  `dff528db00bec86d7589f6f91614ccef061775814d53e88828fdeb03d6d30570`.
- **Rows per season:** 2004 1234 · 2005 1291 · 2006 1426 · 2007 1494 · 2008 1485 ·
  2009 1494 · 2010 1489 · 2011 1505 · 2012 1556 · 2013 1603 · 2014 1610 · 2015 1622 ·
  2016 1617 · 2017 1651 · 2018 1641 · 2019 1633 · 2020 1100 (the shortened season) ·
  2021 1622 · 2022 1658 · 2023 1698 · 2024 1719 · 2025 1732 · 2026 550.
  The 2026 rows are weeks 1–4 only, the frozen holdout. Week 5 had no ESPN values yet at
  capture.
- **Ids:** `game_id`, `athlete_id` and `team_id` are Int64, parsed from the item's `$ref`
  links as integers and never through a float. There were 0 null ids and 0 duplicate
  (game_id, athlete_id) at capture; `capture()` raises on a duplicate.
- **Columns:** ESPN's stat abbreviations as ESPN ships them. `QBR` is raw QBR (the target),
  `TQBR` is Total QBR (opponent-adjusted, reported only), and `QBP` is the QB's action-play
  count. The rest are ESPN's components (`PA`, `PAS`, `RUN`, `SAC`, `PEN`, `EPE`, ...).

## Re-capture policy

ESPN revises and drops values, which is why these labels are committed rather than
scraped inside every pipeline run. Refreshing them is a deliberate, reviewed commit: the
parquet diff shows what ESPN moved. Add a new season's labels after its bowls finish,
before the annual retrain. Do not re-capture the frozen holdout (2026 weeks 1–4) without a
new pre-registration.
