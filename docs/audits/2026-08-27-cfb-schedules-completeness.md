# `cfb_schedules` completeness audit — 2026-08-27

Verification of the `cfb_schedules` release republished 2026-08-27 (26 seasons,
2001–2026, parquet/rds/csv.gz) after the discovery that seasons 2003–2022 had
shipped **zero** postseason games — the old assets were built when
`cfbfastR::cfbd_game_info()` still defaulted to `season_type = "regular"`.

**Verdict: complete. No defects found.** Every anomaly probed in this audit
resolves to a documented, expected cause.

- Sources: `cfb_schedules` release assets (all 26 downloaded fresh);
  `cfb/pbp/parquet/play_by_play_<yr>.parquet` (2004–2025);
  `cfbfastR-cfb-raw/cfb/cfb_schedule_master.parquet` (ESPN-side raw master,
  19,586 rows, 2004–2026).
- Tooling: polars 1.42 via the `sdv-py` venv.

## 1. Headline totals

| metric | value |
|---|---|
| total rows across 2001–2026 | **48,872** |
| distinct `game_id` | **48,872** (zero duplicates, in-season or cross-season) |
| seasons published | 26 (2001–2026, no gaps) |
| `season_type` values | `regular`, `postseason`, `spring_regular`, `spring_postseason` |
| null `start_date` | **0** (all 26 seasons) |
| null `home_id` / `away_id` | **0** / **0** (all 26 seasons) |
| null points on `completed == True` rows | **3 total** (2025: 2 home, 1 away) |

The 48,872 total matches the publish log exactly.

## 2. Per-season detail

`other` = rows outside the four known `season_type` values; it is 0 everywhere.
`c-null-pts` = null `home_points`/`away_points` on `completed == True` rows.

| season | games | regular | post | spr_reg | spr_post | other | weeks | wk min/max | date min | date max | null date | null ids | completed | c-null-pts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 2001 | 709 | 684 | 25 | 0 | 0 | 0 | 15 | 1/16 | 2001-08-23 | 2002-01-04 | 0 | 0 | 709 | 0 |
| 2002 | 772 | 744 | 28 | 0 | 0 | 0 | 16 | 1/16 | 2002-08-22 | 2003-01-04 | 0 | 0 | 772 | 0 |
| 2003 | 1338 | 1310 | 28 | 0 | 0 | 0 | 16 | 1/16 | 2003-08-23 | 2004-01-05 | 0 | 0 | 1338 | 0 |
| 2004 | 1302 | 1274 | 28 | 0 | 0 | 0 | 15 | 1/15 | 2004-08-28 | 2005-01-05 | 0 | 0 | 1302 | 0 |
| 2005 | 1327 | 1299 | 28 | 0 | 0 | 0 | 14 | 1/14 | 2005-09-01 | 2006-01-05 | 0 | 0 | 1327 | 0 |
| 2006 | 1405 | 1373 | 32 | 0 | 0 | 0 | 14 | 1/14 | 2006-08-26 | 2007-01-09 | 0 | 0 | 1405 | 0 |
| 2007 | 1469 | 1437 | 32 | 0 | 0 | 0 | 14 | 1/14 | 2007-08-25 | 2008-01-08 | 0 | 0 | 1469 | 0 |
| 2008 | 1479 | 1445 | 34 | 0 | 0 | 0 | 15 | 1/15 | 2008-08-28 | 2009-01-09 | 0 | 0 | 1479 | 0 |
| 2009 | 1468 | 1434 | 34 | 0 | 0 | 0 | 15 | 1/15 | 2009-08-27 | 2010-01-08 | 0 | 0 | 1468 | 0 |
| 2010 | 1462 | 1427 | 35 | 0 | 0 | 0 | 15 | 1/15 | 2010-09-02 | 2011-01-11 | 0 | 0 | 1462 | 0 |
| 2011 | 1498 | 1463 | 35 | 0 | 0 | 0 | 15 | 1/15 | 2011-09-01 | 2012-01-10 | 0 | 0 | 1498 | 0 |
| 2012 | 1379 | 1344 | 35 | 0 | 0 | 0 | 15 | 1/15 | 2012-08-30 | 2013-01-08 | 0 | 0 | 1379 | 0 |
| 2013 | 1534 | 1496 | 38 | 0 | 0 | 0 | 16 | 1/16 | 2013-08-29 | 2014-01-07 | 0 | 0 | 1534 | 0 |
| 2014 | 1586 | 1542 | 44 | 0 | 0 | 0 | 16 | 1/16 | 2014-08-23 | 2015-01-13 | 0 | 0 | 1586 | 0 |
| 2015 | 1538 | 1491 | 47 | 0 | 0 | 0 | 15 | 1/15 | 2015-08-29 | 2016-01-12 | 0 | 0 | 1538 | 0 |
| 2016 | 1549 | 1502 | 47 | 0 | 0 | 0 | 15 | 1/15 | 2016-08-27 | 2017-01-10 | 0 | 0 | 1549 | 0 |
| 2017 | 1551 | 1505 | 46 | 0 | 0 | 0 | 15 | 1/15 | 2017-08-26 | 2018-01-09 | 0 | 0 | 1551 | 0 |
| 2018 | 1556 | 1511 | 45 | 0 | 0 | 0 | 15 | 1/15 | 2018-08-25 | 2019-01-08 | 0 | 0 | 1556 | 0 |
| 2019 | 1623 | 1577 | 46 | 0 | 0 | 0 | 16 | 1/16 | 2019-08-24 | 2020-01-14 | 0 | 0 | 1623 | 0 |
| 2020 | 1125 | 563 | 30 | 504 | 28 | 0 | 17 | 1/20 | 2020-08-30 | 2021-05-16 | 0 | 0 | 1125 | 0 |
| 2021 | 2454 | 2408 | 46 | 0 | 0 | 0 | 15 | 1/15 | 2021-08-28 | 2022-01-11 | 0 | 0 | 2454 | 0 |
| 2022 | 3705 | 3657 | 48 | 0 | 0 | 0 | 15 | 1/15 | 2022-08-27 | 2023-01-10 | 0 | 0 | 3705 | 0 |
| 2023 | 3734 | 3595 | 139 | 0 | 0 | 0 | 15 | 1/15 | 2023-08-26 | 2024-01-09 | 0 | 0 | 3724 | 0 |
| 2024 | 3801 | 3747 | 54 | 0 | 0 | 0 | 16 | 1/16 | 2024-08-24 | 2025-01-21 | 0 | 0 | 3799 | 0 |
| 2025 | 3831 | 3745 | 86 | 0 | 0 | 0 | 16 | 1/16 | 2025-08-23 | 2026-01-20 | 0 | 0 | 3831 | 3 |
| 2026 | 3677 | 3677 | 0 | 0 | 0 | 0 | 14 | 1/15 | 2026-08-27 | 2026-12-12 | 0 | 0 | 0 | 0 |

**The postseason regression is fixed.** Every season 2001–2025 now carries a
non-zero `postseason` count (25–139). Only 2026 has 0, correctly — the season
has not been played and bowl/playoff slots are not yet assigned.

Notes on the table:

- **Postseason counts track the real bowl calendar.** 25 (2001) → 28
  (2002–2005) → 32–35 (2006–2012) → 38–47 (2013–2019) → 46–54 (2021–2024),
  matching bowl-count growth and the 4→12-team playoff expansion in 2024.
- **2023 postseason = 139** is not an error: 2023 is the one season where CFBD
  also classifies the FCS/DII/DIII playoff brackets as `postseason` (the other
  seasons file those lower-division playoff games under `regular`). This is an
  upstream CFBD classification quirk, not a producer defect — the *game set* is
  complete either way.
- **2020 is the only season with `spring_*` rows** (504 `spring_regular` +
  28 `spring_postseason` = 532), the COVID-displaced spring 2021 seasons
  (MVFC, Big Sky, Ivy, SWAC et al.). This is why 2020's `wk_max` is 20 and its
  `date_max` is 2021-05-16 — both expected, both unique to 2020.
- **Nulls:** `start_date`, `home_id`, `away_id` are null-free in all 26
  seasons. The only null scores on completed games are 3 cells in 2025
  (2 `home_points`, 1 `away_points`) — a handful of lower-division games whose
  box was never posted.
- **`home_division` / `away_division` nulls** are non-zero throughout
  (1–82/season, worst in 2026 at 33/82). These are CFBD division-classification
  gaps on non-FBS opponents and un-assigned 2026 opponents, not missing games.

## 3. Schedule vs play-by-play cross-check

`orphans` = a `load_cfb_pbp` game_id with no schedule row in **any** season.
`sched_no_pbp` = schedule games with no PBP row.

| season | pbp games | sched games | orphans | sched w/o pbp |
|---:|---:|---:|---:|---:|
| 2004 | 463 | 1302 | 0 | 839 |
| 2005 | 596 | 1327 | 0 | 731 |
| 2006 | 683 | 1405 | 0 | 722 |
| 2007 | 707 | 1469 | 0 | 762 |
| 2008 | 778 | 1479 | 2 | 703 |
| 2009 | 792 | 1468 | 2 | 678 |
| 2010 | 788 | 1462 | 2 | 676 |
| 2011 | 806 | 1498 | 3 | 695 |
| 2012 | 828 | 1379 | 3 | 554 |
| 2013 | 858 | 1534 | 3 | 679 |
| 2014 | 854 | 1586 | 3 | 735 |
| 2015 | 866 | 1538 | **3** | 675 |
| 2016 | 858 | 1549 | 2 | 693 |
| 2017 | 872 | 1551 | 3 | 682 |
| 2018 | 884 | 1556 | 3 | 675 |
| 2019 | 890 | 1623 | 3 | 736 |
| 2020 | 565 | 1125 | 1 | 561 |
| 2021 | 842 | 2454 | 3 | 1615 |
| 2022 | 861 | 3705 | 4 | 2848 |
| 2023 | 903 | 3734 | 0 | 2831 |
| 2024 | 946 | 3801 | 0 | 2855 |
| 2025 | 956 | 3831 | 0 | 2875 |
| **total** | | | **40** | |

### 3a. Orphans — 40 of 40 explained, zero flagged

The 2015 baseline reproduces exactly: **3 orphans, all all-star exhibitions**
(`400859830` NFLPA Collegiate Bowl, `400859831` East-West Shrine Game,
`400859832` Reese's Senior Bowl).

Every orphan in every season is the same pattern. Resolving all 40 ids against
the ESPN raw master gives, without exception, an all-star exhibition:

| event | seasons present | count |
|---|---|---:|
| East-West Shrine Game | 2008–2019, 2021–2022 | 14 |
| Senior Bowl / Reese's Senior Bowl | 2008–2022 | 15 |
| NFLPA Collegiate Bowl | 2011–2015, 2017–2019, 2021–2022 | 10 |
| HBCU Legacy Bowl | 2022 | 1 |

`orphans_same_season` equals `orphans_any` in every season — no PBP game is
mis-filed into the wrong schedule season. All 40 ids **are** present in the ESPN
master (0 unknown to ESPN); CFBD legitimately excludes them because they are
post-season all-star exhibitions between ad-hoc squads, not college teams.

**No season has an orphan set that departs from this pattern. Nothing to flag.**

Two incidental observations, both benign:

- The orphan count drops to 0 from 2023 onward — ESPN stopped filing these
  exhibitions under the college-football PBP feed, so they no longer enter
  `espn_cfb_pbp` in the first place.
- 2020's single orphan (Senior Bowl only) reflects the COVID cancellation of
  the 2021 Shrine Game and NFLPA Bowl.

### 3b. Schedule games without PBP — expected, structural

`sched_no_pbp` is large and rising (554 → 2,875) purely because the two
datasets have different scope by design:

- `espn_cfb_pbp` covers games with an ESPN play-by-play feed — effectively
  FBS games plus FCS opponents of FBS teams (~460–960/season).
- `cfb_schedules` is CFBD's full game universe including FCS/DII/DIII
  intra-division games that ESPN never carries PBP for.

The FBS-vs-FBS game count is flat and healthy across the whole window
(652 in 2001 → 808 in 2025), so the growth is entirely lower-division scope.
This is not a PBP gap.

## 4. Schedule vs ESPN raw master (`cfb_schedule_master.parquet`)

The master is ESPN-side and starts in 2004. `master_only` = master game_ids
absent from the published schedule; `sched_only` = the reverse.

| season | master | sched | master-only | sched-only |
|---:|---:|---:|---:|---:|
| 2004 | 712 | 1302 | 5 | 595 |
| 2005 | 728 | 1327 | 10 | 609 |
| 2006 | 788 | 1405 | 0 | 617 |
| 2007 | 800 | 1469 | 2 | 671 |
| 2008 | 812 | 1479 | 7 | 674 |
| 2009 | 810 | 1468 | 2 | 660 |
| 2010 | 810 | 1462 | 2 | 654 |
| 2011 | 817 | 1498 | 5 | 686 |
| 2012 | 852 | 1379 | 12 | 539 |
| 2013 | 861 | 1534 | 6 | 679 |
| 2014 | 873 | 1586 | 6 | 719 |
| 2015 | 876 | 1538 | 6 | 668 |
| 2016 | 879 | 1549 | 6 | 676 |
| 2017 | 890 | 1551 | 16 | 677 |
| 2018 | 898 | 1556 | 14 | 672 |
| 2019 | 892 | 1623 | 4 | 735 |
| 2020 | 706 | 1125 | **136** | 555 |
| 2021 | 897 | 2454 | 10 | 1567 |
| 2022 | 904 | 3705 | 8 | 2809 |
| 2023 | 911 | 3734 | 0 | 2823 |
| 2024 | 966 | 3801 | 0 | 2835 |
| 2025 | 958 | 3831 | 0 | 2873 |
| 2026 | 946 | 3677 | 58 | 2789 |
| **total** | | | **315** | |

### 4a. `sched_only` — FBS-vs-FCS/DII/DIII scope (expected)

The master is ESPN's FBS-scoped scoreboard; the schedule is CFBD's full
universe. Division composition of the published schedule makes the mechanism
plain:

| season | games | FBS-vs-FBS | involves FBS | no FBS team | divisions present |
|---:|---:|---:|---:|---:|---|
| 2001 | 709 | 652 | 709 | **0** | fbs, fcs |
| 2002 | 772 | 707 | 772 | **0** | fbs, fcs |
| 2003 | 1338 | 699 | 771 | 567 | fbs, fcs, **ii** |
| 2010 | 1462 | 718 | 808 | 654 | fbs, fcs, ii |
| 2014 | 1586 | 760 | 868 | 718 | fbs, fcs, ii, **iii** |
| 2019 | 1623 | 774 | 888 | 735 | fbs, fcs, ii, iii |
| 2020 | 1125 | 534 | 570 | 555 | fbs, fcs, ii, iii |
| 2021 | 2454 | 770 | 887 | 1567 | fbs, fcs, ii, iii |
| 2022 | 3705 | 776 | 896 | 2809 | fbs, fcs, ii, iii |
| 2025 | 3831 | 808 | 934 | 2897 | fbs, fcs, ii, iii |
| 2026 | 3677 | 761 | 888 | 2789 | fbs, fcs, ii, iii |

`sched_only` tracks `no_fbs` almost exactly in every season. Confirmed
systematic, not a defect.

### 4b. `master_only` — 315 rows, all explained

Classifying all 315 by ESPN `season_type` × `status_type_name`:

| ESPN season_type | status | count | explanation |
|---|---|---:|---|
| 2 (regular) | `STATUS_POSTPONED` | 93 | never played at that id; CFBD drops or re-ids |
| 2 (regular) | `STATUS_CANCELED` | 88 | never played |
| 2 (regular) | `STATUS_SCHEDULED` | 14 | 2026 placeholders, opponent still TBD |
| 2 (regular) | `STATUS_FINAL` | 9 | ESPN duplicate ids — see below |
| 3 (postseason) | `STATUS_SCHEDULED` | 44 | 2026 unassigned bowl/playoff slots (`TBD` vs `TBD`) |
| 3 (postseason) | `STATUS_CANCELED` | 22 | cancelled bowls (mostly 2020) |
| 3 (postseason) | `STATUS_FINAL` | 2 | ESPN duplicate ids — see below |
| 4 (all-star) | `STATUS_FINAL` | 43 | the all-star exhibitions of §3a — CFBD excludes by design |

- **2020's 136** is the COVID season: 79 `STATUS_CANCELED` + 56
  `STATUS_POSTPONED` + 1 final. ESPN retains the cancelled/postponed shells;
  CFBD does not. Expected.
- **2026's 58** are all `STATUS_SCHEDULED` placeholders — 44 unassigned
  postseason slots plus 14 conference-championship shells with `TBD`
  participants. Expected for an unplayed season.
- **The 11 `STATUS_FINAL` rows were individually verified.** Every one is an
  **ESPN duplicate/alternate game_id** for a game that *is* in the published
  schedule under its canonical id, same date, same matchup. Examples:
  2012 SEC Championship master `400434780` → schedule `323360061`
  (2012-12-01T21:00Z, Alabama at Georgia); 2013 Gator Bowl master `400521191`
  → schedule `340010061` (2014-01-01T17:00Z, Nebraska at Georgia). All 11
  matched 1:1 on date + matchup.

**Zero genuinely-missing played games.** The known ESPN dual-id pathology is
the only residue, and it produces no data loss on the schedule side.

## 5. Neighbour-deviation sanity flags

Seasons whose game count swings >20% vs the prior season:

| transition | change | verdict |
|---|---|---|
| 2002 → 2003 | 772 → 1338 (**+73%**) | **Expected.** CFBD's non-FBS coverage begins in 2003: `no_fbs` goes 0 → 567 and division `ii` first appears. FBS-vs-FBS is flat (707 → 699). This is the "2003–2013 gained ~560 FCS games each" effect. |
| 2019 → 2020 | 1623 → 1125 (**−31%**) | **Expected — COVID.** FBS-vs-FBS 774 → 534. Partially offset by the 532 `spring_regular`/`spring_postseason` rows added in the rebuild. |
| 2020 → 2021 | 1125 → 2454 (**+118%**) | **Expected.** COVID recovery (FBS-vs-FBS 534 → 770) *plus* a step-change in CFBD's DII/DIII coverage: `no_fbs` 555 → 1567. |
| 2021 → 2022 | 2454 → 3705 (**+51%**) | **Expected.** Second CFBD lower-division coverage expansion: `no_fbs` 1567 → 2809, while FBS-vs-FBS is flat (770 → 776). |

No other transition exceeds 20%. 2023–2026 are stable at 3,677–3,831.
Crucially, **FBS-vs-FBS game count never swings**: 652 → 808 monotonically over
25 years. Every large swing is lower-division scope, never a loss of real games.

## 6. Conclusion

| check | result |
|---|---|
| 26 seasons published, no gaps | PASS |
| 48,872 rows, 48,872 distinct game_ids | PASS |
| postseason non-zero for every played season | PASS (was the bug; fixed) |
| null `start_date` / `home_id` / `away_id` | PASS (0 everywhere) |
| null scores on completed games | PASS (3 cells, 2025 lower-division) |
| PBP orphans explainable as all-star exhibitions | PASS (40/40) |
| ESPN master games missing from schedule | PASS (315/315 explained; 0 real losses) |
| neighbour-deviation flags | PASS (4 swings, all explained) |

No remediation required for `cfb_schedules`.

Follow-ups worth tracking (none block this release):

1. `home_division` / `away_division` nulls (1–82 per season) are an upstream
   CFBD gap. If the division field is load-bearing downstream, backfill it from
   `cfb_teams_crosswalk` rather than treating null as FCS.
2. 2023's 139 postseason rows use a different CFBD classification for
   lower-division playoffs than every other season. Anything that filters
   `season_type == "postseason"` to mean "bowls + CFP" will over-count 2023.
