# Per-season parquet convention audit — 2026-08-27

Audit only. **No files were changed by this audit** beyond adding this report.

> **Superseded in this same PR.** This report records the state *before* the
> commit that introduced the committed-parquet policy. The headline finding
> below — that `cfb/` is gitignored wholesale and therefore zero datasets have a
> tracked parquet — was true when the audit ran and is the reason the policy
> changed. `.gitignore` now excludes `cfb/**` and re-admits directories plus
> `cfb/**/parquet/*.parquet`, and 2,254 per-season parquet are committed, so the
> `committed` column in the matrix below reads as the pre-change baseline rather
> than current state. The gap list in section 4 is still live; the headline is not.

Scope: every directory under `cfbfastR-cfb-data/cfb/` and every CFB-related
release tag on `sportsdataverse/sportsdataverse-data`. Release inventory taken
via `gh api repos/sportsdataverse/sportsdataverse-data/releases[/tags/<tag>]`
REST (the GraphQL quota is exhausted; `gh release list/view` was not used).
237 tags total on the repo, 44 CFB-related.

---

## 0. Headline finding — the premise does not hold in this repo

The rule under audit is *"every dataset that backs a release must have a
per-season parquet committed in the repo at `cfb/<dataset>/parquet/<dataset>_<year>.parquet`."*

**In `cfbfastR-cfb-data`, that rule is currently satisfied by exactly zero
datasets — because the entire `cfb/` tree is gitignored.**

```text
git ls-files cfb/                              ->  0
git ls-tree -r origin/main | grep -c '^cfb/'   ->  0
```

`.gitignore` line 11–12:

```gitignore
# Built dataset artifacts live on sportsdataverse-data release tags, not in git.
cfb/
```

That ignore has been in place since `f4fd1c7` ("feat(summaries): season-level
team & player summaries (espn_cfb_15) + presentation (#2)") — i.e. since the
repo's first dataset feature. `origin/main`'s top level contains no `cfb/` at
all. This matches `CLAUDE.md`'s stated model ("Publish dataset releases to
`sportsdataverse/sportsdataverse-data`"), and contradicts the committed-parquet
rule.

The local `cfb/` tree that this audit enumerates is therefore a **local build
cache**, not repo content. Consequences:

- The "gaps" named in the audit brief (`rb_eval`, `snapshots`, `model_pbp`,
  `power_index`) are gaps *in one machine's build cache*, not in the repo or on
  the releases. Three of the four turn out to be non-gaps entirely (§3).
- The convention **is** implemented — in the sibling `cfbfastR-cfb-raw`, which
  commits its `cfb/` tree (41,412 `json/`, 20,695 each of `betting/`,
  `game_rosters/`, `play_participants/`, `power_index/`, `team_box_extra/`,
  518 `recruits/`, 46 `schedules/`, plus `cfb_schedule_master.parquet`).

**This is a policy decision, not a defect to patch.** Before any remediation in
§5 is actioned, someone must decide whether `cfbfastR-cfb-data` is meant to
commit built parquet at all. Committing all 28 datasets would add roughly 1 GB
for `pbp/` alone. The recommendation in §5 is to **ratify the current
release-only model** and correct the rule statement, not to un-ignore `cfb/`.

Everything below is reported against that reality: "local #parq" is
build-cache state, "release" columns are the authoritative published state.

---

## 1. Matrix — local `cfb/` dirs

`stem==dir?` compares the local parquet filename stem to the directory name.
"local gaps" / "release gaps" = missing years inside the observed min–max range.

| dataset dir | `parquet/` dir | committed | local #parq | local years | release tag | release #parq | release years | stem==dir? | local gaps | release gaps |
|---|---|---|---:|---|---|---:|---|---|---|---|
| `adv_defensive` | Y | **no (ignored)** | 22 | 2004-2025 | `espn_cfb_adv_defensive` | 22 | 2004-2025 | Y | none | none |
| `adv_defensive_players` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_defensive_players` | 22 | 2004-2025 | Y | none | none |
| `adv_drives` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_drives` | 22 | 2004-2025 | Y | none | none |
| `adv_passing` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_passing` | 22 | 2004-2025 | Y | none | none |
| `adv_receiving` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_receiving` | 22 | 2004-2025 | Y | none | none |
| `adv_rushing` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_rushing` | 22 | 2004-2025 | Y | none | none |
| `adv_situational` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_situational` | 22 | 2004-2025 | Y | none | none |
| `adv_specialists` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_specialists` | 22 | 2004-2025 | Y | none | none |
| `adv_team` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_team` | 22 | 2004-2025 | Y | none | none |
| `adv_team_gamelog` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_team_gamelog` | 22 | 2004-2025 | Y | none | none |
| `adv_turnover` | Y | no | 22 | 2004-2025 | `espn_cfb_adv_turnover` | 22 | 2004-2025 | Y | none | none |
| `cfb_ratings_weekly` | Y | no | 22 | 2004-2025 | `cfb_ratings_weekly` | 22 | 2004-2025 | Y | none | none |
| `cfb_recruits` | Y | no | 25 | 2002-2026 | `cfb_recruits` | 25 | 2002-2026 | Y | none | none |
| `cfb_returning_production` | Y | no | 21 | 2005-2025 | `cfb_returning_production` | 21 | 2005-2025 | Y | none | none |
| `cfb_team_summaries_weekly` | Y | no | 22 | 2004-2025 | `cfb_team_summaries_weekly` | 22 | 2004-2025 | Y | none | none |
| `cfb_team_talent` | Y | no | 22 | 2005-2026 | `cfb_team_talent` | 22 | 2005-2026 | Y | none | none |
| `crosswalk` | Y | no | 25 | 2014-2025 | `cfb_crosswalk` | 24+1 | 2014-2025 | **N** (see §4.1) | none | none |
| `fpi_weekly` | Y | no | 21 | 2005-2025 | `cfb_fpi_weekly` | 21 | 2005-2025 | **N** — stem `cfb_fpi_weekly` | none | none |
| `model_pbp` | Y | no | **1** | 2024 | `espn_cfb_model_pbp` | 22 | 2004-2025 | Y | n/a | none |
| `passing` | Y | no | 22 | 2004-2025 | `espn_cfb_passing` | 22 | 2004-2025 | **N** — stem `cfb_passing` | none | none |
| `pbp` | Y | no | 22 | 2004-2025 | `espn_cfb_pbp` | 22 | 2004-2025 | LEGACY `play_by_play` (allowed) | none | none |
| `percentiles` | Y | no | 22 | 2004-2025 | `espn_cfb_percentiles` | 22 | 2004-2025 | **N** — stem `cfb_percentiles` | none | none |
| `power_index` | Y | no | **11** | 2015-2025 | `espn_cfb_power_index` | 12 | 2015-2026 | Y | none | none |
| `rb_eval` | **NO** | no | 0 | — | `espn_cfb_model_artifacts` (body) | 0 per-season | — | n/a | n/a | n/a |
| `receiving` | Y | no | 22 | 2004-2025 | `espn_cfb_receiving` | 22 | 2004-2025 | **N** — stem `cfb_receiving` | none | none |
| `rushing` | Y | no | 22 | 2004-2025 | `espn_cfb_rushing` | 22 | 2004-2025 | **N** — stem `cfb_rushing` | none | none |
| `snapshots` | **NO** | no | 0 | — | **(none — unpublishable by design)** | — | — | n/a | n/a | n/a |
| `team_summaries` | Y | no | 22 | 2004-2025 | `espn_cfb_team_summaries` | 22 | 2004-2025 | **N** — stem `cfb_team_summaries` | none | none |

**No release tag has a hole inside its published year range.** Every
`release gaps` cell is `none`.

## 2. Matrix — release tags with **no** local `cfb/` dir

These 17 tags are built by `R/espn_cfb_0N_*.R` (or by `sdv-py` /
`cfb_model_publish`) and published straight to the release; the build output is
not retained in this checkout's `cfb/` tree.

| release tag | assets | parquet stem | parquet years | gaps | notes |
|---|---:|---|---|---|---|
| `cfb_schedules` | 78 | `cfb_schedules` | 2001-2026 (26) | none | the Job A dataset; parquet + rds + csv.gz × 26 |
| `espn_cfb_schedules` | 48 | `cfb_schedule` | 2004-2026 (23) | none | **second, older schedules tag** — see §4.3 |
| `espn_cfb_team_box` | 48 | `team_box` | 2004-2026 (23) | none | stem ≠ tag suffix but self-consistent |
| `espn_cfb_player_box` | 45 | `player_box` | 2004-2025 (22) | none | |
| `espn_cfb_drives` | 45 | `drives` | 2004-2025 (22) | none | |
| `espn_cfb_game_rosters` | 45 | `game_rosters` | 2004-2025 (22) | none | |
| `espn_cfb_betting` | 48 | `betting` | 2004-2026 (23) | none | |
| `espn_cfb_linescores` | 44 | `linescores` | 2004-2025 (22) | none | recent-seasons-only per CLAUDE.md — in fact full 2004+ |
| `espn_cfb_play_participants` | 25 | `play_participants` | 2014-2025 (12) | none | 2014+ only (ESPN participants array start) |
| `espn_cfb_rosters` | 68 | `roster` **and** `rosters` | 2004-2025 (22) | none | **naming violation — §4.2** |
| `cfb_ratings` | 67 | `cfb_ratings` | 2004-2025 (22) | none | + 1 oracle card json |
| `cfb_recruiting_proj` | 11 | `cfb_recruiting_proj` | 2016-2025 (10) | none | + 1 oracle card json |
| `cfbfastR_cfb_pbp` | 51 | `play_by_play` | 2014-2025 (12) | none | legacy cfbfastR-format PBP |
| `cfb_model_artifacts` | 19 | (not per-season) | — | — | 9 `.ubj`, 8 `.json`, 2 `.parquet` |
| `espn_cfb_model_artifacts` | 22 | (not per-season) | — | — | model bundle incl. `rb_eval` GAM output |
| `espn_cfb_injuries` | **0** | — | — | — | **empty tag** — §4.4 |
| `espn_cfb_player_boxscores` | **0** | — | — | — | **empty tag** — §4.4 |
| `espn_cfb_team_boxscores` | **0** | — | — | — | **empty tag** — §4.4 |

## 3. The four "known-missing" items — confirmed and explained

### 3.1 `rb_eval/` has no `parquet/` dir — **correct, not a gap**

`rb_eval` is **not a per-season dataset.** It is the xREPA GAM model output:

```text
rb_eval/  calibration.{csv,parquet}  rush_plays.parquet  rusher_seasons.parquet
          xrepa_calibration.{csv,parquet,png}  xrepa_final.{json,pkl}  xrepa_loso.parquet
```

Whole-corpus artifacts (one GAM fitted across all rushing plays), not
season-sliced. `CLAUDE.md`'s model registry lists it as *"GAM `.pkl` + card →
`espn_cfb_model_artifacts` (per release body)"*, cadence manual. It is produced
by `python -m rb_eval` and referenced nowhere in `R/`. **No `parquet/<yr>` dir
should exist. No action.**

### 3.2 `snapshots/` has no `parquet/` dir — **correct, and publishing is explicitly forbidden**

`snapshots/` is a through-week cut, two levels deeper than the audited shape:

```text
snapshots/through_wk01..wk16/{passing,percentiles,receiving,rushing,team_summaries}/{parquet,csv,rds}/
```

5,075 files. Each leaf *does* have a `parquet/` dir with correctly-named
per-season files — the top-level `snapshots/` simply is not itself a dataset.

It has no release tag by design. `python/cfb_data_build/cli.py:119`:

```python
"--through-week snapshots cannot be published; canonical tags hold season-final builds"
```

**No action.** (If anything, the convention doc should record `snapshots/` as an
explicit second legacy exception alongside `pbp/`.)

### 3.3 `model_pbp/` has only 1 local season — **build-cache staleness, release is complete**

Local: `model_pbp_2024.parquet` only. Release `espn_cfb_model_pbp`: **22
seasons, 2004–2025, no gaps**, stem `model_pbp_<yr>` (matches convention).

So the published dataset is whole; this machine only ever materialised 2024
(`R/espn_cfb_16_model_pbp.R` last ran for that season). The tag additionally
carries two one-off training artifacts (`cfb_model_pbp_2004.parquet`,
`cfb_pbp_train_full_2004.parquet`) that break the single-stem rule — see §4.5.

**No data action.** Cosmetic cleanup only.

### 3.4 `power_index/` has 11 seasons, not 22 — **correct; the dataset starts in 2015**

`espn_cfb_power_index` publishes **2015–2026 (12 seasons), no gaps**. The local
cache has 11 (2015–2025, missing only the just-published 2026).

ESPN's college-football FPI/power-index endpoint has no data before 2015, and
`CLAUDE.md` already documents this: *"`power_index` / `linescores` are
recent-seasons-only."* The framing "11 vs 22 elsewhere" compares against
datasets whose natural start is 2004 — the comparison, not the dataset, is
wrong.

(Note `linescores` is grouped with `power_index` in that CLAUDE.md sentence but
actually publishes the full 2004–2025 — the doc is stale on that half.)

## 4. Naming violations

### 4.1 `cfb_crosswalk` — three stems on one tag (structural, low risk)

`crosswalk/` holds three logically distinct products:

| stem | seasons |
|---|---|
| `cfb_teams_crosswalk_<yr>` | 12 (2014-2025) |
| `cfb_schedule_crosswalk_<yr>` | 12 (2014-2025) |
| `cfb_rosters_crosswalk` | 1, no year |

The directory is `crosswalk/` but no stem is `crosswalk_<yr>`. This is a
deliberate multi-product tag (each has its own loader: `load_cfb_teams_crosswalk`,
`load_cfb_schedule_crosswalk`, plus the season-less rosters crosswalk).
**Recommend documenting as an accepted exception**, not renaming — the loaders
and `config.py` `CFB_ROSTERS_CROSSWALK_URL` are wired to these names.

### 4.2 `espn_cfb_rosters` — `roster_<yr>` vs `rosters_<yr>` split (real violation)

Confirmed exactly as described in the brief:

| stem | years | count |
|---|---|---:|
| `roster_<yr>` | 2004–2022 | 19 |
| `rosters_<yr>` | 2023–2025 | 3 |

Any consumer globbing one stem silently loses the other era.
**Already owned by another agent (`cfb-rosters-pipeline`) — noted here only,
no action taken by this audit.**

### 4.3 Two schedules tags coexist: `cfb_schedules` and `espn_cfb_schedules`

| tag | stem | years | assets |
|---|---|---|---:|
| `cfb_schedules` | `cfb_schedules_<yr>` | 2001-2026 (26) | 78 |
| `espn_cfb_schedules` | `cfb_schedule_<yr>` (singular) | 2004-2026 (23) | 48 |

`sportsdataverse/config.py:15` points `CFB_TEAM_SCHEDULE_URL` at the
**`cfb_schedules`** release — the correct, wider, freshly-republished one.
`R/espn_cfb_10_schedules_creation.R:42` and `R/releases_init.R:25` still
target **`espn_cfb_schedules`**, and `python/cfb_data_build/config.py:137`
does too. So the R/py producer writes one tag while the Python loader reads
the other.

**This is the highest-value finding in Job B.** It is not a data defect today
(Job A verified `cfb_schedules` is complete and correct), but the producer and
the consumer are pointed at different tags, so the next producer run will
refresh the tag nobody reads.

### 4.4 Three empty release tags

`espn_cfb_injuries`, `espn_cfb_player_boxscores`, `espn_cfb_team_boxscores`
each have **0 assets**. The latter two are superseded by
`espn_cfb_player_box` / `espn_cfb_team_box` (45/48 assets, populated).
`espn_cfb_injuries` is created by `R/releases_init.R` and fed by
`R/espn_cfb_14_injuries_creation.R`, but has never been populated
(CLAUDE.md: *"Datasets NOT produced: `officials`, betting `propbets`"* — injuries
is not listed but is likewise unproduced).

### 4.5 `espn_cfb_model_pbp` carries two non-conforming extras

Alongside the 22 conforming `model_pbp_<yr>.parquet`:
`cfb_model_pbp_2004.parquet` and `cfb_pbp_train_full_2004.parquet` — one-off
training artifacts parked on a data tag. They belong on
`espn_cfb_model_artifacts`.

### 4.6 Stem ≠ directory name in 6 committed-cache datasets

`fpi_weekly`→`cfb_fpi_weekly`, `passing`→`cfb_passing`,
`percentiles`→`cfb_percentiles`, `receiving`→`cfb_receiving`,
`rushing`→`cfb_rushing`, `team_summaries`→`cfb_team_summaries`.

Each adds a `cfb_` prefix the directory lacks. These are **self-consistent
local-and-release** (the release uses the same prefixed stem), so nothing is
broken — but they violate the stated `stem == dirname` rule as literally as
`pbp`→`play_by_play` does. If the rule is kept, either rename the six
directories to `cfb_*` (zero release churn, cache-only) or record all seven as
accepted legacy stems.

---

## 5. Prioritized gap list + remediation steps

**Nothing below has been executed.** Ordered by risk.

### P1 — Resolve the two-schedules-tags split (§4.3)

Producer writes `espn_cfb_schedules`; `sdv-py` reads `cfb_schedules`. Decide one.

Recommended (keep `cfb_schedules`, the one Job A verified and the loader reads):

```sh
# 1. confirm the divergence
gh api repos/sportsdataverse/sportsdataverse-data/releases/tags/cfb_schedules      --jq '.assets|length'
gh api repos/sportsdataverse/sportsdataverse-data/releases/tags/espn_cfb_schedules --jq '.assets|length'

# 2. repoint the producers (3 sites)
#    R/espn_cfb_10_schedules_creation.R:42   "espn_cfb_schedules" -> "cfb_schedules"
#    R/releases_init.R:25                    key rename
#    python/cfb_data_build/config.py:137     "espn_cfb_schedules" -> "cfb_schedules"

# 3. verify stem: producer emits cfb_schedule_<yr>, the live tag uses cfb_schedules_<yr>
grep -n 'cfb_schedule' R/espn_cfb_10_schedules_creation.R

# 4. after repointing, one rebuild+publish for the current season, then diff
Rscript R/espn_cfb_10_schedules_creation.R -s 2026 -e 2026
```

Then deprecate `espn_cfb_schedules` (leave assets, edit body to point at the
successor). Do **not** delete — `cfbfastR`'s R loader may still read it; check
`cfbfastR/R/cfb_load*.R` first.

### P2 — Ratify or reject the committed-parquet rule (§0)

The rule as written is unimplemented and, at ~1 GB for `pbp/` alone, probably
should stay that way.

```sh
# evidence for the decision
cd cfbfastR-cfb-data
du -sh cfb/                       # local build-cache size
git ls-files cfb/ | wc -l         # 0
grep -n -A2 'release tags, not in git' .gitignore
```

Recommended action: **keep `cfb/` gitignored** and amend `CLAUDE.md`'s
"Inputs / outputs" section to state that `cfb/{dataset}/…` is a local build
cache and the release tag is the artifact of record. If instead the rule is to
be enforced, it needs a per-dataset opt-in list plus Git LFS — not a blanket
un-ignore.

### P3 — Fix the `roster_` / `rosters_` stem split (§4.2)

**Owned by `cfb-rosters-pipeline`. Do not action from here.** Recorded for
completeness: 19 `roster_2004..2022` vs 3 `rosters_2023..2025` on
`espn_cfb_rosters`.

### P4 — Retire the three empty tags (§4.4)

```sh
for t in espn_cfb_player_boxscores espn_cfb_team_boxscores; do
  gh api repos/sportsdataverse/sportsdataverse-data/releases/tags/$t --jq '.id,.body'
done
# then PATCH each body to name its successor (espn_cfb_player_box / espn_cfb_team_box)
# gh api -X PATCH repos/sportsdataverse/sportsdataverse-data/releases/<id> -f body='DEPRECATED — see espn_cfb_<x>_box'
```

For `espn_cfb_injuries`: either populate it (`R/espn_cfb_14_injuries_creation.R`
exists and is wired) or add injuries to `CLAUDE.md`'s "Datasets NOT produced"
list. Prefer the latter unless there is a consumer.

### P5 — Move the two training artifacts off `espn_cfb_model_pbp` (§4.5)

```sh
gh api repos/sportsdataverse/sportsdataverse-data/releases/tags/espn_cfb_model_pbp \
  --jq '.assets[] | select(.name|test("^cfb_(model_pbp|pbp_train_full)_2004")) | "\(.id) \(.name)"'
# download, re-upload to espn_cfb_model_artifacts, then delete from the data tag
```

Low urgency — they are inert extras, not corrupt data.

### P6 — Reconcile the six `cfb_`-prefixed stems (§4.6)

Cache-only rename, zero release churn:

```sh
cd cfbfastR-cfb-data/cfb
for d in fpi_weekly passing percentiles receiving rushing team_summaries; do
  echo "$d -> cfb_$d"   # then update the writer in python/cfb_data_build/config.py
done
```

Alternative (cheaper, recommended): amend the convention doc to list
`pbp/→play_by_play_<yr>`, the six `cfb_`-prefixed stems, `crosswalk/`'s three
stems, and `snapshots/`'s nesting as accepted exceptions.

### P7 — Refresh the stale local build cache (operator convenience only)

Not a repo defect; only affects local work off this checkout.

```sh
Rscript R/espn_cfb_16_model_pbp.R -s 2004 -e 2025   # model_pbp: 1 -> 22 local seasons
Rscript R/espn_cfb_12_power_index_creation.R -s 2026 -e 2026   # power_index: +2026
```

### P8 — Correct the stale `linescores` note in CLAUDE.md (§3.4)

`CLAUDE.md` says *"`power_index` / `linescores` are recent-seasons-only."*
True for `power_index` (2015+), false for `linescores`
(`espn_cfb_linescores` publishes 2004–2025, 22 seasons, no gaps). One-line doc
fix.

---

## 6. Summary counts

| | count |
|---|---:|
| release tags on `sportsdataverse-data` | 237 |
| CFB-related tags | 44 |
| CFB tags with per-season parquet | 33 |
| CFB tags with a hole inside their year range | **0** |
| CFB tags that are empty | 3 |
| local `cfb/` dataset dirs | 28 |
| local dirs with a `parquet/` dir | 26 |
| local dirs *committed to git* | **0** (whole tree gitignored) |
| naming violations found | 5 (§4.1, §4.2, §4.3, §4.5, §4.6) |
| "known-missing" items that were real gaps | **0 of 4** |
