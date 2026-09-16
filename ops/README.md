# ops/

Recurring operational tools that are NOT pipeline stages (D-series placement rules).

Stages live in `python/` as numbered `*_creation.py` shims; drivers live in `scripts/` (bash only).

`ops/oneoff/` holds dated one-shots (`YYYYMMDD_<what>.py`); `ops/init/` one-time bootstraps.

_Currently empty — added so a new operational tool has an obvious home other than `scripts/`._

## Matchup reference tables (annual, before the season)

`data/cfb_matchup_*.csv` feed the eleven matchup-line side-input columns CFBD
does not publish. Two are rebuilt from their sources, two are imported values.

```sh
# derived -- rerun whenever the coordinator table or a new season lands
PYTHONPATH=python uv run python -m cfb_data_build.matchup_reference refresh \
    --start-season 2004 --end-season <season>
```

* `cfb_matchup_coach_continuity.csv` — derived from `cfb_matchup_coordinators.csv`
  (2014+, one season per source season after the first).
The QB rebuild always starts at 2004: it replaces the file, and its career
columns are cumulative, so a later start would drop the backfill AND undercount
every season it wrote. The command refuses a later start.

* `cfb_matchup_qb_starters.csv` — derived from the ESPN pbp release (2004+):
  the passer with the most attempts. It is the REALIZED starter, so the table
  is only complete once a season has been played, and the columns it feeds are
  post-hoc for that season.
* `cfb_matchup_coordinators.csv` — IMPORTED. Append the new season's
  coordinator names before August; the derived continuity table then rebuilds.
* `cfb_matchup_returning_production.csv`, `cfb_matchup_roster_talent.csv` —
  IMPORTED values that cannot be re-derived (see the module docstring for what
  they are NOT: CFBD's `/player/returning` and `/talent` are different metrics).

A team absent from a table is left null rather than guessed; `match_team_names`
returns the unmatched spellings so a new member shows up as a report, not a
silent hole.

### Season floors

| columns | from | floor |
|---|---|---|
| head coach, tenure, weather, team / venue meta | CFBD | 2004 |
| QB block (5) | ESPN pbp release | 2004 |
| coach continuity (2) | coordinator table (2013+), needs a prior season | 2014 |
| talent | CFBD `/talent` | 2015 |
| returning production (3), roster talent | imported values | 2015 |

Earlier seasons are left null rather than guessed. Extending returning
production, roster talent or coordinator continuity further back needs a
source none of these APIs publish.
