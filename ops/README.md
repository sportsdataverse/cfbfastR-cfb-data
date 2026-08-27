# ops/

Recurring operational tools that are NOT pipeline stages (D-series placement rules).

Stages live in `python/` as numbered `*_creation.py` shims; drivers live in `scripts/` (bash only).

`ops/oneoff/` holds dated one-shots (`YYYYMMDD_<what>.py`); `ops/init/` one-time bootstraps.

_Currently empty — added so a new operational tool has an obvious home other than `scripts/`._
