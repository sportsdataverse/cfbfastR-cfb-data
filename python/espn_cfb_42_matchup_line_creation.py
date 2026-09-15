"""Builder: the per-game matchup line (``cfb_matchup_line``).

Thin entrypoint over ``cfb_data_build.matchup_build``. One row per FBS-vs-FBS
game (bowls dropped, CFP kept), 268 columns: the home / away as-of features of
stage 41 (week 1 from the prior season's full-season values), the prior-season
``prev_*`` block, pace, CFBD pregame ELO with the opponent-ELO rolls, consensus
betting lines, game meta, weather, team / venue meta, talent and the head
coach with tenure (the source's proprietary side inputs stay null). Reads TWO
seasons of the ``cfbfastR_cfb_pbp`` release (the prior season feeds ``prev_*``),
CFBD ``/games`` for both, and CFBD ``/lines``, ``/games/weather``, ``/teams``,
``/talent``, ``/coaches`` for the season (needs ``CFBD_API_KEY``).

Example:
    One season::

        uv run python python/espn_cfb_42_matchup_line_creation.py -s 2025 -e 2025 --base ../cfb
"""

from __future__ import annotations

from _shim import run_dataset

DATASET = "matchup_line"

if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
