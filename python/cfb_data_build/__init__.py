"""cfb_data_build -- Python builders for the public CFB datasets.

The Python half of the cfb-data dual-write: it reproduces the R producers
(``R/_data_utils.R`` + ``R/espn_cfb_*_creation.R``) that reshape cfb-raw
``final.json`` game payloads into the tidy per-season frames released under the
``espn_cfb_*`` piggyback tags.

Design (mirrors the R side):

* :mod:`cfb_data_build.reshape` -- the generic, dataset-agnostic engine:
  flatten one JSON block to one row per element, stamp game identity, and
  JSON-encode list cells (port of ``dict_to_row`` / ``flat_block_df`` /
  ``stringify_list_cols``).
* :mod:`cfb_data_build.pbp` -- the only dataset with custom output handling:
  a faithful port of cfbfastR's ``.pbp_apply_output_schema`` (column tiering +
  ordering) that the R ``conform_pbp`` delegates to.

Parity bar: byte-for-value equality against the R-released parquet
(``python/tests/cfb_data_build/test_parity.py``).
"""

from __future__ import annotations

from cfb_data_build.pbp import apply_pbp_output_schema, build_pbp_frame
from cfb_data_build.reshape import bind_games, flat_block_frame
from cfb_data_build.team_summaries import build_team_summaries

__all__ = [
    "flat_block_frame",
    "bind_games",
    "build_pbp_frame",
    "apply_pbp_output_schema",
    "build_team_summaries",
]
