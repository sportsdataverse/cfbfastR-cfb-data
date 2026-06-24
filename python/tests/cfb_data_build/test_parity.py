"""Pilot parity tests: the Python ``cfb_data_build`` builders must reproduce the
R producers' released output byte-for-value.

Oracle provenance
-----------------
The golden frames in ``fixtures/oracle_*_401628455.parquet`` were produced by
running the *actual* R reshape on the committed game fixture
``fixtures/final_401628455.json`` (game 401628455, 2024 wk 1), under R 4.5.3
with ``cfbfastR`` installed (so ``conform_pbp`` applied the canonical schema):

* ``play_participants`` = ``reshape_play_participants(g)`` -> ``stringify_list_cols``
  (``R/espn_cfb_05`` + ``R/_data_utils.R``), 56 cols x 158 rows.
* ``pbp`` = ``conform_pbp(reshape_pbp(g))`` -> ``stringify_list_cols``
  (``R/espn_cfb_01`` -> ``cfbfastR:::.pbp_apply_output_schema(output="default")``),
  371 cols x 169 rows.

Parity bar: column names AND order exact; row count exact; values type-normalized
on the oracle dtype (numeric/bool -> Float64 tol 1e-9; string exact, incl. the
JSON-encoded ``teamParticipants`` which is byte-identical R<->py). See
``_parity_helpers.assert_frame_parity``.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from tests.cfb_data_build._parity_helpers import assert_frame_parity

FIX = Path(__file__).parent / "fixtures"
GID = 401628455


def _load_game() -> dict:
    return json.loads((FIX / f"final_{GID}.json").read_text(encoding="utf-8"))


def test_play_participants_parity() -> None:
    from cfb_data_build.reshape import flat_block_frame

    g = _load_game()
    py = flat_block_frame(g["play_participants"], g)
    oracle = pl.read_parquet(FIX / f"oracle_play_participants_{GID}.parquet")
    assert_frame_parity(py, oracle, name="play_participants")


def test_pbp_parity() -> None:
    from cfb_data_build.pbp import build_pbp_frame

    g = _load_game()
    py = build_pbp_frame(g)  # default output tier
    oracle = pl.read_parquet(FIX / f"oracle_pbp_{GID}.parquet")
    assert_frame_parity(py, oracle, name="pbp")
