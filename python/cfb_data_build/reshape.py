"""Generic JSON-block -> tidy frame reshape (polars port of ``R/_data_utils.R``).

Faithful translation of the R reshape primitives:

* ``dict_to_row`` / ``.play_to_row`` (``R/_data_utils.R:39-46``,
  ``R/espn_cfb_01_pbp_creation.R:9-16``) -- per-cell normalization.
* ``flat_block_df`` (``R/_data_utils.R:49-57``) -- a list-of-flat-dicts block
  to a per-element frame stamped with game identity.
* ``stringify_list_cols`` (``R/_data_utils.R:68-80``) -- JSON-encode surviving
  list cells so parquet/csv share one (string) schema.
* ``bind_games`` (``R/_data_utils.R:60-64``) -- drift-safe vertical union.

R reads the payload with ``jsonlite::fromJSON(simplifyVector = FALSE)``, so every
JSON value arrives as a length-N R list and ``dict_to_row`` applies:

====================  ===================================================
JSON value            cell
====================  ===================================================
``null`` / ``[]``     ``NA`` (-> polars null)
length-1 array        the single element, *unboxed*
length-2+ array       the list, later JSON-encoded to a string
scalar                the scalar
====================  ===================================================

Python's ``json.loads`` already yields native scalars/lists, so
:func:`_norm_cell` reproduces the same rule directly. The R pipeline reshapes
first and stringifies at write time; we fold the JSON-encode into the reshape
so the returned frame is release-ready (output-equivalent, since the union of
keys is identical either way).
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl


def _norm_cell(v: Any) -> Any:
    """Normalize one parsed-JSON value to a parquet-ready scalar or JSON string.

    Mirrors ``dict_to_row`` + the list-column branch of ``stringify_list_cols``:
    ``None``/empty -> ``None``; length-1 list -> its (recursively normalized)
    element; any surviving list/dict -> compact ``auto_unbox`` JSON string. The
    separators/escaping match R ``jsonlite::toJSON(auto_unbox = TRUE)`` byte for
    byte for the flat ESPN structures these blocks carry (verified against the
    ``teamParticipants`` oracle).
    """
    if v is None:
        return None
    if isinstance(v, list):
        if len(v) == 0:
            return None
        if len(v) == 1:
            v = v[0]  # R: length-1 list is unboxed to its single element
    if isinstance(v, (list, dict)):
        # length-2+ list, or a dict (nested object) -> JSON string.
        # NB: R would unbox a 1-key object to its lone value; such bare
        # top-level objects do not occur in the CFB blocks, so encoding the
        # whole object here is parity-safe and the sensible general behavior.
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    return v


def stamp_identity(
    df: pl.DataFrame, game: dict[str, Any], *, week: bool = True
) -> pl.DataFrame:
    """Stamp game identity at the FRAME level (port of the ``df$game_id <- ...`` lines).

    Applied AFTER the row union -- critical for frames with heterogeneous keys
    (e.g. player_box stat categories), where R appends ``game_id`` / ``season``
    *after* the full column union, not interleaved with per-row keys.
    ``with_columns`` overwrites ``game_id`` in place when the block already
    carries it (keeps its position) and appends ``season`` / ``week`` at the end.
    Empty frames pass through unchanged (matches R's 0-row skip).
    """
    if df.height == 0:
        return df
    cols = {
        "game_id": pl.lit(int(game["id"]), dtype=pl.Int64),
        "season": pl.lit(int(game["season"]), dtype=pl.Int64),
    }
    if week and game.get("week") is not None:
        cols["week"] = pl.lit(int(game["week"]), dtype=pl.Int64)
    return df.with_columns(**cols)


def flat_block_frame(
    block: list[dict[str, Any]] | None, game: dict[str, Any]
) -> pl.DataFrame:
    """A list-of-flat-dicts block -> one row per element, stamped with identity.

    Port of ``flat_block_df(block, g)``. Returns an empty frame for a
    missing/empty block (matches R's ``data.frame()``). Column union across rows
    is handled by ``pl.from_dicts`` (== ``data.table::rbindlist(fill = TRUE)``);
    identity columns are stamped at the frame level, then trailed.
    """
    if not block:
        return pl.DataFrame()
    rows = [{k: _norm_cell(v) for k, v in elem.items()} for elem in block]
    return stamp_identity(
        pl.from_dicts(rows, infer_schema_length=None), game, week=True
    )


def bind_games(frames: list[pl.DataFrame | None]) -> pl.DataFrame:
    """Vertically union per-game frames, dropping ``None``/empty (port of ``bind_games``).

    Uses ``how="diagonal_relaxed"`` so frames with differing column sets union by
    name (== ``rbindlist(fill = TRUE, use.names = TRUE)``), with dtype widening
    when a shared column inferred differently across games.
    """
    keep = [f for f in frames if f is not None and f.height > 0]
    if not keep:
        return pl.DataFrame()
    return pl.concat(keep, how="diagonal_relaxed")
