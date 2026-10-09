"""Paper Index per game (stage 67) and the season luck columns on ``team_summaries``.

The model is ``sportsdataverse.paper_index`` (Game on Paper's Paper Index; the
weights were fitted there and are not refitted here). This module adds no formula.
It owns the producer parts only:

* :func:`paper_index_games_table`: the sdv-py rows for a pbp frame, unchanged, plus
  ``paper_index_span``. One row per team per scored game, ids Int64 as the released
  pbp carries them (a float or string id raises).
* :func:`attach_luck`: the sdv-py ``deserved_wins`` roll-up of those rows joined
  onto a ``team_summaries`` frame: ``deserved_wins``, ``luck_wins``, ``luck_z``,
  their ranks, ``paper_index_games_n`` and ``paper_index_span``.
* :func:`build_paper_index_games`: the stage-67 builder, one season of the
  committed ``cfb/pbp`` under ``base``.

Games counted are the ones the sdv-py function scores: completed, with a winner,
both sides at 20 or more scrimmage snaps, regular season and postseason, FBS and
FCS opponents alike. That is wider than ``valid_games`` (FBS vs FBS only), so
``paper_index_games_n`` is published beside the sums.

LEAKAGE: every column in :data:`LUCK_COLUMNS` is derived from game OUTCOMES (the
weights were fitted on who won, and ``luck_wins`` subtracts from the real win
count). None of them may be a model feature. The feature and metric lists that
are built by name pattern exclude them by name
(``cfb_higher_models.data.feature_columns``, ``league_averages.metric_columns``);
``tests/cfb_data_build/test_paper_index.py`` fails if either exclusion is removed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
from sportsdataverse.paper_index import (
    GAMES_SCHEMA,
    HOLDOUT_SEASONS,
    PBP_COLUMNS,
    TRAIN_SEASONS,
    deserved_wins,
    paper_index_games,
)

from cfb_data_build.team_summaries import _rank

LEAGUE = "cfb"

#: the released pbp's id columns; all Int64 there, and published unchanged
_ID_COLUMNS = ("game_id", "pos_team_id", "homeTeamId", "awayTeamId")

#: how sdv-py's keep-floor UserWarning starts (a regex, matched at the start of
#: the message); stage 67 turns it into an error. The stage test replays the real
#: warning, so a reworded message upstream fails there instead of passing here.
_KEEP_FLOOR_WARNING = r"paper_index_games: inputs computable for"

#: ``cfb_paper_index_games``: the sdv-py schema plus the in-sample label
OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64,
    "team_id": pl.Int64,
    **GAMES_SCHEMA,
    "paper_index_span": pl.Utf8,
}

#: every column :func:`attach_luck` adds to ``team_summaries`` (and so to
#: ``team_summaries_weekly``), in output order. All outcome-derived: see LEAKAGE.
LUCK_COLUMNS: tuple[str, ...] = (
    "deserved_wins",
    "luck_wins",
    "luck_z",
    "luck_wins_rank",
    "luck_z_rank",
    "paper_index_games_n",
    "paper_index_span",
)


def _span(season: pl.Expr) -> pl.Expr:
    """``train`` / ``holdout`` / ``out_of_span`` from the sdv-py fit spans (read at call time)."""
    train, holdout = TRAIN_SEASONS[LEAGUE], HOLDOUT_SEASONS[LEAGUE]
    return (
        pl.when(season.is_between(*train))
        .then(pl.lit("train"))
        .when(season.is_between(*holdout))
        .then(pl.lit("holdout"))
        .otherwise(pl.lit("out_of_span"))
    )


def paper_index_span(season: int) -> str:
    """Where ``season`` sits against the Paper Index fit.

    ``train``: the season's shares were part of the fit (in-sample). ``holdout``:
    scored out of sample at fit time, though not fully clean (the EP model behind
    the EPA, success and explosiveness inputs was trained on seasons that include
    the holdout). ``out_of_span``: never seen by the fit, never evaluated.
    """
    return pl.select(_span(pl.lit(season))).item()


def paper_index_games_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """sdv-py ``paper_index_games`` rows for ``pbp``, plus ``paper_index_span``.

    Raises:
        TypeError: an id column is not Int64 (the released pbp's dtype).
        ValueError: ``pbp`` lacks a column the sdv-py function reads.
    """
    for c in _ID_COLUMNS:
        if c in pbp.columns and pbp.schema[c] != pl.Int64:
            raise TypeError(
                f"{c} is {pbp.schema[c]}: the released pbp carries Int64 ids and "
                "cfb_paper_index_games publishes them unchanged"
            )
    games = paper_index_games(pbp.select([c for c in PBP_COLUMNS if c in pbp.columns]), LEAGUE)
    return games.with_columns(paper_index_span=_span(pl.col("season"))).select(list(OUTPUT_SCHEMA))


def attach_luck(
    team_data: pl.DataFrame,
    games: pl.DataFrame,
    season: int,
    *,
    through_week: int | None = None,
) -> pl.DataFrame:
    """Join the season's deserved wins and luck onto a ``team_summaries`` frame.

    ``games`` is :func:`paper_index_games_table` output for the games the frame
    covers (a ``through_week`` snapshot passes only that snapshot's games). The
    sums are sdv-py ``deserved_wins``; ``luck_wins + deserved_wins`` is the
    team's wins over the ``paper_index_games_n`` games counted. A team with no
    scored game gets ``paper_index_games_n = 0`` and null luck. Ranks are over
    the frame's teams that have a value, 1 = the luckiest.

    Raises:
        ValueError: ``games`` carries another season; or, on a full-season build
            (``through_week`` is None), no team in the frame matched a scored
            game. An early snapshot can be empty; a season cannot, so that is an
            id that stopped matching, not a team without games.
        TypeError: a float ``team_id`` on either side.
    """
    luck = deserved_wins(games)
    stray = luck.filter(pl.col("season") != season)
    if stray.height:
        raise ValueError(
            f"paper index games for {season} carry other seasons: "
            f"{sorted(stray['season'].unique().to_list())}"
        )
    key = team_data.schema["team_id"]
    src = luck.schema["team_id"]
    if not src.is_integer() or not (key == pl.Utf8 or key.is_integer()):
        raise TypeError(
            f"paper index team_id is {src}, team_summaries team_id is {key}: "
            "ids join through an integer or string parse, never a float"
        )
    luck = luck.select(
        pl.col("team_id").cast(key),
        "deserved_wins",
        "luck_wins",
        "luck_z",
        paper_index_games_n=pl.col("games"),
    )
    assert team_data.schema["team_id"] == luck.schema["team_id"]
    joined = team_data.join(luck, on="team_id", how="left", validate="1:1")
    if through_week is None and joined.height and joined["luck_wins"].count() == 0:
        # the left join and the zero fill below would publish this as a table
        # of zero counts and null luck, with nothing saying the ids missed
        raise ValueError(
            f"{season}: none of the {joined.height} team_summaries rows matched a "
            f"Paper Index game ({luck.height} teams scored): the team_id formats "
            "no longer agree"
        )
    return joined.with_columns(
        paper_index_games_n=pl.col("paper_index_games_n").fill_null(0),
        luck_wins_rank=_rank("luck_wins", descending=True),
        luck_z_rank=_rank("luck_z", descending=True),
        paper_index_span=pl.lit(paper_index_span(season)),
    ).select(*team_data.columns, *LUCK_COLUMNS)


def build_paper_index_games(season: int, *, base: str = "cfb") -> pl.DataFrame:
    """``cfb_paper_index_games`` for one season of the committed pbp under ``base``.

    Reads only ``{base}/pbp/parquet`` (no release fallback). Returns the empty
    :data:`OUTPUT_SCHEMA` frame when the season's pbp is missing.

    Raises:
        ValueError: sdv-py's keep-floor warning fired (inputs computable for
            fewer than 98% of the decided games). A warning there; here it
            would publish a season with games missing, so the stage fails. No
            committed season 2004-2026 trips it.
    """
    path = Path(base) / "pbp" / "parquet" / f"play_by_play_{season}.parquet"
    if not path.is_file():
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    pbp = pl.read_parquet(path, columns=list(PBP_COLUMNS))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=_KEEP_FLOOR_WARNING, category=UserWarning)
        try:
            return paper_index_games_table(pbp)
        except UserWarning as feed_loss:
            raise ValueError(f"paper_index_games {season}: {feed_loss}") from feed_loss
